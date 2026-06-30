// ===========================================================================
// src/pseudoalign.h  --  Shared (host + device) RNA-seq EM primitives
// ---------------------------------------------------------------------------
// Project 3.22 : RNA-seq Quantification / Pseudo-alignment
//
// WHAT THIS PROJECT COMPUTES
//   Given millions of sequencing READS and a reference TRANSCRIPTOME, estimate
//   how many reads came from each transcript -- i.e. each transcript's ABUNDANCE.
//   "Pseudo-alignment" (kallisto, Salmon) skips slow base-level alignment: it
//   hashes each read's k-mers, looks up which transcripts are COMPATIBLE, and
//   collapses reads that hit the *same set* of transcripts into one EQUIVALENCE
//   CLASS (ec). All we keep is, per ec: (a) how many reads landed in it, and
//   (b) the set of transcript ids it is compatible with. The abundances are then
//   recovered by an EXPECTATION-MAXIMISATION (EM) loop over the ecs.
//
//   This file is the per-ec MATH, written once as __host__ __device__ inline
//   functions (PSA_HD) so the CPU reference (reference_cpu.cpp, host compiler)
//   and the GPU kernel (kernels.cu, nvcc) run BYTE-FOR-BYTE identical arithmetic.
//   That parity is what lets us VERIFY the GPU result against the CPU exactly.
//
// THE EM IN ONE PARAGRAPH (see THEORY.md for the full derivation)
//   Unknowns: rho[t] = the fraction of all reads that originate from transcript
//   t (the abundances; they sum to 1). A read in ec e could have come from any
//   transcript in e. EM alternates:
//     E-STEP  : for ec e with members {t in e}, split its `count[e]` reads among
//               its members in proportion to each member's CURRENT "weight"
//               w_t = rho[t] / efflen[t]  (length-normalised: a longer transcript
//               yields more reads at the same molar abundance, so we divide it
//               out). The expected reads transcript t gets from ec e is
//                   count[e] * w_t / (sum over members s of w_s).
//     M-STEP  : new fractional read count alpha[t] = sum over all ecs of those
//               contributions; renormalise alpha to get the next rho.
//   Iterate to convergence. This is the classic kallisto/RSEM mixture-model EM.
//
// WHY A GPU
//   Real runs have 10^5-10^7 ecs and tens of EM iterations. Each EM iteration is
//   a sparse matrix-vector style sweep: every ec independently reads the current
//   rho of its few members and scatters contributions back. One GPU thread per
//   ec makes the E-step embarrassingly parallel; the scatter into per-transcript
//   accumulators is an ATOMIC REDUCTION. (cuSPARSE could express this as a sparse
//   GEMV -- see THEORY.md "Where this sits in the real world".)
//
// DETERMINISM TRICK (same idea as projects 5.01 and 11.09)
//   Float atomicAdd is order-dependent (non-associative) -> non-reproducible, and
//   it would NOT match the CPU bit-for-bit. So the M-step accumulates each
//   contribution in FIXED-POINT integers (atomicAdd on unsigned long long), which
//   commute -> the GPU result is reproducible AND equals the CPU result exactly.
//
//   Keep this header free of CUDA-only constructs (no __global__, no kernels) so
//   the plain host compiler can include it from reference_cpu.cpp.
//
// READ THIS AFTER: nothing (start here), then reference_cpu.h, then kernels.cuh.
// ===========================================================================
#pragma once

#include <cstdint>   // std::uint32_t, std::int32_t

// PSA_HD expands to "__host__ __device__" under nvcc and to nothing under the
// host compiler, so the very same inline functions compile in both worlds.
#ifdef __CUDACC__
#define PSA_HD __host__ __device__
#else
#define PSA_HD
#endif

// ---------------------------------------------------------------------------
// Fixed-point scale for the deterministic M-step accumulation.
//   Each EM iteration distributes exactly `total_reads` units of "expected read
//   count" across the transcripts. We store those fractional counts as integers
//   scaled by PSA_SCALE. 2^24 ~ 1.6e7 gives ~7 significant digits of resolution
//   on a single contribution; the largest possible accumulated value is
//   total_reads * PSA_SCALE, which for our samples (and for millions of reads)
//   stays far below the 1.8e19 ceiling of unsigned long long.
// ---------------------------------------------------------------------------
// constexpr (not plain `static const`): a constexpr is usable in BOTH host and
// device code under nvcc, whereas a namespace-scope `static const double` has only
// host storage and triggers "identifier undefined in device code".
constexpr double PSA_SCALE = 16777216.0;        // 2^24, as a double for clean math

// Quantize a non-negative fractional read count to fixed-point integer units.
//   `v` is expected to be >= 0 (it is an expected read count). We add 0.5 before
//   truncating so the rounding is to-nearest (not toward zero); rounding the same
//   way on host and device keeps the two results identical.
PSA_HD inline unsigned long long psa_to_fixed(double v) {
    return static_cast<unsigned long long>(v * PSA_SCALE + 0.5);
}

// Convert a fixed-point accumulator back to a floating-point read count.
PSA_HD inline double psa_from_fixed(unsigned long long q) {
    return static_cast<double>(q) / PSA_SCALE;
}

// ---------------------------------------------------------------------------
// psa_weight: the length-normalised weight of transcript t under abundances rho.
//   w_t = rho[t] / effLen[t]. A read is more likely to come from a transcript
//   that is both more ABUNDANT (larger rho) and LONGER (more places a read can
//   start) -- but we are estimating molecular abundance, so we divide the length
//   back out. effLen is the "effective length" (transcript length minus mean
//   fragment length + 1); we guard against a non-positive length so a degenerate
//   transcript contributes zero weight instead of dividing by zero.
// ---------------------------------------------------------------------------
PSA_HD inline double psa_weight(double rho_t, double eff_len_t) {
    return (eff_len_t > 0.0) ? (rho_t / eff_len_t) : 0.0;
}

// ---------------------------------------------------------------------------
// psa_ec_contributions: the E-step for ONE equivalence class.
//   Given the ec's read `count`, its `k` member transcript ids in `members`, the
//   current abundances `rho` and effective lengths `eff_len`, compute how many of
//   the ec's reads each member transcript expects to receive, and write those k
//   numbers into `out_contrib` (a scratch array of length >= k supplied by the
//   caller). The split is proportional to each member's weight w_t:
//       out_contrib[j] = count * w_{members[j]} / (sum_j w_{members[j]}).
//   If every member currently has zero weight (e.g. all rho == 0 at iteration 0
//   is avoided by uniform init, but be safe), the reads are spread UNIFORMLY so
//   no count is lost. This function is the heart shared by CPU and GPU; both call
//   it per ec so their numbers match exactly.
//
//   Returns nothing; results land in out_contrib[0..k-1].
// ---------------------------------------------------------------------------
PSA_HD inline void psa_ec_contributions(double count,
                                        const std::int32_t* members, int k,
                                        const double* rho, const double* eff_len,
                                        double* out_contrib) {
    // First pass: each member's weight, and the running total of weights.
    double denom = 0.0;
    for (int j = 0; j < k; ++j) {
        const std::int32_t t = members[j];
        const double w = psa_weight(rho[t], eff_len[t]);
        out_contrib[j] = w;          // stash the weight; normalised in pass two
        denom += w;
    }
    // Second pass: turn weights into expected read counts.
    if (denom > 0.0) {
        const double inv = count / denom;
        for (int j = 0; j < k; ++j) out_contrib[j] *= inv;   // w * (count/denom)
    } else {
        // Pathological all-zero-weight ec: distribute the reads evenly.
        const double share = (k > 0) ? (count / static_cast<double>(k)) : 0.0;
        for (int j = 0; j < k; ++j) out_contrib[j] = share;
    }
}

// Small upper bound on members-per-ec, used to size a fixed per-thread scratch
// array in the GPU kernel (avoids dynamic allocation inside a kernel). Real ecs
// in kallisto are almost always small (a read maps to a handful of isoforms of
// one gene); we cap the teaching sample well under this. The loader rejects any
// ec larger than this so we never silently overflow the scratch.
constexpr int PSA_MAX_EC_SIZE = 32;
