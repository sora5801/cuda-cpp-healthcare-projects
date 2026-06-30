// ===========================================================================
// src/pairhmm_core.h  --  The ONE TRUE PairHMM math, shared by CPU and GPU
// ---------------------------------------------------------------------------
// Project 3.3 : Variant Calling Acceleration
//
// WHY THIS HEADER EXISTS  (PATTERNS.md §2 -- the HD-macro idiom)
//   The per-cell recurrence of the PairHMM forward algorithm is written exactly
//   ONCE here, decorated `__host__ __device__`. The CPU reference
//   (reference_cpu.cpp) and the GPU kernel (kernels.cu) both call these same
//   inline functions, so they execute *byte-for-byte identical* floating-point
//   operations. That turns verification from "approximately equal" into "equal
//   to a few ULP", which is the whole point of having a reference.
//
//   Keep this header CUDA-type-free (no __global__, no <cuda_runtime.h>) so the
//   plain host compiler (cl.exe / g++) can include it via reference_cpu.cpp,
//   while nvcc includes it via kernels.cu. The PAIRHMM_HD macro expands to the
//   CUDA decorators under nvcc and to nothing under the host compiler.
//
// WHAT THE PairHMM COMPUTES  (the science, in one paragraph)
//   Germline variant calling (GATK HaplotypeCaller, NVIDIA Parabricks) asks:
//   given a sequencing READ, how likely is it that the read was produced by a
//   candidate HAPLOTYPE (a hypothesised local genome sequence), accounting for
//   sequencing errors? The answer is P(read | haplotype), a marginal likelihood
//   summed over every way the read could be aligned to the haplotype under a
//   pair Hidden Markov Model with three hidden states:
//       M  (Match/mismatch) -- read base aligns to a haplotype base
//       I  (Insertion)      -- read base is an extra base not in the haplotype
//       D  (Deletion)       -- a haplotype base is skipped (missing from read)
//   The forward algorithm fills a dynamic-programming table over (read position
//   i, haplotype position j) and sums all alignment paths in O(R*H) time. This
//   single computation is the dominant runtime cost of variant calling, so it
//   is exactly what production GPU callers parallelise. See ../THEORY.md.
//
// READ THIS AFTER: ../THEORY.md (the derivation). Used by: reference_cpu.cpp,
// kernels.cu.
// ===========================================================================
#pragma once

#include <cmath>     // std::pow, std::log10 (host); device uses the same intrinsics
#include <cstdint>   // fixed-width integer types for bases/qualities

// --- The host/device portability macro (PATTERNS.md §2) --------------------
// Under nvcc (__CUDACC__ defined) every function below is compiled for BOTH the
// host and the device. Under the plain C++ compiler the decorators vanish, so
// the very same source compiles as ordinary host code for reference_cpu.cpp.
#ifdef __CUDACC__
#define PAIRHMM_HD __host__ __device__
#else
#define PAIRHMM_HD
#endif

// ---------------------------------------------------------------------------
// Transition probabilities of the pair-HMM.
//   In the full GATK model the gap-open / gap-continue probabilities come from
//   per-base insertion/deletion quality scores. For this teaching version we use
//   a SINGLE fixed set of transition probabilities (a common simplification --
//   GATK itself falls back to constants when GCP tables are flat). They are the
//   probabilities of moving between the three hidden states from one read base
//   to the next. Documented as constants here so the recurrence below reads
//   cleanly; THEORY.md derives where the real per-base values come from.
//
//   delta  = P(open a gap)            -- M -> I  or  M -> D
//   epsilon= P(extend a gap)          -- I -> I  or  D -> D
//   So:  P(M->M) = 1 - 2*delta,  P(I->M) = P(D->M) = 1 - epsilon.
// ---------------------------------------------------------------------------
struct PairHmmParams {
    double delta;    // gap-open probability (per read base)
    double epsilon;  // gap-extend probability

    // Derived transition probabilities (computed once, on the host, then copied
    // to the device verbatim so both sides use identical bit patterns).
    double m_to_m;   // 1 - 2*delta
    double m_to_gap; // delta            (used for both M->I and M->D)
    double gap_to_m; // 1 - epsilon      (used for both I->M and D->M)
    double gap_to_gap; // epsilon        (used for both I->I and D->D)
};

// Fill the derived fields from delta/epsilon. Called once on the host; the whole
// struct (a handful of doubles) is then handed to both code paths.
PAIRHMM_HD inline void pairhmm_finalize_params(PairHmmParams& p) {
    p.m_to_m     = 1.0 - 2.0 * p.delta;
    p.m_to_gap   = p.delta;
    p.gap_to_m   = 1.0 - p.epsilon;
    p.gap_to_gap = p.epsilon;
}

// ---------------------------------------------------------------------------
// base_emission_prob: P(observe read base `r` | true haplotype base `h`, quality q).
//   A Phred quality score Q encodes an error probability e = 10^(-Q/10). If the
//   read base MATCHES the haplotype base, the base was read correctly with prob
//   (1 - e). If it MISMATCHES, this particular wrong base was emitted with prob
//   e/3 (the error could have gone to any of the 3 other bases, uniformly).
//   This is exactly GATK's emission model for the Match state.
//
//   r, h : encoded bases (0..3 = A,C,G,T; any other value treated as mismatch).
//   q    : Phred base-quality (clamped to a sane floor so e never hits 0 or 1).
//   Returns the emission probability in (0,1).
// ---------------------------------------------------------------------------
PAIRHMM_HD inline double base_emission_prob(uint8_t r, uint8_t h, int q) {
    // Clamp quality into [1, 60]: Q=0 would give e=1 (no information) and very
    // large Q underflows; real pipelines clamp similarly.
    if (q < 1)  q = 1;
    if (q > 60) q = 60;
    // e = 10^(-q/10). pow() is identical on host and device (both IEEE-754).
    const double e = pow(10.0, -static_cast<double>(q) / 10.0);
    // Matching bases: probability of a correct read. Mismatch: e spread over the
    // 3 alternative bases.
    return (r == h) ? (1.0 - e) : (e / 3.0);
}

// ---------------------------------------------------------------------------
// PairHmmCell: the three forward probabilities for one (i, j) DP cell.
//   m[i][j] = sum of all alignment paths ending with read base i aligned to
//             haplotype base j in a MATCH/MISMATCH state, and similarly for the
//             Insertion (i) and Deletion (d) states. Stored together because the
//             recurrence for each depends on neighbours of all three.
// ---------------------------------------------------------------------------
struct PairHmmCell {
    double m;  // Match state forward probability
    double i;  // Insertion state forward probability
    double d;  // Deletion state forward probability
};

// ---------------------------------------------------------------------------
// pairhmm_step: THE ONE TRUE RECURRENCE. Compute cell (read base ri, hap base hj)
// from its three already-computed neighbours. This is the inner loop body of the
// forward algorithm, factored out so CPU and GPU run identical arithmetic.
//
//   diag = cell (i-1, j-1)   -- up-left  (a match advances both sequences)
//   up   = cell (i-1, j)     -- up       (an insertion advances the read only)
//   left = cell (i,   j-1)   -- left      (a deletion advances the haplotype only)
//
//   Recurrence (GATK forward, linear-probability space):
//     M[i][j] = emission * ( P(M->M)*diag.m + P(I->M)*diag.i + P(D->M)*diag.d )
//     I[i][j] =             P(M->I)*up.m   + P(I->I)*up.i
//     D[i][j] =             P(M->D)*left.m + P(D->D)*left.d
//   where `emission` = base_emission_prob(read[i], hap[j], qual[i]).
//
//   Returns the filled cell. Pure function of its inputs -> deterministic.
// ---------------------------------------------------------------------------
PAIRHMM_HD inline PairHmmCell pairhmm_step(const PairHmmParams& p,
                                           uint8_t read_base, uint8_t hap_base, int qual,
                                           PairHmmCell diag, PairHmmCell up, PairHmmCell left) {
    const double emission = base_emission_prob(read_base, hap_base, qual);
    PairHmmCell out;
    // Match: came from any of the three states one step back along the diagonal,
    // then emitted this (base, base) pair.
    out.m = emission * (p.m_to_m * diag.m + p.gap_to_m * diag.i + p.gap_to_m * diag.d);
    // Insertion: opened or extended a gap in the haplotype (read got ahead).
    out.i = p.m_to_gap * up.m + p.gap_to_gap * up.i;
    // Deletion: opened or extended a gap in the read (haplotype got ahead).
    out.d = p.m_to_gap * left.m + p.gap_to_gap * left.d;
    return out;
}

// ---------------------------------------------------------------------------
// Encoding helpers shared by data loading on both sides. A,C,G,T -> 0,1,2,3.
//   Any other character maps to 4 ("N"/unknown), which never equals a real base
//   so it always takes the mismatch branch in base_emission_prob.
// ---------------------------------------------------------------------------
PAIRHMM_HD inline uint8_t encode_base(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default:            return 4;  // N / unknown
    }
}
