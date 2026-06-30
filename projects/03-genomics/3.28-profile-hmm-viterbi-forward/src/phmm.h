// ===========================================================================
// src/phmm.h  --  The shared __host__ __device__ profile-HMM core (CPU/GPU parity)
// ---------------------------------------------------------------------------
// Project 3.28 : Profile HMM (Viterbi / Forward)
//
// THE SINGLE MOST IMPORTANT FILE FOR CORRECTNESS
//   This header holds the *per-cell math* of the two dynamic-programming
//   recurrences this project teaches -- Viterbi (max-sum) and Forward
//   (log-sum-exp) -- as `__host__ __device__` inline functions. Because the CPU
//   reference (reference_cpu.cpp) and the GPU kernel (kernels.cu) BOTH call the
//   exact same inline functions, they execute byte-for-byte identical floating-
//   point operations. That is the HD-core idiom from docs/PATTERNS.md §2, and it
//   is why we can verify GPU==CPU to ~machine precision instead of a loose
//   tolerance.
//
//   CONSTRAINT: this file is included by BOTH nvcc (kernels.cu) AND the plain
//   host C++ compiler (reference_cpu.cpp). So it must contain NO CUDA-only
//   constructs (no __global__, no <<< >>>, no device-only intrinsics). The
//   PHMM_HD macro below expands to `__host__ __device__` under nvcc and to
//   nothing under the host compiler, so the same source compiles in both worlds.
//
// READ THIS AFTER: the README "What this computes" section.
// READ THIS BEFORE: reference_cpu.cpp (loops these), kernels.cu (one thread per
//   sequence loops these), and main.cu (orchestrates).  The full derivation is
//   in ../THEORY.md.
// ===========================================================================
#pragma once

#include <cmath>      // std::log, std::exp, std::log1p, INFINITY  (host + device)
#include <cstdint>

// ---------------------------------------------------------------------------
// PHMM_HD : the host/device decorator switch (PATTERNS.md §2).
//   * Under nvcc, __CUDACC__ is defined, so functions become callable from BOTH
//     host and device code.
//   * Under the host compiler, __CUDACC__ is NOT defined, so the decorators must
//     vanish (the host compiler does not understand __host__/__device__).
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define PHMM_HD __host__ __device__
#else
#define PHMM_HD
#endif

// ---------------------------------------------------------------------------
// THE ALPHABET
//   We model PROTEIN sequences over the standard 20 amino acids. A residue is
//   stored as a small integer code 0..19 (see aa_code() in reference_cpu.cpp).
//   Using a compact integer code (instead of a char) lets us index the profile's
//   emission table directly: emission for (match state k, residue a) lives at
//   match_emit[k*ALPHA + a].
// ---------------------------------------------------------------------------
constexpr int ALPHA = 20;   // number of amino-acid symbols

// ---------------------------------------------------------------------------
// NEGATIVE INFINITY in log space.
//   All probabilities are stored as natural logs (log p). An impossible event
//   (p = 0) has log p = -infinity. We use a large finite sentinel rather than
//   true -INF so that arithmetic like (-INF) + (finite) stays well-defined and
//   identical on host and device (true -inf + finite is fine too, but a finite
//   sentinel also keeps max() and the log-sum-exp trick numerically clean and
//   avoids NaN from (-inf) - (-inf)). -1e30 is far below any real log-prob here.
// ---------------------------------------------------------------------------
constexpr double LOG_ZERO = -1.0e30;

// ===========================================================================
// THE PROFILE-HMM MODEL  (a simplified Plan-7 architecture)
// ---------------------------------------------------------------------------
// A profile HMM turns a multiple-sequence alignment of a protein family into a
// position-specific scoring model. The real HMMER "Plan 7" has Match (M),
// Insert (I) and Delete (D) states per column plus flanking N/C/J states. For a
// teaching version we keep the three core per-column states M, I, D -- enough to
// show the full 2-D DP and the three-way recurrence -- and fold the flanking
// states into simple begin/end transitions. THEORY §7 explains exactly what the
// production model adds.
//
// For a profile of length M (number of match columns, indexed k = 1..M) we store:
//   * match_emit[k][a]  = log P(emit residue a | match state k)
//   * insert_emit[a]    = log P(emit residue a | any insert state)  (position-
//                          independent here, as in HMMER's default)
//   * the 7 Plan-7 core transition logs PER column k (see TransLog below).
//
// We keep the model FLAT (plain arrays) and copy it whole into GPU CONSTANT
// memory: every thread (every database sequence) reads the SAME profile but
// never writes it, so the constant cache can broadcast it warp-wide -- the same
// trick as the query fingerprint in flagship 1.12.
// ===========================================================================

// The seven Plan-7 core transitions out of column k. Stored as natural logs.
// Naming: t_XY = log P(state X at column k -> state Y at column k+1 or k).
struct TransLog {
    double mm;   // M_k -> M_{k+1}   : match to next match (the "stay on profile" path)
    double mi;   // M_k -> I_k       : match to insert (an insertion after column k)
    double md;   // M_k -> D_{k+1}   : match to delete (skip the next column)
    double im;   // I_k -> M_{k+1}   : insert back to match
    double ii;   // I_k -> I_k       : insert to insert (extend the insertion)
    double dm;   // D_k -> M_{k+1}   : delete back to match
    double dd;   // D_k -> D_{k+1}   : delete to delete (extend the deletion)
};

// The whole profile, flattened for easy host<->device copy.
//   M               : number of match columns (profile length)
//   match_emit      : [ (M+1) * ALPHA ]  log-emission, row k is column k (k=1..M; row 0 unused)
//   insert_emit     : [ ALPHA ]          log-emission for insert states
//   trans           : [ (M+1) ]          TransLog per column (index 0..M; see kernels/ref for use)
//   We size match_emit and trans with (M+1) rows so we can use the natural
//   1-based column index k without subtracting 1 everywhere (row 0 is a pad).
//   MAX_M caps the model so the constant-memory image is a fixed compile-time
//   size (constant memory cannot be variably sized).
struct ProfileHMM {
    int      M;                                   // profile length (match columns)
    double   match_emit[ (/*MAX_M+1*/ 65) * ALPHA ];
    double   insert_emit[ ALPHA ];
    TransLog trans[ 65 ];                         // index 0..MAX_M
};

// The compile-time cap on profile length. 64 match columns is plenty for a
// teaching demo (real Pfam profiles run to hundreds; THEORY §7). Kept in sync
// with the array sizes above (MAX_M+1 == 65).
constexpr int MAX_M = 64;

// The compile-time cap on a database sequence length. The GPU keeps one DP
// "column" of size (M+1) in registers/local memory per thread and rolls it
// across the sequence, so sequence length is bounded only by how long we let a
// thread loop; we cap it so host and device agree on buffer sizes.
constexpr int MAX_L = 256;

// ---------------------------------------------------------------------------
// log_sum_exp(a, b) : the numerically-stable log of (exp(a) + exp(b)).
//   THE FORWARD ALGORITHM works in probability space but we store logs to avoid
//   underflow (a product of hundreds of small probabilities underflows a double
//   to 0). Adding two probabilities in log space is the "log-sum-exp" trick:
//       log(e^a + e^b) = max(a,b) + log(1 + e^(-|a-b|))
//   Factoring out the larger term keeps the exp() argument <= 0, so it never
//   overflows; log1p(x)=log(1+x) is accurate even when x is tiny. This single
//   helper is the ONLY difference between Forward (uses this) and Viterbi (uses
//   max). Both CPU and GPU call THIS function, so their sums match exactly.
// ---------------------------------------------------------------------------
PHMM_HD inline double log_sum_exp(double a, double b) {
    // If either term is the -inf sentinel, the sum is just the other term.
    if (a <= LOG_ZERO) return b;
    if (b <= LOG_ZERO) return a;
    // Stable form: pull out the max so exp() sees a non-positive argument.
    const double hi = (a > b) ? a : b;
    const double lo = (a > b) ? b : a;
    return hi + log1p(exp(lo - hi));   // log1p(exp(<=0)) is safe and accurate
}

// max2 / max3 : plain maxima used by the VITERBI recurrence (max-sum). We define
// our own (rather than std::max) so the exact same code path runs on host and
// device with no <algorithm> dependency inside device code.
PHMM_HD inline double max2(double a, double b) { return (a > b) ? a : b; }
PHMM_HD inline double max3(double a, double b, double c) { return max2(max2(a, b), c); }

// ===========================================================================
// THE PER-CELL RECURRENCES
// ---------------------------------------------------------------------------
// Both Viterbi and Forward fill a dynamic-programming lattice with three planes
// M[i][k], I[i][k], D[i][k] where i = sequence position (1..L) and k = profile
// column (1..M). The cell value is the score (log-prob) of the best (Viterbi) /
// total (Forward) way to emit the first i residues and end in that state at
// column k.
//
//   Match recurrence (emit residue x_i from match column k):
//       M[i][k] = e_M(k, x_i) + COMBINE( M[i-1][k-1] + t_mm(k-1),
//                                        I[i-1][k-1] + t_im(k-1),
//                                        D[i-1][k-1] + t_dm(k-1) )
//   Insert recurrence (emit residue x_i from insert state k):
//       I[i][k] = e_I(x_i)    + COMBINE( M[i-1][k] + t_mi(k),
//                                        I[i-1][k] + t_ii(k) )
//   Delete recurrence (emit NOTHING; a silent state, so no i decrement):
//       D[i][k] =               COMBINE( M[i][k-1] + t_md(k-1),
//                                        D[i][k-1] + t_dd(k-1) )
//
// COMBINE = max3/max2 for Viterbi, log_sum_exp(...) for Forward. The two
// algorithms are *the same recurrence* with a different combine operator -- the
// single most important idea in this project, and why we factor the combine out.
//
// Rather than three separate functions we expose the COMBINE primitives above
// and let kernels.cu / reference_cpu.cpp write the recurrence loop once each.
// The emission lookups below are shared so indexing is identical on both sides.
// ===========================================================================

// e_M(profile, k, a) : log emission of residue code a from MATCH column k.
//   Row k of the flat match_emit table; guarded so an out-of-range residue
//   (shouldn't happen for codes 0..19) returns LOG_ZERO instead of reading OOB.
PHMM_HD inline double emit_match(const ProfileHMM& p, int k, int a) {
    return p.match_emit[k * ALPHA + a];
}

// e_I(profile, a) : log emission of residue code a from an INSERT state
//   (position-independent in this teaching model, matching HMMER's default
//   where insert emissions are tied to the background distribution).
PHMM_HD inline double emit_insert(const ProfileHMM& p, int a) {
    return p.insert_emit[a];
}
