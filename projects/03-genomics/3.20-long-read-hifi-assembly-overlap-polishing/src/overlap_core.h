// ===========================================================================
// src/overlap_core.h  --  The shared __host__ __device__ overlap math
// ---------------------------------------------------------------------------
// Project 3.20 : Long-Read HiFi Assembly Overlap & Polishing
//
// WHY THIS HEADER EXISTS  (PATTERNS.md sec 2 -- the most useful idiom)
//   The *per-pair physics* of read overlap -- how a k-mer is hashed, how a
//   minimiser window is reduced, how two reads' shared seeds are chained into
//   a collinear overlap score -- is written ONCE here, as `OVL_HD` inline
//   functions. The CPU reference (reference_cpu.cpp, compiled by the host C++
//   compiler) and the GPU kernel (kernels.cu, compiled by nvcc) both #include
//   this file and call the SAME functions. Because the chaining score is built
//   entirely from INTEGER additions and comparisons, the two sides agree
//   BIT-FOR-BIT (tolerance == 0; see ../THEORY.md "How we verify correctness").
//
//   Keep this header free of CUDA-only constructs (no __global__, no <<<>>>,
//   no kernel-only types) so the host compiler can include it too. The OVL_HD
//   macro expands to `__host__ __device__` under nvcc and to nothing under the
//   host compiler -- the one trick that lets the same source run on both.
//
// THE PIPELINE THIS HEADER IMPLEMENTS (read ../THEORY.md for the full "why"):
//   1. encode_base()   : A/C/G/T -> 2-bit code (and a reject for ambiguous N).
//   2. kmer hashing    : a length-K window of bases -> a 2K-bit integer, taken
//                        CANONICAL (min of the forward strand and its reverse
//                        complement) so a read and its reverse read seed alike.
//   3. minimiser pick  : within each window of W consecutive k-mers, keep only
//                        the one with the smallest hash -- a sparse, strand-
//                        symmetric, deletion-robust subsample of all k-mers.
//      (Steps 1-3 run in scripts/make_synthetic.py at data-gen time AND can be
//       recomputed here; the loader hands us the already-extracted minimisers,
//       so the parity-critical code at run time is the CHAINING below.)
//   4. chain scoring   : given the seed anchors two reads share (positions that
//                        carry the same minimiser hash), find the best COLLINEAR
//                        chain -- the longest run of anchors whose query- and
//                        target-coordinates both increase -- and return its
//                        integer score. That score is the overlap strength.
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh, kernels.cu.
// ===========================================================================
#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// OVL_HD: the host/device decorator switch (PATTERNS.md sec 2).
//   * Under nvcc (__CUDACC__ defined) every function below is compiled for
//     BOTH the CPU and the GPU, so kernels.cu can call them from a thread.
//   * Under the plain host compiler the decorators do not exist, so we expand
//     OVL_HD to nothing and reference_cpu.cpp gets ordinary inline functions.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define OVL_HD __host__ __device__
#else
#define OVL_HD
#endif

// ---------------------------------------------------------------------------
// Compile-time problem parameters. They are constants (not runtime args) so the
// inner loops unroll and the on-thread scratch arrays have fixed size -- which
// is what lets each GPU thread own one read-pair with no dynamic allocation.
// ---------------------------------------------------------------------------

// K  : k-mer length in bases. 15 is a typical minimiser k-mer size for HiFi
//      overlap (minimap2 uses k=15..19); 2*K = 30 bits fits a uint32_t hash.
constexpr int OVL_K = 15;

// W  : minimiser window, in k-mers. One minimiser is kept per W consecutive
//      k-mers, so on average ~ 2/(W+1) of all k-mers survive (the (w,k)-minimiser
//      density result). W = 5 gives a comfortably sparse seed set for teaching.
constexpr int OVL_W = 5;

// MAX_ANCHORS : the on-thread cap on how many shared seeds a single read pair
//      may chain. Long real reads can share thousands of minimisers; for this
//      teaching demo the synthetic reads are short, so 64 anchors is plenty and
//      bounds each thread's local scratch (64 ints -> registers/local memory).
constexpr int OVL_MAX_ANCHORS = 64;

// Chaining score weights (INTEGER, so CPU and GPU sums are identical):
//   MATCH_AWARD : points for extending a chain by one more collinear anchor.
//   GAP_PENALTY_SHIFT : the gap cost is (coordinate drift >> this), i.e. a cheap
//                       integer approximation of "penalise non-collinear jumps".
//   MAX_GAP    : anchors farther apart than this (in either read) cannot chain
//                directly -- it bounds how much indel a single link may absorb.
constexpr int OVL_MATCH_AWARD       = 1;   // +1 per chained anchor (chain length)
constexpr int OVL_GAP_PENALTY_SHIFT = 4;   // penalty = |drift| >> 4  (integer)
constexpr int OVL_MAX_GAP           = 500; // max coordinate gap a link may span

// ---------------------------------------------------------------------------
// ovl_encode_base: map an ASCII base to its 2-bit code, or 0xFF for "ambiguous".
//   A=0  C=1  G=2  T=3  (so the complement of a code c is simply 3 - c == c ^ 3).
//   Lowercase is accepted. Anything else (N, gaps, junk) returns 0xFF, which the
//   k-mer builder treats as "this window is invalid, skip it".
//   Used only at data-generation/encode time; kept here so the encoding is
//   documented in one place alongside the hashing it feeds.
// ---------------------------------------------------------------------------
OVL_HD inline uint8_t ovl_encode_base(char b) {
    switch (b) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default:            return 0xFF;   // ambiguous / non-ACGT
    }
}

// ---------------------------------------------------------------------------
// ovl_canonical_kmer_hash: hash a length-OVL_K window given its forward 2-bit
//   code packing `fwd` and its reverse-complement packing `rev`.
//   A k-mer and its reverse complement describe the SAME piece of double-
//   stranded DNA, so we hash whichever packing is numerically smaller -- the
//   "canonical" k-mer. This makes seeding strand-symmetric: a read overlapping
//   the reverse complement of another read still shares minimisers with it.
//   We then run the canonical code through a fast integer mix (a splitmix-style
//   finalizer) so that minimiser selection by "smallest hash" is not biased
//   toward poly-A (all-zero) k-mers. The mix is deterministic and identical on
//   CPU and GPU.
// ---------------------------------------------------------------------------
OVL_HD inline uint32_t ovl_canonical_kmer_hash(uint32_t fwd, uint32_t rev) {
    uint32_t canon = (fwd < rev) ? fwd : rev;   // strand-symmetric representative
    // splitmix32 finalizer: a bijective avalanche so close k-mers hash far apart.
    uint32_t x = canon;
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

// ---------------------------------------------------------------------------
// ovl_chain_link_score: the per-LINK contribution when a chain hops from anchor
//   `b` (earlier) to anchor `a` (later). Both anchors are shared seeds, each a
//   pair (query position, target position). For the hop to be a valid collinear
//   extension, BOTH coordinates must increase (a is "down-right" of b) and the
//   drift in the two reads must be similar (an overlap is a near-diagonal band).
//
//   Returns the integer reward for the link, or a large negative sentinel
//   (OVL_REJECT) if the link is illegal. The reward is
//       MATCH_AWARD  -  (|dq - dt| >> GAP_PENALTY_SHIFT)
//   where dq, dt are the query/target coordinate gaps: a perfectly collinear
//   step (dq == dt) costs nothing; an indel of size g costs ~ g/16 points.
//   Everything is integer -> the same value on both processors (bit-identical).
//
//   qa,ta : query/target position of the LATER anchor a
//   qb,tb : query/target position of the EARLIER anchor b
// ---------------------------------------------------------------------------
constexpr int OVL_REJECT = -1000000;   // "this link is not allowed"

OVL_HD inline int ovl_chain_link_score(int qa, int ta, int qb, int tb) {
    const int dq = qa - qb;            // advance along the query read
    const int dt = ta - tb;           // advance along the target read
    // Must move strictly forward in BOTH reads (collinear, no backtracking).
    if (dq <= 0 || dt <= 0) return OVL_REJECT;
    // Reject links that jump too far in either read (not a single overlap band).
    if (dq > OVL_MAX_GAP || dt > OVL_MAX_GAP) return OVL_REJECT;
    // Diagonal drift = how much the two coordinate gaps disagree (an indel).
    int drift = dq - dt;
    if (drift < 0) drift = -drift;     // integer abs, branch-light
    const int penalty = drift >> OVL_GAP_PENALTY_SHIFT;   // integer gap cost
    return OVL_MATCH_AWARD - penalty;  // may be <= 0 for a sloppy link
}

// ---------------------------------------------------------------------------
// ovl_chain_dp: the O(A^2) collinear chaining dynamic program, shared verbatim
//   by the CPU reference and the GPU kernel so their scores match bit-for-bit.
//
//   Inputs are A shared-seed anchors as two parallel arrays:
//     aq[k], at[k] = (query position, target position) of anchor k.
//   The anchors are assumed ordered by query position ascending (the caller
//   builds them that way), so a valid chain links a LATER anchor a to an EARLIER
//   anchor b (b < a) only when ovl_chain_link_score allows it. The recurrence is
//       f[a] = MATCH_AWARD + max( 0, max over b<a of ( f[b] - MATCH_AWARD + link(b,a) ) )
//   simplified here to the equivalent f[a] = max(MATCH_AWARD, max_b f[b]+link(b,a)),
//   and the answer is max_a f[a] -- the strongest collinear chain.
//
//   `f` is caller-provided scratch of length >= n (so this function allocates
//   nothing and is safe to call from one GPU thread). Returns the best score, or
//   0 when there are no anchors.
//
//   STRAND NOTE: to score a REVERSE-strand overlap, the caller negates the target
//   positions before calling (a reverse overlap is collinear in (q, -t)); see
//   ovl_chain_best_both_strands below.
// ---------------------------------------------------------------------------
OVL_HD inline int ovl_chain_dp(const int* aq, const int* at, int n, int* f) {
    int best = 0;
    for (int a = 0; a < n; ++a) {
        int fa = OVL_MATCH_AWARD;                 // a fresh chain starting at a
        for (int b = 0; b < a; ++b) {
            const int link = ovl_chain_link_score(aq[a], at[a], aq[b], at[b]);
            if (link == OVL_REJECT) continue;     // illegal extension
            const int cand = f[b] + link;         // extend b's chain by a
            if (cand > fa) fa = cand;             // best predecessor wins
        }
        f[a] = fa;
        if (fa > best) best = fa;                 // global best chain
    }
    return best;
}

// ---------------------------------------------------------------------------
// ovl_chain_best_both_strands: run the chaining DP in BOTH orientations and
//   return the larger score. Real overlappers must consider both strands: a read
//   may overlap another read's FORWARD copy (query and target advance together,
//   a +1 diagonal) OR its REVERSE-COMPLEMENT (as one read advances the other
//   retreats, a -1 anti-diagonal). Canonical minimiser hashing makes both kinds
//   of overlap SHARE seeds; this function makes both kinds CHAIN:
//     * forward  : chain on (aq, at) as given.
//     * reverse  : chain on (aq, -at) -- negating target coords turns the
//                  anti-diagonal into a collinear one the same DP can score.
//   `f`, `neg` are caller-provided scratch of length >= n. Returns max(fwd, rev).
//   This is the one function main()'s per-pair score really calls.
// ---------------------------------------------------------------------------
OVL_HD inline int ovl_chain_best_both_strands(const int* aq, const int* at, int n,
                                              int* f, int* neg) {
    const int fwd = ovl_chain_dp(aq, at, n, f);   // same-strand overlap
    for (int k = 0; k < n; ++k) neg[k] = -at[k];  // flip target axis for reverse
    const int rev = ovl_chain_dp(aq, neg, n, f);  // reverse-strand overlap
    return (fwd > rev) ? fwd : rev;
}
