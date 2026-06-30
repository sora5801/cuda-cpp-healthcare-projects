// ===========================================================================
// src/blast_core.h  --  The ONE shared scoring core (CPU/GPU byte-for-byte)
// ---------------------------------------------------------------------------
// Project 3.7 : BLAST-Style Homology Search
//
// WHY THIS HEADER EXISTS  (PATTERNS.md sec 2 -- the shared __host__ __device__ core)
//   The single most important idiom in this repo: put the *per-element math*
//   that the CPU reference and the GPU kernel must agree on into ONE header,
//   marked `__host__ __device__`, so both sides run the EXACT same code. Here
//   that math is:
//       * residue encoding (amino-acid char -> 0..23 index), and
//       * gapless X-drop extension scoring of a seed hit.
//   Because every operation here is INTEGER (BLOSUM62 scores are integers, the
//   k-mer match test is exact), the CPU and GPU produce BIT-IDENTICAL results
//   -- so main.cu can verify with an EXACT tolerance of 0 (PATTERNS.md sec 4).
//
//   IMPORTANT: keep this header free of CUDA-ONLY constructs (no __global__, no
//   __constant__, no <cuda_runtime.h>). It is compiled BOTH by the host C++
//   compiler (for reference_cpu.cpp) AND by nvcc (for kernels.cu). The HD macro
//   below expands to `__host__ __device__` only under nvcc, and to nothing for
//   the plain host compiler -- so the very same source lines compile on both.
//
// THE SCIENCE IN ONE PARAGRAPH  (full version in ../THEORY.md)
//   Homology search asks: which database (DB) sequences are evolutionarily
//   related to my query? BLAST's answer is "seed-filter-extend":
//     (1) SEED   -- find short exact word (k-mer) matches between query and DB.
//     (2) EXTEND -- from each seed, walk left and right along the diagonal,
//                   adding a BLOSUM62 substitution score per aligned residue,
//                   and remember the highest score seen (the maximal-scoring
//                   ungapped segment = an HSP, High-scoring Segment Pair).
//                   Stop a direction when the running score drops X below the
//                   best-so-far (the "X-drop" rule) -- this is what makes BLAST
//                   fast: it abandons hopeless extensions early.
//   The DB sequence's homology score is the best HSP score over all its seeds.
//   This is the embarrassingly-parallel scan the catalog highlights: each DB
//   sequence is INDEPENDENT, so we give each one its own GPU thread.
//
// READ THIS BEFORE: reference_cpu.h, kernels.cuh. The GPU mapping is in THEORY.
// ===========================================================================
#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// HD: the host/device decorator macro (PATTERNS.md sec 2).
//   * Under nvcc, __CUDACC__ is defined, so HD == "__host__ __device__":
//     the compiler emits BOTH a CPU version and a GPU version of the function.
//   * Under the plain host compiler those keywords do not exist, so HD expands
//     to nothing and the function is an ordinary inline host function.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ---------------------------------------------------------------------------
// The amino-acid alphabet.
//   BLAST scores PROTEIN sequences (20 standard amino acids) plus the ambiguity
//   codes B (Asx), Z (Glx), X (any), and '*' (stop) -- 24 symbols total, the
//   classic BLOSUM62 row/column order. We encode each residue as an index
//   0..23 so the substitution score is a simple 24x24 table lookup.
//
//   ALPHA  : the canonical ordering (index i <-> ALPHA[i]).
//   N_ALPHA: 24, the matrix dimension.
//   Any character not in ALPHA (lowercase, digits, gaps) maps to X (= "any"),
//   which BLOSUM62 scores conservatively (mostly -1). This makes the encoder
//   total: every input byte yields a valid 0..23 index.
// ---------------------------------------------------------------------------
constexpr int  N_ALPHA = 24;
constexpr char ALPHA[N_ALPHA + 1] = "ARNDCQEGHILKMFPSTWYVBZX*";
constexpr int  IDX_X = 22;   // index of 'X' (the "any residue" fallback)

// encode_residue: map one ASCII byte to its 0..23 alphabet index.
//   Implemented as a straight linear scan of ALPHA so it is dependency-free and
//   identical on host and device (no lookup table to keep in sync). 24 compares
//   is trivial next to the alignment work. Unknown bytes -> IDX_X.
//   `c` is taken as int so callers can pass an already-uppercased char.
HD inline int encode_residue(int c) {
    // Uppercase ASCII letters in place (so 'a'..'z' match 'A'..'Z'). Branchless
    // on host and device; the device compiler turns it into a predicated sub.
    if (c >= 'a' && c <= 'z') c -= ('a' - 'A');
    for (int i = 0; i < N_ALPHA; ++i) {
        if (ALPHA[i] == c) return i;   // found the canonical symbol
    }
    return IDX_X;                       // not a known residue -> treat as "any"
}

// ---------------------------------------------------------------------------
// SeqView: a non-owning window into an already-encoded sequence buffer.
//   Both the query and each DB sequence live as flat arrays of int8 residue
//   indices (0..23). A SeqView is just {pointer, length}; it carries no storage,
//   so it is cheap to pass into __host__ __device__ functions. The DB is stored
//   as one big concatenated buffer plus per-sequence (offset,length) so the GPU
//   sees a single coalesced array (see reference_cpu.h SequenceDB).
// ---------------------------------------------------------------------------
struct SeqView {
    const int8_t* data;   // pointer to the first residue index
    int           len;    // number of residues
};

// ---------------------------------------------------------------------------
// blosum_at: read score(a,b) from a flat 24x24 BLOSUM62 matrix.
//   `mat` is row-major [N_ALPHA*N_ALPHA]; a,b are encoded residue indices.
//   We pass the matrix as a parameter (rather than a global) so this function
//   stays pure and usable from host code; on the GPU the SAME values live in
//   constant memory and are passed in as a device pointer (see kernels.cu).
//   The score is the log-odds of substitution a<->b in homologous proteins:
//   positive => more likely than chance (conservative swap or identity),
//   negative => disfavoured. It is an INTEGER, which is the key to exactness.
// ---------------------------------------------------------------------------
HD inline int blosum_at(const int8_t* mat, int a, int b) {
    return mat[a * N_ALPHA + b];
}

// ---------------------------------------------------------------------------
// gapless_xdrop: THE scoring core. Given one seed hit -- query position qpos
// aligned to DB position dpos (so they sit on the same diagonal) -- extend the
// ungapped alignment left and right, BLOSUM-scoring each aligned residue pair,
// and return the maximal segment score (the HSP score) found around the seed.
//
//   This is exactly BLAST's ungapped extension. "Gapless" means we never insert
//   or delete a residue: query[qpos+t] is always compared to db[dpos+t] for the
//   same offset t, so both walk in lock-step (the seed fixes the diagonal).
//
//   The X-DROP rule: while extending in one direction we track the running
//   score `run` and the best score `best_dir` seen so far in that direction.
//   The instant run falls more than `xdrop` BELOW best_dir, we stop -- the
//   alignment has decayed past recovery, so continuing only wastes work. This
//   early-exit is the whole reason seed-extend is fast versus full DP.
//
//   PARAMETERS
//     q,d    : SeqViews of the query and this DB sequence (encoded indices).
//     qpos   : start column in the query  (0..q.len-1), the seed's query anchor.
//     dpos   : start column in the DB seq  (0..d.len-1), the seed's DB anchor.
//     k      : seed length; the k exact-match residues at [qpos,dpos) are the
//              guaranteed core of the HSP and are scored first.
//     mat    : flat 24x24 BLOSUM62 matrix (see blosum_at).
//     xdrop  : the X-drop threshold (a positive integer; larger => extend more).
//   RETURNS  : the best ungapped segment score around this seed (>= the seed's
//              own score, which is always positive for an exact match).
//
//   COMPLEXITY: O(extension length) per seed, bounded by the sequence length;
//   in practice the X-drop stops it early. No allocation, no recursion -> this
//   compiles to a tight loop on both CPU and GPU.
//
//   Called by BOTH reference_cpu.cpp (host loop over seeds) and kernels.cu (one
//   GPU thread per DB sequence loops over its seeds) -> identical results.
// ---------------------------------------------------------------------------
HD inline int gapless_xdrop(SeqView q, SeqView d,
                            int qpos, int dpos, int k,
                            const int8_t* mat, int xdrop) {
    // (1) Score the seed core itself: the k residues that matched exactly.
    //     They are identical, so each contributes BLOSUM(a,a) (always > 0 for
    //     real residues). This seed score is our starting `best`.
    int seed_score = 0;
    for (int t = 0; t < k; ++t) {
        int a = q.data[qpos + t];
        int b = d.data[dpos + t];
        seed_score += blosum_at(mat, a, b);
    }

    // (2) Extend RIGHT, starting just past the seed. `run` is the score of the
    //     segment [seed .. current]; `best_right` is the best run seen. We add
    //     one aligned pair per step and apply the X-drop stop.
    int best_right = 0;   // best ADDED score to the right of the seed (>= 0)
    int run = 0;          // running added score to the right
    {
        int t = k;        // offset past the seed core
        while (qpos + t < q.len && dpos + t < d.len) {
            run += blosum_at(mat, q.data[qpos + t], d.data[dpos + t]);
            if (run > best_right) best_right = run;          // new high-water mark
            if (best_right - run > xdrop) break;             // X-drop: give up right
            ++t;
        }
    }

    // (3) Extend LEFT, starting just before the seed, walking toward index 0.
    int best_left = 0;    // best ADDED score to the left of the seed (>= 0)
    {
        int run2 = 0;
        int t = 1;        // offset before the seed core
        while (qpos - t >= 0 && dpos - t >= 0) {
            run2 += blosum_at(mat, q.data[qpos - t], d.data[dpos - t]);
            if (run2 > best_left) best_left = run2;
            if (best_left - run2 > xdrop) break;             // X-drop: give up left
            ++t;
        }
    }

    // (4) The HSP score is the seed core plus the best left and right extensions.
    //     best_left/best_right are >= 0 (we only keep extensions that helped),
    //     so the result is never worse than the seed alone.
    return seed_score + best_left + best_right;
}
