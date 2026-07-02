// ===========================================================================
// src/attn_core.h  --  The ONE TRUE per-element math, shared by CPU and GPU
// ---------------------------------------------------------------------------
// Project 7.3 : Clinical NLP over Notes & Records   (see ../THEORY.md)
//
// WHY THIS HEADER EXISTS  (PATTERNS.md §2, the "__host__ __device__ core" idiom)
//   A transformer self-attention encoder block is built from a few tiny scalar
//   recipes: how a token embedding is filled, how the query/key/value projection
//   matrices are filled, and the softmax numerics. If the CPU reference
//   (reference_cpu.cpp, compiled by cl.exe) and the GPU kernels (kernels.cu,
//   compiled by nvcc) each wrote their own copy of those formulas, they would
//   drift and the GPU-vs-CPU check would become fuzzy. Instead we write every
//   scalar recipe EXACTLY ONCE, here, as an inline function tagged
//   `__host__ __device__`, and call it from both sides. Verification can then be
//   near-exact (see main.cu tolerances).
//
//   This header must be includable by the plain host compiler, so it contains
//   NO CUDA-only constructs (no __global__, no <<<>>>). The ATTN_HD macro below
//   evaporates to nothing under cl.exe and becomes `__host__ __device__` under
//   nvcc -- that is the whole trick.
//
// THE MODEL IN ONE BREATH  (full derivation in ../THEORY.md)
//   A note is a sequence of S token ids. Look up each id's D-dim embedding to
//   get X [S x D]. Project X three ways with learned matrices:
//       Q = X Wq,  K = X Wk,  V = X Wv        (each [S x D])
//   Split the D columns into H heads of width dh = D/H. For each head h:
//       scores = Q_h K_hᵀ / sqrt(dh)          ([S x S] token-token affinities)
//       A_h    = softmax(scores, per row)     (row = "who does token i look at")
//       O_h    = A_h V_h                       ([S x dh] context vectors)
//   Concatenate the H heads back to O [S x D]. That O is the encoder-block output
//   -- a contextualized embedding per token. Padding tokens are masked out of the
//   softmax (they contribute -inf to scores) so they never receive attention.
//
//   The two matrix multiplies (Q Kᵀ and A V) are the GEMM-dominated bottleneck
//   the catalog names; on the GPU we hand them to cuBLAS (batched DGEMM). The
//   softmax and the masking are the hand-written kernel. Everything numeric that
//   BOTH sides must agree on lives in this file.
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cu  (both include this file).
// ===========================================================================
#pragma once

#include <cmath>     // std::sqrt, std::exp
#include <cstddef>   // std::size_t

// ---------------------------------------------------------------------------
// ATTN_HD : the host/device portability shim (see header intro).
//   * Under nvcc (__CUDACC__ defined) a function tagged ATTN_HD compiles for
//     BOTH host and device, so the same object can be called from main.cu's host
//     code and from inside a __global__ kernel.
//   * Under a plain host compiler the decorators do not exist, so ATTN_HD must
//     expand to nothing -- the function is then just an ordinary inline.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define ATTN_HD __host__ __device__
#else
#define ATTN_HD
#endif

namespace attn {

// Special token ids (must match scripts/make_synthetic.py and the vocab order).
enum : int {
    TOK_CLS = 0,   // sequence-summary token; its output row is the note vector
    TOK_PAD = 1    // padding token; masked out of every softmax
};

// ===========================================================================
// SECTION A -- the "pretend-learned" parameter recipes
// ---------------------------------------------------------------------------
// A real clinical BERT LEARNS its embeddings and projection weights on billions
// of tokens. This teaching project cannot train, so we FABRICATE deterministic
// weights with fixed integer recipes -- identical on every machine, so the demo
// output is byte-stable (CLAUDE.md determinism rule). The recipes are mirrored
// exactly in scripts/make_synthetic.py (which writes the embedding table into
// the sample file) so the file and the code never disagree.
// ===========================================================================

// proj_entry: entry (i,j) of a [D x D] projection matrix.
//   kind selects which matrix: 0 = Wq, 1 = Wk, 2 = Wv.
//   We start from the IDENTITY (so Q,K,V resemble X) and add a small structured
//   perturbation, so the three projections are distinct yet the planted
//   "he"~"patient" alignment (built into the embeddings) survives. Because both
//   the CPU reference and the GPU rebuild Wq/Wk/Wv from THIS function, they use
//   bit-identical weights.
//     i : row (input dim)      in [0, D)
//     j : col (output dim)     in [0, D)
//     kind : 0/1/2 for Wq/Wk/Wv
ATTN_HD inline double proj_entry(int i, int j, int kind) {
    double base = (i == j) ? 1.0 : 0.0;                       // identity backbone
    // Small, bounded perturbation in ~[-0.125, 0.125]; integer math => exact.
    double pert = static_cast<double>(((i * 3 + j * 5 + kind * 11) % 7) - 3) / 24.0;
    return base + pert;
}

// ===========================================================================
// SECTION B -- softmax numerics (the one place precision bites)
// ---------------------------------------------------------------------------
// softmax turns a row of raw attention scores into a probability distribution
// over the S key positions. The naive exp(x)/Σexp(x) OVERFLOWS for large x, so
// the standard, numerically-stable form subtracts the row maximum first:
//       p_j = exp(x_j - m) / Σ_k exp(x_k - m),   m = max_k x_k
// This is algebraically identical (the m cancels) but never exponentiates a
// positive number. BOTH the CPU reference and the GPU softmax kernel use the
// two helpers below, so their rounding matches.
// ===========================================================================

// row_max: the maximum of a length-`n` score row (skips nothing; masked
//   positions are already set to a large negative sentinel by the caller so
//   they lose the max). Returned value feeds softmax_exp for stability.
ATTN_HD inline double row_max(const double* row, int n) {
    double m = row[0];
    for (int k = 1; k < n; ++k)
        if (row[k] > m) m = row[k];
    return m;
}

// softmax_exp: the stabilized numerator exp(x - m) for one score.
//   Kept as a named function (not inlined by hand) so the CPU loop and the GPU
//   thread call the EXACT same std::exp with the EXACT same argument order.
ATTN_HD inline double softmax_exp(double x, double m) {
    return std::exp(x - m);
}

// attn_scale: the 1/sqrt(dh) scaling applied to Q·Kᵀ before softmax.
//   Dividing by sqrt of the head width keeps the dot products from growing with
//   dh (which would push softmax into a one-hot regime with vanishing
//   gradients). dh = D / H is the per-head dimension.
ATTN_HD inline double attn_scale(int dh) {
    return 1.0 / std::sqrt(static_cast<double>(dh));
}

// ===========================================================================
// SECTION C -- an interpretable, deterministic per-token summary
// ---------------------------------------------------------------------------
// After attention we want ONE human-meaningful number per token to print. We
// use the SHANNON ENTROPY of the token's attention distribution: low entropy =
// the token focuses on a few positions (sharp, e.g. a pronoun locking onto its
// antecedent); high entropy = it spreads attention broadly. Same formula on both
// sides so the printed table is verifiable.
//   p : length-`n` attention row (a probability distribution, sums to 1)
//   returns entropy in NATS (natural log); 0 for a one-hot row.
// ===========================================================================
ATTN_HD inline double attn_entropy(const double* p, int n) {
    double h = 0.0;
    for (int k = 0; k < n; ++k) {
        double pk = p[k];
        if (pk > 1.0e-300)              // 0*log0 := 0; guard the log domain
            h -= pk * std::log(pk);
    }
    return h;
}

}  // namespace attn
