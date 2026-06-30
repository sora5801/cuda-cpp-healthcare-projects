// ===========================================================================
// src/attention_math.h  --  The one true per-element attention math (CPU/GPU)
// ---------------------------------------------------------------------------
// Project 3.18 : Protein Language Model Inference
//
// THE SHARED `__host__ __device__` CORE  (PATTERNS.md §2)
//   The single most useful idiom in this repository: put the *per-element math*
//   that both the CPU reference and the GPU kernel must agree on in ONE header,
//   decorated `__host__ __device__`, so the host compiler (cl.exe / g++) and
//   nvcc emit the SAME formulas. The CPU reference loops these; each GPU thread
//   calls them. Verification then compares two implementations of identical
//   arithmetic, so any divergence is a real bug, not a transcription slip.
//
//   This file deliberately contains NO CUDA-only types and NO `__global__`
//   kernels -- only inline scalar/vector helpers -- so the plain host compiler
//   can `#include` it when building reference_cpu.cpp.
//
// WHAT LIVES HERE
//   * AH_HD                     -- the host/device decorator macro.
//   * AttnConfig                -- the model's tiny hyper-parameters.
//   * token_of() / EMBED ...    -- deterministic synthetic embedding generation
//                                  (so CPU and GPU start from identical inputs).
//   * proj_one()                -- one output of a linear projection (a dot
//                                  product of an input row with a weight column).
//   * scaled_score()            -- one entry q.k / sqrt(d_head) of the attention
//                                  logit matrix.
//   * softmax_inplace()         -- the numerically-stable row softmax used by
//                                  BOTH sides (subtract row max, exp, normalize).
//
// READ THIS BEFORE: reference_cpu.h, kernels.cuh. It is the contract both honor.
// ===========================================================================
#pragma once

#include <cmath>     // std::exp, std::sqrt, std::fabs  (host); nvcc maps these on device
#include <cstdint>   // fixed-width integer types for the deterministic RNG

// ---------------------------------------------------------------------------
// AH_HD: when compiled by nvcc (__CUDACC__ defined) we want these functions
// callable from BOTH host and device, so we decorate them `__host__ __device__`.
// When compiled by the plain host compiler for reference_cpu.cpp, those CUDA
// keywords do not exist, so the macro expands to nothing. This is the exact
// HD-macro idiom from PATTERNS.md §2.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define AH_HD __host__ __device__
#else
#define AH_HD
#endif

// ---------------------------------------------------------------------------
// AttnConfig: the (deliberately tiny) shape of our one transformer block.
//   A production ESM-2 has L up to ~1024, d_model 1280, 20 heads, 33 layers.
//   We teach the SAME arithmetic at a size a learner can trace by hand and a
//   CPU can verify exactly. All fields are plain ints so the struct is trivially
//   copyable to the device as a kernel argument.
//
//   Invariant: d_model == n_heads * d_head  (heads partition the embedding).
// ---------------------------------------------------------------------------
struct AttnConfig {
    int seq_len = 0;    // L : number of residues in the protein (rows of every matrix)
    int d_model = 0;    // model embedding width (columns of the residue embeddings)
    int n_heads = 0;    // H : number of attention heads
    int d_head  = 0;    // d_model / H : width each head operates on
};

// ---------------------------------------------------------------------------
// THE 20 CANONICAL AMINO ACIDS, in a fixed order. A protein sequence is a string
// over this alphabet; its index in this table is the token id fed to the model.
// (Real PLMs add special tokens <cls>/<eos>/<pad>/<mask>; we keep the 20 to
//  focus on the attention math -- see THEORY §"Where this sits in the real world".)
// ---------------------------------------------------------------------------
static const char AA_ALPHABET[21] = "ACDEFGHIKLMNPQRSTVWY";  // 20 chars + NUL

// token_of: map an amino-acid letter to its token id in [0,19], or -1 if the
// character is not one of the 20 canonical residues. Used by the loader to turn
// the sequence string into integer tokens; both CPU and GPU index the same
// embedding table with these ids.
AH_HD inline int token_of(char c) {
    // A tiny linear scan over 20 letters -- clarity over a lookup table.
    for (int i = 0; i < 20; ++i)
        if (AA_ALPHABET[i] == c) return i;
    return -1;  // unknown residue (caller decides how to handle)
}

// ---------------------------------------------------------------------------
// DETERMINISTIC SYNTHETIC WEIGHTS / EMBEDDINGS
//   A real PLM ships hundreds of megabytes of trained float weights. For a
//   self-contained, reproducible teaching demo we GENERATE every weight from a
//   fixed integer hash, so CPU and GPU build byte-identical tensors with no data
//   files and no RNG-order ambiguity. splitmix64 is a tiny, well-mixed hash:
//   given the same 64-bit key it returns the same 64-bit value on every machine.
// ---------------------------------------------------------------------------
AH_HD inline uint64_t splitmix64(uint64_t x) {
    // The published splitmix64 finalizer. Pure integer math => identical on host
    // and device, which is what makes our synthetic tensors reproducible.
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// hash_to_unit: turn a 64-bit hash into a deterministic float in [-1, 1).
//   We take the top 24 bits (the float mantissa width) so the value is exactly
//   representable, then rescale. Same bits in -> same float out, everywhere.
AH_HD inline float hash_to_unit(uint64_t h) {
    const uint32_t m = static_cast<uint32_t>(h >> 40);   // top 24 bits
    const float u = static_cast<float>(m) / 16777216.0f; // [0,1)
    return 2.0f * u - 1.0f;                              // [-1,1)
}

// embed_value: the (residue-token, feature) entry of the synthetic embedding
//   table. Mixing the token id and the feature index through splitmix64 gives a
//   stable pseudo-random embedding that nonetheless depends on the residue, so
//   different residues get different vectors (as a trained table would).
//   `salt` lets us reuse this generator for distinct tensors (Wq/Wk/Wv/Wo).
AH_HD inline float embed_value(int token, int feature, uint64_t salt) {
    const uint64_t key = (static_cast<uint64_t>(token) << 32)
                       ^ (static_cast<uint64_t>(feature) * 0x100000001B3ULL)
                       ^ salt;
    return hash_to_unit(splitmix64(key));
}

// weight_value: the (row,col) entry of a synthetic projection weight matrix.
//   Each of Wq, Wk, Wv, Wo is a d_model x d_model matrix; `salt` selects which
//   one. Scaling by 1/sqrt(d_model) keeps projected vectors O(1) (the standard
//   "Xavier-ish" variance-preserving init), so softmax logits stay in a sane
//   range and the demo's numbers are interpretable.
AH_HD inline float weight_value(int row, int col, int d_model, uint64_t salt) {
    const uint64_t key = (static_cast<uint64_t>(row) << 32)
                       ^ (static_cast<uint64_t>(col) * 0x9E3779B1u)
                       ^ salt;
    const float raw = hash_to_unit(splitmix64(key));
    return raw / std::sqrt(static_cast<float>(d_model));
}

// Salts that name our four projection matrices (any distinct constants work).
static const uint64_t SALT_EMBED = 0xE3BED0ULL;  // residue embedding table
static const uint64_t SALT_WQ    = 0x511EAULL;   // query projection
static const uint64_t SALT_WK    = 0x511EBULL;   // key   projection
static const uint64_t SALT_WV    = 0x511ECULL;   // value projection
static const uint64_t SALT_WO    = 0x511EDULL;   // output projection

// ---------------------------------------------------------------------------
// proj_one: ONE scalar output of a linear projection  y[j] = sum_k x[k]*W[k,j].
//   Used to project a residue's d_model-vector `x` onto column `out_col` of a
//   weight matrix selected by `salt`. The sum accumulates in DOUBLE so the CPU
//   and GPU agree to near machine precision regardless of add order (the inner
//   loop is sequential and identical on both sides). Returns a float (the tensor
//   element). Complexity: O(d_model) per call.
// ---------------------------------------------------------------------------
AH_HD inline float proj_one(const float* x, int d_model, int out_col, uint64_t salt) {
    double acc = 0.0;
    for (int k = 0; k < d_model; ++k)
        acc += static_cast<double>(x[k]) * weight_value(k, out_col, d_model, salt);
    return static_cast<float>(acc);
}

// ---------------------------------------------------------------------------
// scaled_score: ONE entry of the attention logit matrix for a head:
//     score(i,j) = (q_i . k_j) / sqrt(d_head)
//   `q` and `k` are this head's d_head-wide query/key vectors for residues i,j.
//   The 1/sqrt(d_head) scaling (Vaswani et al. 2017) keeps the dot product's
//   variance ~1 so softmax does not saturate. Double accumulation again. The
//   caller then runs softmax over j to get attention weights.
// ---------------------------------------------------------------------------
AH_HD inline float scaled_score(const float* q, const float* k, int d_head) {
    double dot = 0.0;
    for (int t = 0; t < d_head; ++t)
        dot += static_cast<double>(q[t]) * static_cast<double>(k[t]);
    return static_cast<float>(dot / std::sqrt(static_cast<double>(d_head)));
}

// ---------------------------------------------------------------------------
// softmax_inplace: numerically-stable softmax of a length-n logit row, in place.
//   softmax(s)_j = exp(s_j - m) / sum_l exp(s_l - m),  where m = max_l s_l.
//   Subtracting the row max `m` before exp prevents overflow (exp of a large
//   logit) without changing the result -- the canonical stable softmax. BOTH the
//   CPU reference and the GPU kernel call THIS function on identical inputs, so
//   their attention weights match. Accumulate the denominator in double.
//   Complexity: O(n). Operates on `row[0..n-1]`.
// ---------------------------------------------------------------------------
AH_HD inline void softmax_inplace(float* row, int n) {
    // (1) row maximum -- the stabilizing shift.
    float m = row[0];
    for (int j = 1; j < n; ++j)
        if (row[j] > m) m = row[j];
    // (2) exponentiate the shifted logits and accumulate their sum.
    double sum = 0.0;
    for (int j = 0; j < n; ++j) {
        const float e = std::exp(row[j] - m);
        row[j] = e;
        sum += static_cast<double>(e);
    }
    // (3) normalize so the row sums to 1 (a probability distribution over keys).
    const float inv = static_cast<float>(1.0 / sum);
    for (int j = 0; j < n; ++j)
        row[j] *= inv;
}
