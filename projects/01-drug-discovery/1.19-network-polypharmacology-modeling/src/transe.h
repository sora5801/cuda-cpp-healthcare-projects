// ===========================================================================
// src/transe.h  --  Shared (host + device) TransE scoring core
// ---------------------------------------------------------------------------
// Project 1.19 : Network / Polypharmacology Modeling
//
// WHAT THIS FILE IS
//   The SINGLE source of truth for the per-candidate math of TransE link
//   prediction. Both the CPU reference (reference_cpu.cpp, compiled by the host
//   C++ compiler) and the GPU kernel (kernels.cu, compiled by nvcc) include this
//   header and call the SAME inline functions, so they compute byte-for-byte
//   identical numbers -- which is what lets us verify GPU == CPU EXACTLY
//   (tolerance 0.0; see ../THEORY.md "How we verify correctness").
//
//   The "HD" idiom (PATTERNS.md sec 2): TRANSE_HD expands to `__host__ __device__`
//   when nvcc is compiling (so the function is usable on the GPU) and to nothing
//   under the plain host compiler (which has never heard of those keywords). Keep
//   this header free of CUDA-only types and of `__global__` so the host compiler
//   can include it unchanged.
//
// THE MODEL (one paragraph; full treatment in THEORY.md)
//   A knowledge graph fact is a triple (head h, relation r, tail t). TransE
//   embeds every entity and relation as a d-dimensional vector and asserts that
//   a TRUE fact satisfies  h + r ~= t. The plausibility SCORE of a candidate tail
//   t is the NEGATIVE distance between (h + r) and t:
//       score(t) = - || (h + r) - t ||        (L2 norm, "L2 variant" of TransE)
//   A larger (closer to 0) score means a more plausible link. For drug-target /
//   off-target prediction we fix h = a query drug, r = the TARGETS relation, and
//   score EVERY protein tail t -- then rank. Each tail is INDEPENDENT, so the GPU
//   gives each candidate its own thread (the "independent jobs" pattern, 1.12).
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cu. The data model (how these
// vectors are loaded and laid out) is in reference_cpu.h.
// ===========================================================================
#pragma once

#ifdef __CUDACC__
#define TRANSE_HD __host__ __device__   // nvcc: usable on host AND device
#else
#define TRANSE_HD                       // host compiler: the decorators don't exist
#endif

// ---------------------------------------------------------------------------
// transe_squared_distance
//   Compute the SQUARED L2 distance  sum_k ( (h[k] + r[k]) - t[k] )^2  between
//   the "translated head" (h + r) and one candidate tail t, both d-dimensional.
//
//   We return the SQUARED distance (not the square root) because:
//     * it is monotonic in the true distance, so ranking by -d^2 ranks identically
//       to ranking by -d (the sqrt is order-preserving) -- we skip a sqrt per
//       candidate, and
//     * it keeps the arithmetic a plain sum of products, so the CPU and GPU do the
//       EXACT same float operations in the EXACT same order -> identical results.
//
//   Parameters (all caller-owned, read-only):
//     h   : pointer to the head (query drug) embedding, length `dim`
//     r   : pointer to the relation (TARGETS) embedding, length `dim`
//     t   : pointer to ONE candidate tail (protein) embedding, length `dim`
//     dim : embedding dimension d
//   Returns: the squared L2 distance as a float (smaller = more plausible link).
//
//   Complexity: O(dim) multiply-adds, no branches -> trivially vectorizable on
//   both CPU and GPU. Called once per candidate by both implementations.
// ---------------------------------------------------------------------------
TRANSE_HD inline float transe_squared_distance(const float* h, const float* r,
                                               const float* t, int dim) {
    float acc = 0.0f;   // running sum of squared per-dimension residuals
    // Walk the embedding dimension by dimension. The loop order is fixed and
    // identical on host and device, so the floating-point summation order (and
    // therefore the rounded result) matches exactly -- this is what makes the
    // GPU-vs-CPU check pass at tolerance 0.
    for (int k = 0; k < dim; ++k) {
        const float diff = (h[k] + r[k]) - t[k];   // residual in dimension k
        acc += diff * diff;                          // accumulate its square
    }
    return acc;
}

// ---------------------------------------------------------------------------
// transe_score
//   The plausibility score we actually rank by: the NEGATIVE distance. We use
//   the negative SQUARED distance so "higher score = better candidate" while
//   avoiding the sqrt (see the note above). Defined as a tiny wrapper so the
//   intent ("bigger is more plausible") is explicit at every call site.
// ---------------------------------------------------------------------------
TRANSE_HD inline float transe_score(const float* h, const float* r,
                                    const float* t, int dim) {
    return -transe_squared_distance(h, r, t, dim);
}
