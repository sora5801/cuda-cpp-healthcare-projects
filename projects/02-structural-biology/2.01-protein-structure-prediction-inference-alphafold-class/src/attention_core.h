// ===========================================================================
// src/attention_core.h  --  The shared, per-element attention MATH (CPU==GPU)
// ---------------------------------------------------------------------------
// Project 2.1 : Protein Structure Prediction Inference (AlphaFold-class)
//               REDUCED-SCOPE TEACHING VERSION (see ../THEORY.md sec 7 and the
//               README "Limitations": we implement ONE Evoformer building block
//               -- scaled dot-product self-attention -- not the whole pipeline).
//
// WHY THIS HEADER EXISTS  (PATTERNS.md sec 2: the __host__ __device__ core)
//   The single most useful idiom in this repo: put the *per-element physics*
//   (here, the per-element transformer math) in ONE header as `__host__
//   __device__` inline functions. Then:
//     * reference_cpu.cpp includes it through the HOST compiler, and
//     * kernels.cu includes it through nvcc for the DEVICE,
//   so the CPU reference and the GPU kernel run BYTE-FOR-BYTE-IDENTICAL math.
//   That makes verification a tight numeric check instead of a loose guess.
//   We keep CUDA-only constructs (no __global__, no <cuda_runtime.h>) OUT of
//   this header so the plain C++ host compiler can include it unchanged.
//
// WHAT "ATTENTION" IS, IN ONE BREATH
//   A protein is a sequence of L residues. AlphaFold/ESMFold represent each
//   residue by a feature vector of width d ("the single representation"). A
//   self-attention layer lets every residue look at every other residue and
//   pull in information: residue i's new vector is a weighted average of all
//   residues' "value" vectors, where the weights come from how similar i's
//   "query" vector is to each residue's "key" vector. That is exactly the
//   operation stacked dozens of times inside an Evoformer block.
//
// THE FORMULA (one attention head)
//   Given, for the L residues, matrices Q,K,V each of shape [L x d]:
//       scores[i][j] = (Q[i] . K[j]) / sqrt(d)            // affinity i->j
//       w[i][:]      = softmax_j(scores[i][:])            // sum_j w[i][j] = 1
//       Out[i][:]    = sum_j w[i][j] * V[j][:]            // context-mixed vec
//   The 1/sqrt(d) scale keeps the dot products from growing with d (so softmax
//   does not saturate). This file provides the three scalar primitives those
//   three lines are built from: a dot product, the scale, and a stable softmax
//   numerator. The CPU loop and the GPU kernel both call these.
//
// READ THIS BEFORE: reference_cpu.h, kernels.cuh. Then reference_cpu.cpp and
// kernels.cu, which are two implementations of the SAME three lines above.
// ===========================================================================
#pragma once

#include <cmath>     // std::expf, std::sqrtf  (host side); device uses __expf-free expf
#include <cstddef>   // std::size_t

// ---------------------------------------------------------------------------
// AC_HD : expands to `__host__ __device__` when compiled by nvcc, and to
// nothing when compiled by the plain host compiler. This is the macro that
// lets ONE function body live on both sides (PATTERNS.md sec 2). __CUDACC__ is
// defined by nvcc while it is compiling a translation unit; cl.exe/g++ never
// define it, so reference_cpu.cpp sees a normal inline function.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define AC_HD __host__ __device__
#else
#define AC_HD
#endif

// ---------------------------------------------------------------------------
// Model dimensions (compile-time constants so loops unroll and shared-memory
// sizes are known at compile time).
//   D_MODEL is the per-residue feature width. Real AF2 uses 256 for the single
//   representation and 128 for the pair representation; ESMFold's language
//   model uses up to 2560. We use a small, didactic 32 so the whole [L x d]
//   matrices fit in a tiny committed sample and the math is easy to trace by
//   hand. It is a power of two and a multiple of the 32-lane warp, which keeps
//   the GPU reduction clean.
// ---------------------------------------------------------------------------
constexpr int D_MODEL = 32;   // feature dimension d of each residue vector

// ---------------------------------------------------------------------------
// dot_d : the inner product of two length-d residue vectors, a . b = sum_k a[k]*b[k].
//   This is the single most-executed operation in a transformer: every
//   score[i][j] is one dot product, and there are L*L of them. We compute it in
//   double internally so the CPU and GPU accumulate in the SAME order with the
//   SAME rounding -> identical results (the kernel calls this exact function).
//   Returns the dot product as a double; the caller casts/scales as needed.
//
//   Parameters:
//     a, b : pointers to two length-`d` contiguous float vectors (residue rows).
//     d    : the vector length (== D_MODEL at the call sites; passed explicitly
//            so the function is self-contained and unit-testable).
//   Complexity: O(d) multiply-adds. No allocation, no side effects.
// ---------------------------------------------------------------------------
AC_HD inline double dot_d(const float* a, const float* b, int d) {
    double acc = 0.0;                 // accumulate in double for parity + accuracy
    for (int k = 0; k < d; ++k) {
        // promote each float to double BEFORE multiplying so both sides round
        // the product identically (FMA contraction is disabled in this scalar
        // form -> deterministic across host and device; see THEORY sec 5).
        acc += static_cast<double>(a[k]) * static_cast<double>(b[k]);
    }
    return acc;
}

// ---------------------------------------------------------------------------
// scaled_score : the raw attention affinity (Q[i] . K[j]) / sqrt(d).
//   Dividing by sqrt(d) is the "scaled" in "scaled dot-product attention"
//   (Vaswani et al. 2017): without it, dot products grow ~proportionally to d,
//   pushing softmax into regions where one weight is ~1 and the rest ~0 (tiny
//   gradients in training; over-peaked attention at inference). We keep the
//   value in double through the divide, matching CPU and GPU exactly.
//
//   Parameters:
//     q_i, k_j : length-`d` query and key residue vectors.
//     d        : feature width (used for both the dot length and the scale).
//   Returns: the scaled affinity as a double.
// ---------------------------------------------------------------------------
AC_HD inline double scaled_score(const float* q_i, const float* k_j, int d) {
    // sqrt(d) computed in double; for d=32 this is ~5.656854. Using double here
    // (not sqrtf) guarantees the host and device divide by the identical bits.
    const double inv_sqrt_d = 1.0 / std::sqrt(static_cast<double>(d));
    return dot_d(q_i, k_j, d) * inv_sqrt_d;
}

// ---------------------------------------------------------------------------
// stable_exp : exp(s - m), the numerically-stable softmax numerator.
//   Softmax of a row is exp(s_j) / sum_j exp(s_j). If any s_j is large, exp
//   overflows to +inf and the ratio becomes NaN. The standard fix subtracts the
//   row maximum m first: exp(s_j - m) is in (0, 1], and dividing by the (also
//   shifted) sum gives the identical mathematical result with no overflow. Both
//   the CPU reference and the GPU kernel call this with the SAME m, so the
//   shifted exponentials match bit-for-bit.
//
//   Parameters:
//     s : one scaled score (a single scores[i][j]).
//     m : the maximum scaled score over row i (the shift).
//   Returns: exp(s - m) as a double in (0, 1].
// ---------------------------------------------------------------------------
AC_HD inline double stable_exp(double s, double m) {
    return std::exp(s - m);   // std::exp resolves to the double overload on both sides
}
