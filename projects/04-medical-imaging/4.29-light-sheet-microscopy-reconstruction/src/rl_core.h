// ===========================================================================
// src/rl_core.h  --  Shared __host__ __device__ Richardson-Lucy per-pixel math
// ---------------------------------------------------------------------------
// Project 4.29 : Light-Sheet Microscopy Reconstruction (reduced-scope teaching
//                version: 2D single-view Richardson-Lucy deconvolution).
//
// THE HD-MACRO IDIOM (PATTERNS.md §2 -- "the single most useful idiom")
//   The *per-pixel physics* of one Richardson-Lucy (RL) iteration is expressed
//   here ONCE as tiny `__host__ __device__` inline functions. Both sides use it:
//     * reference_cpu.cpp  (compiled by the host C++ compiler, cl.exe) and
//     * kernels.cu         (compiled by nvcc)
//   include this header, so the CPU reference and the GPU kernels run
//   BYTE-FOR-BYTE-IDENTICAL arithmetic at the per-pixel level. That turns
//   verification from "approximately equal" into "equal to a tiny, explained
//   floating-point tolerance" (see THEORY.md "How we verify correctness").
//
//   Keep this header free of CUDA-only *types* and of `__global__` so the plain
//   host compiler can also include it. The ONLY CUDA-specific thing here is the
//   __host__ __device__ decoration, which we hide behind RL_HD.
//
// WHERE THE MATH COMES FROM
//   Richardson-Lucy is the maximum-likelihood deconvolution for POISSON noise
//   (photon shot noise -- exactly what a fluorescence camera sees). Given a blurry
//   measured image b, a point-spread function (PSF) h, and the current estimate
//   x_k of the true image, one RL iteration is:
//
//       x_{k+1} = x_k * [ h^T  conv  ( b / (h conv x_k) ) ]
//
//   where `conv` is convolution, `h^T` is the flipped PSF (correlation), `*` and
//   `/` are ELEMENT-WISE. This header supplies the two element-wise steps; the
//   two convolutions are done with cuFFT on the GPU (kernels.cu) and a matching
//   DFT on the CPU (reference_cpu.cpp). Read THEORY.md "The math" for the
//   derivation and why every factor is non-negative (RL preserves positivity).
//
// READ THIS BEFORE: reference_cpu.h, kernels.cuh.
// ===========================================================================
#pragma once

// RL_HD expands to `__host__ __device__` when compiled by nvcc (so the function
// can run on BOTH the CPU and the GPU), and to nothing when compiled by the
// plain host compiler (which has never heard of those keywords). This is the
// portable HD-macro from PATTERNS.md §2.
#ifdef __CUDACC__
#define RL_HD __host__ __device__
#else
#define RL_HD
#endif

// A tiny positive floor. RL divides the measurement by the current re-blurred
// estimate; if that estimate is ~0 in some pixel the ratio would blow up. We
// clamp the denominator to RL_EPS so the update stays finite and deterministic.
// The same constant is used on both sides so the clamp happens at identical
// pixels -- essential for CPU==GPU agreement.
#ifndef RL_EPS
#define RL_EPS 1.0e-7
#endif

// -----------------------------------------------------------------------------
// rl_ratio(measured, reblurred): the element-wise "correction ratio" step.
//   Computes  b_i / max(reblurred_i, RL_EPS).
//   * measured   -- one pixel of the observed (blurry, noisy) image b, photons.
//   * reblurred  -- one pixel of (h conv x_k): the current estimate seen through
//                   the same blur, i.e. what x_k PREDICTS the camera would see.
//   Intuition: if the prediction is too low the ratio > 1 (push that region up
//   next), if too high the ratio < 1 (pull it down). Where they match, ratio = 1
//   and the estimate is left unchanged -- the fixed point of RL.
//   Returns a plain double; both host and device call the identical code.
// -----------------------------------------------------------------------------
RL_HD inline double rl_ratio(double measured, double reblurred) {
    double denom = reblurred > RL_EPS ? reblurred : RL_EPS;  // guard divide-by-~0
    return measured / denom;
}

// -----------------------------------------------------------------------------
// rl_apply(current, correction): the element-wise multiplicative update step.
//   Computes  x_{k+1,i} = max(current_i * correction_i, 0).
//   * current     -- one pixel of the current estimate x_k.
//   * correction  -- one pixel of ( h^T conv ratio ): how much to scale it by.
//   RL is MULTIPLICATIVE, which is what keeps the intensity non-negative (a real
//   physical constraint: you cannot have negative photons). We still clamp at 0
//   to absorb any tiny negative excursion from floating-point round-off in the
//   FFT-based convolution, so the estimate stays a valid image on both paths.
// -----------------------------------------------------------------------------
RL_HD inline double rl_apply(double current, double correction) {
    double v = current * correction;
    return v > 0.0 ? v : 0.0;   // enforce non-negativity (photon counts >= 0)
}
