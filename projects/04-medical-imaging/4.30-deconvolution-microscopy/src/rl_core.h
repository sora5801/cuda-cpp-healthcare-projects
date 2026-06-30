// ===========================================================================
// src/rl_core.h  --  The shared, per-pixel Richardson-Lucy math (CPU == GPU)
// ---------------------------------------------------------------------------
// Project 4.30 : Deconvolution Microscopy
//
// WHY THIS FILE EXISTS  (PATTERNS.md  section 2 -- the HD-macro idiom)
//   Richardson-Lucy (RL) deconvolution is an *iterative* algorithm. Every
//   iteration touches each pixel with two trivial element-wise operations:
//       (a) a RATIO    : r = observed / blurred_estimate     (with a guard)
//       (b) a MULTIPLY : est_new = est_old * correction       (clamped >= 0)
//   The CONVOLUTIONS between (a) and (b) differ in *how* they are computed
//   (the CPU reference does a direct circular convolution; the GPU does the
//   identical circular convolution via cuFFT) -- but these two PER-PIXEL steps
//   must be byte-for-byte identical on both sides, or the two iterations would
//   slowly drift apart for reasons that have nothing to do with the FFT.
//
//   So we put (a) and (b) in ONE header as `__host__ __device__` inline
//   functions. The CPU reference loops over them; the GPU kernels call the
//   same functions from one thread per pixel. Same math, both sides.
//
//   This header is included by BOTH:
//       * reference_cpu.cpp  (compiled by the host C++ compiler, cl.exe)
//       * kernels.cu / main.cu (compiled by nvcc)
//   so it MUST stay free of CUDA-only constructs (no __global__, no float2,
//   no <cuda_runtime.h>). Only the HD decorators, guarded by __CUDACC__.
//
// READ THIS BEFORE: reference_cpu.h, kernels.cuh.
// ===========================================================================
#pragma once

// ---------------------------------------------------------------------------
// The HD macro. Under nvcc (__CUDACC__ defined) a function tagged RL_HD is
// compiled for BOTH the host and the device. Under the plain host compiler the
// decorators do not exist, so RL_HD expands to nothing and the function is an
// ordinary inline host function. This is the single trick that buys CPU/GPU
// numerical parity (PATTERNS.md section 2).
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define RL_HD __host__ __device__
#else
#define RL_HD
#endif

// A tiny floor used in two places:
//   * to avoid dividing by zero when the current blurred estimate is ~0, and
//   * to keep the running estimate strictly non-negative (RL assumes a
//     non-negative intensity image -- fluorescence photon counts -- and the
//     multiplicative update preserves that as long as we never let a pixel go
//     to exactly zero, which would freeze it forever).
// We use double everywhere in the RL math: RL is run for many iterations and
// single precision would accumulate visible error. The images are small
// (teaching scale), so double costs us nothing meaningful.
#ifndef RL_EPS
#define RL_EPS 1.0e-12
#endif

// ---------------------------------------------------------------------------
// rl_ratio: the data-fidelity ratio of one pixel.
//   observed         = the measured (blurred, noisy) pixel value  [photons]
//   blurred_estimate = (current estimate * PSF) at this pixel, i.e. what the
//                      current guess WOULD look like through the microscope.
// Returns observed / blurred_estimate, guarding the denominator so a near-zero
// background pixel cannot blow up. This ratio is what RL then back-projects
// (correlates with the PSF) to form the multiplicative correction.
//
// Intuition: if our blurred estimate already matches the data, the ratio is 1
// (no change). Where we under-predict the data the ratio > 1 (push intensity
// up); where we over-predict it the ratio < 1 (pull it down).
// ---------------------------------------------------------------------------
RL_HD inline double rl_ratio(double observed, double blurred_estimate) {
    const double denom = (blurred_estimate > RL_EPS) ? blurred_estimate : RL_EPS;
    return observed / denom;
}

// ---------------------------------------------------------------------------
// rl_update: the multiplicative RL step for one pixel.
//   old_estimate = the current intensity guess at this pixel
//   correction   = (ratio  (correlated with)  PSF) at this pixel -- the
//                  back-projected data-fidelity term produced by the second
//                  convolution.
// Returns old_estimate * correction, clamped to be non-negative. The clamp is
// a safety net: in exact arithmetic the product of two non-negatives is
// non-negative, but a tiny negative can appear from FFT round-off, and a
// negative intensity is physically meaningless for a fluorescence image.
//
// This is the heart of Richardson-Lucy: a MULTIPLICATIVE (not additive)
// gradient-ascent step on the Poisson log-likelihood. Multiplicative updates
// automatically keep the image non-negative and converge to the maximum-
// likelihood estimate under Poisson (photon shot) noise -- see THEORY.md section math.
// ---------------------------------------------------------------------------
RL_HD inline double rl_update(double old_estimate, double correction) {
    const double v = old_estimate * correction;
    return (v > 0.0) ? v : 0.0;
}
