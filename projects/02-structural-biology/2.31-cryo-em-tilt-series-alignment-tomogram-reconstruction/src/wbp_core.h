// ===========================================================================
// src/wbp_core.h  --  The ONE shared math core (CPU/GPU parity)
// ---------------------------------------------------------------------------
// Project 2.31 : Cryo-EM Tilt-Series Alignment & Tomogram Reconstruction
//
// WHY THIS FILE EXISTS  (PATTERNS.md sec.2 -- the shared __host__ __device__ core)
//   The single per-sample operation of weighted back-projection -- "where does
//   pixel (wx,wy)'s ray cross projection k's detector, and what is the linearly
//   interpolated filtered value there?" -- is run by BOTH the CPU reference
//   (reference_cpu.cpp, host compiler) AND the GPU kernel (kernels.cu, nvcc).
//   If we wrote that arithmetic twice it would inevitably drift. Instead we
//   write it ONCE here as an inline function decorated __host__ __device__:
//     * the host compiler ignores the (undefined) decorators and inlines it into
//       the CPU loop;
//     * nvcc compiles a device copy and inlines it into the kernel.
//   Result: the CPU and GPU execute byte-identical float ops, so their results
//   agree to ~1e-4 (the only residual difference is FMA contraction order; see
//   THEORY.md "Numerical considerations").
//
//   RULES for this header (so the host compiler can include it):
//     * NO __global__, NO CUDA-only types (float2, etc.), NO <cuda_runtime.h>.
//     * Only plain C++ and the WBP_HD decorator below.
//
// READ THIS AFTER: reference_cpu.h (it explains the science/geometry).
// READ THIS BEFORE: reference_cpu.cpp and kernels.cu (both include this).
// ===========================================================================
#pragma once

// WBP_HD expands to "__host__ __device__" only when compiled by nvcc (which
// defines __CUDACC__). Under the plain host compiler it expands to nothing, so
// the very same source is valid C++.  This is the "HD-macro idiom".
#ifdef __CUDACC__
#define WBP_HD __host__ __device__
#else
#define WBP_HD
#endif

// std::cos is needed by the host expansion of ramp_weight_hd(); nvcc also pulls
// in <cmath> for the host pass, and cosf is a device builtin. Including it here
// keeps this header self-contained for the plain C++ compiler.
#include <cmath>

// pi as a float constant, defined once. Back-projection multiplies the angular
// sum by d(theta) ~ pi/n_tilts (the discrete integration measure of the inverse
// Radon transform); using the same literal on both sides keeps the scale equal.
#ifndef WBP_PI_F
#define WBP_PI_F 3.14159265358979323846f
#endif

// ---------------------------------------------------------------------------
// sample_projection_hd : the heart of the back-projection gather.
//   Given one filtered projection row and the world coordinate (wx,wy) of an
//   output pixel, return that pixel's contribution from this single tilt:
//
//     s    = wx*cos(tilt) + wy*sin(tilt)      // signed position along detector
//     fidx = s / ds + center                  // fractional detector-bin index
//     val  = linear-interpolate(row, fidx)    // sample between the two bins
//
//   The ray for pixel (wx,wy) at tilt angle theta projects onto the (rotated)
//   detector at arc-length s; we read the filtered projection there. A pixel
//   whose ray falls off the detector contributes 0 (the `in range` guard).
//
//   PARAMETERS
//     row    : pointer to this tilt's filtered projection, length n_det.
//     n_det  : number of detector bins.
//     wx, wy : world coordinates of the output pixel (same units as ds).
//     cos_t  : cos(tilt_k)   (precomputed on the host so CPU==GPU trig).
//     sin_t  : sin(tilt_k).
//     ds     : detector bin spacing (world units per bin).
//     center : detector index of s=0, i.e. (n_det-1)/2.
//   RETURNS the interpolated contribution (0 if the ray misses the detector).
//
//   Called once per (pixel, tilt) pair -- the innermost operation of the whole
//   reconstruction, so it must be small and branch-light. No allocation, no I/O.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ramp_weight_hd : the ramp filter's frequency-domain weight for spectral bin f.
//
//   This is the SINGLE definition of "the ramp" shared by both ramp paths:
//     * the GPU (kernels.cu) builds a host-side table of these weights, uploads
//       it, and multiplies the cuFFT spectrum by it;
//     * the CPU reference (reference_cpu.cpp) multiplies its explicit-DFT
//       spectrum by the very same weight.
//   Because both use THIS function, the FFT ramp and the DFT ramp are the same
//   mathematical filter and agree to ~1e-4 (only transform round-off differs).
//
//   The weight is the continuous ramp |frequency| with a raised-cosine (Hann)
//   roll-off toward Nyquist (apodization tames the noise a pure ramp amplifies):
//       nu   = (f/n)/ds                     // spatial frequency at bin f
//       apod = 0.5*(1 + cos(pi * f/(nf-1)))  // 1 at DC, 0 at Nyquist
//       weight = nu * apod
//   NOTE: we do NOT fold cuFFT's 1/n inverse-normalization in here; each caller
//   applies its own transform normalization (the CPU DFT pair is already
//   normalized; the GPU divides by n separately). Keeping THIS function to the
//   pure physics keeps it reusable.
//
//   PARAMETERS
//     f      : spectral-bin index, 0..nf-1.
//     nf     : number of spectral bins (n_det/2 + 1 for a real FFT).
//     n      : detector length (so f/n is the normalized frequency in cycles/bin).
//     ds     : detector bin spacing (world units) -> nu in 1/world units.
//   RETURNS the real ramp weight for bin f (>= 0; weight[0] = 0 kills the DC mean).
// ---------------------------------------------------------------------------
WBP_HD inline float ramp_weight_hd(int f, int nf, int n, float ds) {
    const float nu   = (static_cast<float>(f) / static_cast<float>(n)) / ds;
    const float x    = static_cast<float>(f) / static_cast<float>(nf - 1);  // 0..1
    // cosf on device, cos on host: both round to the same float here.
#ifdef __CUDACC__
    const float apod = 0.5f * (1.0f + cosf(WBP_PI_F * x));
#else
    const float apod = 0.5f * (1.0f + std::cos(WBP_PI_F * x));
#endif
    return nu * apod;
}

WBP_HD inline float sample_projection_hd(const float* row, int n_det,
                                         float wx, float wy,
                                         float cos_t, float sin_t,
                                         float ds, float center) {
    // s: where this pixel's ray meets the rotated 1-D detector axis.
    const float s = wx * cos_t + wy * sin_t;
    // fidx: that position expressed as a (fractional) detector-bin index.
    const float fidx = s / ds + center;
    // j0: the integer bin just below fidx. We use a plain truncation toward
    // -inf via a manual floor so host and device agree for negative fidx too.
    const int j0 = (int)(fidx >= 0.0f ? fidx : fidx - 1.0f);  // floor(fidx)
    // Guard: need both j0 and j0+1 inside [0, n_det) to interpolate.
    if (j0 < 0 || j0 + 1 >= n_det) return 0.0f;
    // w: fractional distance of fidx past bin j0, in [0,1) -> linear weight.
    const float w = fidx - (float)j0;
    // Linear interpolation between the two straddling detector bins.
    return row[j0] * (1.0f - w) + row[j0 + 1] * w;
}
