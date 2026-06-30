// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 2.31 : Cryo-EM Tilt-Series Alignment & Tomogram Reconstruction
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls the two host wrappers here:
//   ramp_filter_gpu() (cuFFT) and backproject_gpu() (custom gather kernel).
//   kernels.cu implements both, plus the device kernels. Included only by .cu
//   translation units (it declares __global__ kernels, so the plain C++ host
//   compiler must never see it -- that is why the CPU reference lives in the
//   separate pure-C++ reference_cpu.h).
//
// THE TWO GPU TEACHING POINTS
//   1. RAMP FILTER with cuFFT (the "Weighted" in Weighted Back-Projection).
//      We forward-FFT each aligned projection (batched real-to-complex), multiply
//      every spectral bin by the ramp |f| (with a cosine roll-off to tame noise),
//      then inverse-FFT back to real space. cuFFT does the O(n log n) transform;
//      a tiny element-wise kernel applies the ramp. This is the catalog's named
//      CUDA pattern: "cuFFT for filter application in filtered back-projection".
//
//   2. BACK-PROJECTION as a per-PIXEL GATHER. Every output pixel is independent,
//      so we give each pixel its own thread (a 2-D grid over the slice). Each
//      thread loops over all tilt angles, finds where its ray hits that tilt's
//      detector (s = wx*cos + wy*sin), linearly interpolates the filtered value
//      there (via the SHARED wbp_core.h math), and accumulates. No atomics, no
//      shared memory -- the canonical tomographic reconstruction kernel.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, wbp_core.h. Then
// read kernels.cu for the implementations.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // TiltSeries (pure C++, safe to include in a .cu)

// ---- GPU step 2: ramp filter via cuFFT -----------------------------------
// ramp_filter_gpu: forward R2C FFT each aligned projection, multiply by the ramp
//   |f| (cosine-apodized), inverse C2R FFT, normalize. Mathematically equivalent
//   to ramp_filter_cpu()'s spatial convolution; the two are checked against each
//   other in main.cu within a documented tolerance.
//     ts        : geometry (n_tilts, n_det, ds).
//     aligned   : [n_tilts*n_det] drift-corrected projections (host).
//     filtered  : resized to aligned.size(); the ramp-filtered projections (host).
//     kernel_ms : out-param, GPU time for FFT + ramp + inverse FFT (ms).
void ramp_filter_gpu(const TiltSeries& ts, const std::vector<float>& aligned,
                     std::vector<float>& filtered, float* kernel_ms);

// ---- Device kernel: ramp multiply in the frequency domain ----------------
// ramp_apply_kernel: one thread per (tilt, spectral bin). Multiplies the complex
//   spectrum X[k][f] by the real ramp weight ramp[f] in place. Declared here so
//   the launch is visible; defined in kernels.cu.
//     spec    : [n_tilts*nf] complex spectra (as float2: .x real, .y imag).
//     ramp    : [nf] real ramp weights (|f| with apodization).
//     nf      : bins per projection spectrum (= n_det/2 + 1 for R2C).
//     total   : n_tilts*nf (guards the ragged last block).
__global__ void ramp_apply_kernel(float2* __restrict__ spec,
                                  const float* __restrict__ ramp,
                                  int nf, int total);

// ---- GPU step 4: weighted back-projection gather --------------------------
// backproject_gpu: upload filtered sinogram + trig, launch the 2-D gather grid,
//   copy the reconstructed slice back, and report the kernel time.
//     ts        : geometry.
//     filtered  : [n_tilts*n_det] ramp-filtered projections (host).
//     cosv,sinv : [n_tilts] precomputed trig of the tilt angles (host).
//     slice     : resized to img*img; the reconstruction (host, output).
//     kernel_ms : out-param, GPU back-projection kernel time (ms).
void backproject_gpu(const TiltSeries& ts, const std::vector<float>& filtered,
                     const std::vector<float>& cosv, const std::vector<float>& sinv,
                     std::vector<float>& slice, float* kernel_ms);

// ---- Device kernel: the per-pixel gather ---------------------------------
// backproject_kernel: thread (px,py) reconstructs one slice pixel by summing its
//   contribution from every tilt (shared wbp_core.h sampler). Declared here;
//   defined in kernels.cu.
//     filtered : [n_tilts*n_det] ramp-filtered sinogram (device).
//     cosv,sinv: [n_tilts] precomputed trig (device).
//     n_tilts,n_det,N : counts; N = image side.
//     ds, center, W, pix, scale : geometry constants (see kernels.cu).
//     slice    : [N*N] output reconstruction (device).
__global__ void backproject_kernel(const float* __restrict__ filtered,
                                   const float* __restrict__ cosv,
                                   const float* __restrict__ sinv,
                                   int n_tilts, int n_det, int N,
                                   float ds, float center, float W,
                                   float pix, float scale,
                                   float* __restrict__ slice);
