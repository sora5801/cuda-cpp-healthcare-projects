// ===========================================================================
// src/kernels.cuh  --  GPU SD-OCT reconstruction interface (cuFFT + kernels)
// ---------------------------------------------------------------------------
// Project 4.12 : Optical Coherence Tomography Processing (SD-OCT reconstruction)
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls reconstruct_gpu(); kernels.cu
//   implements the host wrapper (which drives cuFFT) and the custom device
//   kernels. Included only by .cu translation units (it declares __global__
//   kernels and uses float2, so the plain C++ compiler must never see it -- that
//   is why the CPU reference lives in a separate pure-C++ header).
//
// THE BIG IDEA (two GPU patterns in one project)
//   1. CUSTOM KERNEL -- dispersion compensation + preprocessing. Each spectral
//      sample of each A-scan is transformed independently (DC removal, Hann
//      window, dispersion phase), so we assign ONE THREAD PER (A-scan, sample):
//      the classic 1-D grid over a flattened 2-D array. This is the step a
//      library CANNOT do for us -- it is OCT-specific physics.
//   2. LIBRARY CALL -- the FFT. Transforming a length-N spectrum into a depth
//      profile is a solved problem with a superb GPU library, cuFFT. We batch
//      ALL n_ascan FFTs into a SINGLE cufftExecC2C call. The lesson is using the
//      library WITHOUT it being a black box: kernels.cu documents exactly what it
//      computes, the batched layout it expects, and what hand-rolling would take.
//   A final custom kernel takes |A|^2 and normalises each A-scan (the log/display
//   step). So: custom kernel -> cuFFT -> custom kernel. That "wrap a library call
//   in bespoke kernels" shape is how real OCT engines are built.
//
// READ THIS AFTER: oct_core.h, util/cuda_check.cuh, util/timer.cuh, then kernels.cu.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // OctBscan, oct_depth_count (pure C++, safe in .cu)

// ---- Device kernel 1: preprocess + dispersion-compensate -----------------
// preprocess_kernel: fill the cuFFT input buffer with the complex, DC-removed,
// windowed, dispersion-corrected spectrum. ONE THREAD PER SPECTRAL SAMPLE.
//   raw    : [n_ascan*n_spec] raw real spectra (device)
//   dc     : [n_ascan] per-A-scan mean (device; precomputed by dc_kernel)
//   n_ascan,n_spec : dimensions
//   a2,a3  : dispersion coefficients (broadcast to every thread by value)
//   out    : [n_ascan*n_spec] float2 cuFFT input (device; .x real, .y imag)
// Thread t = blockIdx.x*blockDim.x+threadIdx.x owns global sample t; its A-scan
// is t / n_spec and its in-scan sample index is t % n_spec.
__global__ void preprocess_kernel(const float* __restrict__ raw,
                                  const double* __restrict__ dc,
                                  int n_ascan, int n_spec,
                                  double a2, double a3,
                                  float2* __restrict__ out);

// ---- Device kernel 2: per-A-scan DC (mean) -------------------------------
// dc_kernel: one thread per A-scan sums its N raw samples and stores the mean.
// A-scans are short (N ~ 10^3), so a simple per-thread serial sum is clearest and
// plenty fast; a block-reduction would be premature optimisation here (THEORY).
__global__ void dc_kernel(const float* __restrict__ raw, int n_ascan, int n_spec,
                          double* __restrict__ dc);

// ---- Device kernel 3: magnitude + per-A-scan normalise -------------------
// power_norm_kernel: for each A-scan (one thread), read its N/2 complex FFT
// outputs, compute |A[z]|^2, find the peak, and write the normalised power
// (0..1) into the image. Keeping the reduction inside ONE thread per A-scan makes
// the normalisation deterministic (no cross-thread float atomics; PATTERNS.md #3).
//   fft    : [n_ascan*n_spec] full complex FFT output (device); we read [0..N/2).
//   image  : [n_ascan*(N/2)] normalised linear power (device, double).
__global__ void power_norm_kernel(const float2* __restrict__ fft,
                                  int n_ascan, int n_spec,
                                  double* __restrict__ image);

// ---- Host wrapper --------------------------------------------------------
// reconstruct_gpu: the host-callable "do the whole GPU reconstruction".
//   Allocates device buffers, copies the raw B-scan H2D, runs
//   dc_kernel -> preprocess_kernel -> cufftExecC2C (batched) -> power_norm_kernel,
//   copies the normalised image D2H, and reports the measured GPU time
//   (custom kernels + FFT, via CUDA events) in *kernel_ms. main.cu calls exactly
//   this; all CUDA/cuFFT bookkeeping is hidden here.
//     b        : the raw B-scan (host)
//     image    : host output, resized to n_ascan*(N/2) (output parameter, double)
//     kernel_ms: out-param, milliseconds spent on the GPU compute (not H2D/D2H)
void reconstruct_gpu(const OctBscan& b, std::vector<double>& image, float* kernel_ms);
