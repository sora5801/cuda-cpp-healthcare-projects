// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (cuFFT-based RL deconvolution)
// ---------------------------------------------------------------------------
// Project 4.29 : Light-Sheet Microscopy Reconstruction
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls deconvolve_gpu(); kernels.cu
//   implements the host wrapper plus a few tiny element-wise device kernels.
//   Included only by .cu translation units (it declares __global__ kernels), so
//   the plain C++ compiler never sees it -- that is why LsfmData lives in the
//   pure-C++ reference_cpu.h, which this header includes.
//
// THE BIG IDEA (the catalog's pattern: "cuFFT for Fourier-domain deconvolution")
//   Richardson-Lucy needs TWO convolutions per iteration (a re-blur and an
//   adjoint back-projection). Convolution in real space is O(N^2) per pixel; in
//   FREQUENCY space it is a single element-wise MULTIPLY, because the convolution
//   theorem says  F(a conv b) = F(a) . F(b). So the recipe is:
//       X = FFT(image);  Hf = FFT(psf);  product = X . Hf;  result = IFFT(product)
//   cuFFT runs the forward/inverse FFTs on the GPU; our own tiny kernels do the
//   element-wise complex multiply and the two per-pixel RL steps (rl_core.h).
//   That is the whole GPU deconvolution -- and it is exactly how BigStitcher and
//   DeconvolutionLab2 accelerate LSFM (see README "Prior art").
//
//   We precompute F(psf) ONCE (the PSF never changes), so each RL iteration is
//   just: 1 forward FFT, a complex multiply, 1 inverse FFT (the re-blur), the
//   ratio kernel, then the same again with conj(F(psf)) for the adjoint, then the
//   multiply-update kernel. See kernels.cu for the fully-commented loop.
//
// WHY DOUBLE PRECISION (D2Z / Z2D)
//   The CPU reference uses double. To keep GPU==CPU agreement tight across many
//   iterations we use cuFFT's DOUBLE-precision real<->complex transforms (D2Z and
//   Z2D) rather than the single-precision R2C/C2R. The only remaining difference
//   is FFT round-off vs. direct-DFT round-off -- tiny, and the source of the
//   small documented tolerance (THEORY.md "Numerical considerations").
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, rl_core.h. Then kernels.cu.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // LsfmData (pure C++, safe to include in a .cu)

// cufftDoubleComplex is a struct { double x, y; }; we forward-declare our kernels
// in terms of double2 (identical layout) so this header needs no <cufft.h>.
struct double2;   // CUDA's built-in vector type (defined in vector_types.h)

// ---- Device kernels (all "one thread per pixel", the fundamental mapping) ---

// complex_mul_scaled: out[i] = a[i] * b[i] * scale   (element-wise complex product)
//   Used to multiply an image's spectrum by the PSF spectrum (the convolution
//   theorem) and to fold in cuFFT's 1/N normalization via `scale`. If `conj_b`
//   is true it uses conj(b[i]) -- turning convolution into CORRELATION, i.e. the
//   adjoint (flipped-PSF) step of Richardson-Lucy. One thread per frequency bin.
__global__ void complex_mul_scaled(const double2* __restrict__ a,
                                   const double2* __restrict__ b,
                                   int n, double scale, bool conj_b,
                                   double2* __restrict__ out);

// ratio_kernel: out[i] = rl_ratio(measured[i], reblurred[i])  (shared rl_core.h).
//   The per-pixel correction ratio b/reblur. One thread per pixel.
__global__ void ratio_kernel(const double* __restrict__ measured,
                             const double* __restrict__ reblurred,
                             int n, double* __restrict__ out);

// update_kernel: est[i] = rl_apply(est[i], correction[i])  (shared rl_core.h).
//   The multiplicative RL update, in place on the estimate. One thread per pixel.
__global__ void update_kernel(double* __restrict__ est,
                             const double* __restrict__ correction, int n);

// ---- Host wrapper --------------------------------------------------------
// deconvolve_gpu: run the whole Richardson-Lucy deconvolution on the GPU with
//   cuFFT, mirroring deconvolve_cpu() exactly. Returns the deblurred estimate
//   (size H*W, row-major) in `estimate`, and the total GPU time of all iterations
//   (FFTs + kernels) via *kernel_ms. All CUDA/cuFFT bookkeeping is hidden here.
//
//   d          : the loaded measurement + parameters (H, W, sigma, iters)
//   estimate   : host output, resized to H*W (output parameter)
//   kernel_ms  : out-param, milliseconds spent on the GPU (teaching artifact)
void deconvolve_gpu(const LsfmData& d, std::vector<double>& estimate,
                    float* kernel_ms);
