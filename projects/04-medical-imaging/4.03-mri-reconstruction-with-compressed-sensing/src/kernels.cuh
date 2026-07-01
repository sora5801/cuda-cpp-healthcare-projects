// ===========================================================================
// src/kernels.cuh  --  GPU CS-MRI reconstruction interface (cuFFT + FISTA)
// ---------------------------------------------------------------------------
// Project 4.3 : MRI Reconstruction with Compressed Sensing
//
// THE BIG IDEA (the catalog pattern: "cuFFT for gridded FFT" + custom kernels)
//   Compressed-sensing reconstruction is FISTA: repeat {forward FFT, mask against
//   the measured k-space, inverse FFT, gradient step, sparsity soft-threshold,
//   momentum extrapolation}. The two FFTs per iteration dominate the cost, and the
//   FFT is a solved problem -- so we hand it to cuFFT (NOT a black box: kernels.cu
//   documents exactly what cufftExecC2C computes and its layout). The cheap
//   per-pixel steps (masking, prox-gradient, momentum) become three tiny custom
//   kernels, each ONE THREAD PER PIXEL over the n*n image.
//
//   Crucially, every per-pixel formula those kernels use comes from cs_core.h --
//   the SAME inline functions the CPU reference calls -- so the CPU and GPU differ
//   only in the FFT engine (our radix-2 vs cuFFT). See ../THEORY.md "GPU mapping".
//
//   kernels.cu defines the kernels + the reconstruct_gpu wrapper; main.cu calls
//   reconstruct_gpu() and compares its image to reconstruct_cpu().
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, cs_core.h, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // KSpaceData (pure C++, safe to include in a .cu)
#include "cs_core.h"         // Cplx + the shared per-pixel math (also used device-side)

// ---------------------------------------------------------------------------
// Device kernels (one thread per pixel; declarations here, bodies in kernels.cu).
// Each takes the flat pixel count `total = n*n` and guards its ragged last block.
// ---------------------------------------------------------------------------

// kspace_residual_kernel: r = M (F z - y) in k-space. Reads the FFT of z, the
// measured k-space y, and the mask; writes the masked residual back over `fz`.
__global__ void kspace_residual_kernel(Cplx* __restrict__ fz,
                                       const Cplx* __restrict__ y,
                                       const int* __restrict__ mask,
                                       int total);

// prox_grad_kernel: the ISTA update x = soft_threshold(z - t*grad, t*lambda).
// grad is the inverse FFT of the residual (passed in `grad`). Writes new x.
__global__ void prox_grad_kernel(const Cplx* __restrict__ z,
                                 const Cplx* __restrict__ grad,
                                 float t, float lambda,
                                 Cplx* __restrict__ x, int total);

// momentum_kernel: FISTA extrapolation z = x + beta*(x - x_prev). Writes new z.
__global__ void momentum_kernel(const Cplx* __restrict__ x,
                                const Cplx* __restrict__ x_prev,
                                float beta, Cplx* __restrict__ z, int total);

// scale_kernel: multiply every pixel by a real scalar (used for the 1/(n*n) that
// normalizes cuFFT's un-normalized inverse transform, matching ifft2_cpu).
__global__ void scale_kernel(Cplx* __restrict__ v, float s, int total);

// magnitude_kernel: out[i] = |x[i]| -- the final radiologist-facing image.
__global__ void magnitude_kernel(const Cplx* __restrict__ x,
                                 float* __restrict__ out, int total);

// ---------------------------------------------------------------------------
// reconstruct_gpu: the host-callable "do the whole CS reconstruction on the GPU".
//   Builds two cuFFT plans (forward + inverse C2C), warm-starts from the
//   zero-filled adjoint image, runs d.iters FISTA iterations entirely on-device
//   (no per-iteration host<->device copies), and returns the magnitude image plus
//   the GPU time of the iteration loop.
//     * d         : the loaded problem (k-space, mask, lambda, iters)
//     * out_mag   : host output, resized to n*n (the |x| image)
//     * kernel_ms : out-param, GPU-measured ms for the FISTA loop (FFTs + kernels)
// ---------------------------------------------------------------------------
void reconstruct_gpu(const KSpaceData& d, std::vector<float>& out_mag, float* kernel_ms);
