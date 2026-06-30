// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (cuFFT Richardson-Lucy)
// ---------------------------------------------------------------------------
// Project 4.30 : Deconvolution Microscopy
//
// THE BIG IDEA (this project's pattern: USING cuFFT FOR FFT CONVOLUTION)
//   Richardson-Lucy deconvolution spends almost all its time in two
//   convolutions per iteration. The convolution theorem turns each O(N*K)
//   spatial convolution into a cheap O(N) pointwise multiply in the FREQUENCY
//   domain, bracketed by an FFT and an inverse FFT:
//
//       conv(a, b)  =  IFFT( FFT(a) .* FFT(b) )
//
//   The FFT is a solved problem with a world-class GPU library -- cuFFT. The
//   lesson here, like flagship 8.03, is to use that library WITHOUT it being a
//   black box: kernels.cu documents exactly what each cufftExec call computes,
//   the real-to-complex layout it uses, and cuFFT's UNNORMALIZED convention
//   (a forward+inverse round trip scales by N, which we divide back out).
//
//   The only CUSTOM kernels we write are the trivial element-wise steps:
//     * complex pointwise multiply (apply the PSF / its adjoint in frequency),
//     * the real-space RL ratio and multiplicative update (shared rl_core.h).
//   Each is a "one GPU thread per element" map -- the most basic CUDA pattern.
//
//   kernels.cu defines the kernels + the host wrapper deconvolve_rl_gpu().
//   main.cu calls that wrapper. The per-pixel RL math lives in rl_core.h and is
//   compiled for BOTH host and device, so GPU and CPU agree (PATTERNS.md section 2).
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, reference_cpu.h,
//   rl_core.h. Then read kernels.cu for the cuFFT mechanics.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // Image, Psf (pure C++ structs, safe inside a .cu)

// ---- Device kernels (declared here, defined in kernels.cu) ----------------
//
// We forward-declare the custom element-wise kernels so the interface is
// visible, but their parameters use cuFFT's complex type. To avoid pulling
// <cufft.h> into every includer we declare them with float2/double2-free
// wrappers inside kernels.cu instead; here we expose only the HOST entry point.
// (The kernels themselves are documented in detail at their definitions.)

// ---- Host wrapper ---------------------------------------------------------
//
// deconvolve_rl_gpu: run `iters` Richardson-Lucy iterations entirely on the GPU
//   using cuFFT for both convolutions, and return the deconvolved image.
//
//   observed  : the blurry input image (host; copied to the device once)
//   psf       : the blur kernel (host; embedded into a full-size, FFT-shifted
//               image and transformed once on the device -- its spectrum is
//               reused every iteration)
//   iters     : number of RL iterations (same count as the CPU reference)
//   out       : host output image, filled with the deconvolved estimate
//   kernel_ms : out-param, GPU milliseconds for the RL iteration loop (the FFTs
//               + element-wise kernels; excludes one-time setup and copies)
//
// This is the single function main.cu calls; all device allocation, the cuFFT
// plans, and the iteration loop are encapsulated here.
void deconvolve_rl_gpu(const Image& observed, const Psf& psf, int iters,
                       Image& out, float* kernel_ms);
