// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (cuFFT dipole inversion)
// ---------------------------------------------------------------------------
// Project 4.22 : Quantitative Susceptibility Mapping (QSM)
//
// THE BIG IDEA (this project's pattern: USE cuFFT FOR A 3-D SPECTRAL INVERSION)
//   QSM dipole inversion lives entirely in k-space. The forward physics is a
//   pointwise multiply by the dipole kernel D(k); the inverse is a pointwise
//   division (regularized) by D(k). The ONLY thing standing between the field
//   map (real space) and that per-bin arithmetic is a 3-D Fourier transform --
//   forward to get into k-space and inverse to get back:
//
//       chi  =  IFFT3( weight(k) .* FFT3(field) )
//
//   The 3-D FFT is a solved problem with a world-class GPU library -- cuFFT. The
//   lesson here, like flagship 8.03, is to use that library WITHOUT it being a
//   black box: kernels.cu documents exactly what each cufftExec call computes,
//   the double-precision complex layout it uses, and cuFFT's UNNORMALIZED
//   convention (a forward+inverse round trip scales by N, which we divide out).
//
//   The only CUSTOM kernels we write are trivial element-wise maps over the
//   k-space bins:
//     * apply a REAL per-bin weight (TKD 1/D_thr, or the Tikhonov Wiener weight),
//     * one Tikhonov GRADIENT step per bin (the iterative-solver structure).
//   Each is "one GPU thread per k-space bin" -- the most basic CUDA mapping. The
//   per-bin math itself is SHARED with the CPU via qsm_core.h, so GPU and CPU
//   agree to round-off (PATTERNS.md section 2 + 4).
//
//   kernels.cu defines the kernels + the three host wrappers below. main.cu
//   calls them. Included only by .cu translation units (it pulls in reference_cpu.h
//   for the Volume struct, which is pure C++ and safe inside a .cu).
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, reference_cpu.h,
//   qsm_core.h. Then read kernels.cu for the cuFFT mechanics.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // Volume (pure C++ struct, safe inside a .cu)

// ---- Host wrappers (declared here, defined in kernels.cu) ------------------
//
// All three take a real field-map Volume and return a reconstructed chi Volume,
// doing the 3-D FFTs on the GPU with cuFFT and the per-bin weighting/iteration in
// small custom kernels. `kernel_ms` returns the GPU time for the transform +
// weighting work (CUDA-event measured), so main.cu can print an honest timing.

// reconstruct_tkd_gpu: Threshold-based K-space Division on the GPU.
//   field     : measured field-shift volume (host; copied to device once)
//   thr       : TKD threshold (same value as the CPU reference)
//   out       : host output, filled with the reconstructed susceptibility volume
//   kernel_ms : out-param, GPU milliseconds for FFT + weight + IFFT
void reconstruct_tkd_gpu(const Volume& field, double thr,
                         Volume& out, float* kernel_ms);

// reconstruct_tikhonov_iter_gpu: ITERATIVE Tikhonov gradient descent on the GPU.
//   Runs the SAME per-bin gradient step as reconstruct_tikhonov_iter_cpu(), one
//   thread per k-space bin, for `iters` iterations, all on the device (the data
//   spectrum is transformed once and stays resident). This is the pattern the
//   catalog highlights: O(100) iterations of 3-D FFT + gradient updates.
//   field     : measured field-shift volume (host)
//   alpha     : Tikhonov weight (same as the CPU reference)
//   step      : gradient-descent step size (same as the CPU reference)
//   iters     : number of gradient iterations (same as the CPU reference)
//   out       : host output, reconstructed susceptibility volume
//   kernel_ms : out-param, GPU milliseconds for the whole iterative solve
void reconstruct_tikhonov_iter_gpu(const Volume& field, double alpha,
                                   double step, int iters,
                                   Volume& out, float* kernel_ms);
