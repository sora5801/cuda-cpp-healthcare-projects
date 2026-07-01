// ===========================================================================
// src/kernels.cuh  --  GPU oART interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 5.14 : GPU-Accelerated Adaptive MR-Linac Workflow (reduced-scope)
//
// THE BIG IDEA (pattern: per-voxel GATHER + STENCIL, host-driven iteration)
//   Online adaptive radiotherapy on an MR-Linac is a *pipeline of image
//   operations*, and every one of them is embarrassingly parallel per voxel:
//     * warp an image  -> each output voxel independently gathers (bilinear) from
//       a source location -> one thread per output voxel (GATHER, like 4.01).
//     * demons force    -> each voxel reads its own intensities + a 3x3 gradient
//       stencil of the fixed image -> one thread per voxel (STENCIL, like 6.04).
//     * Gaussian smooth -> a separable convolution, one thread per output voxel.
//   The HOST drives the outer Demons iteration (warp -> add force -> smooth),
//   launching a handful of kernels per iteration and keeping all state resident
//   on the device between launches. This "host loop, device kernels, no D2H in
//   the loop" structure is exactly how a real oART engine overlaps its stages.
//
//   Every kernel calls the SAME shared per-voxel functions the CPU reference
//   uses (mrl_registration.h), so the GPU reproduces the CPU result. Only the
//   Gaussian smoother is written twice (a device kernel here, a host loop in
//   reference_cpu.cpp) -- but with identical weights and clamp policy, so the
//   arithmetic still matches.
//
// WHY THIS IS "REDUCED SCOPE"
//   The clinical chain also includes NUFFT MRI reconstruction (cuFFT), a CNN for
//   synthetic-CT (cuDNN), a full 3-D collapsed-cone/Monte-Carlo dose engine, and
//   a fluence re-optimiser (cuSPARSE). We teach the registration + dose-mapping
//   heart on a 2-D slice; ../THEORY.md maps each simplification to the real tool.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, mrl_registration.h.
//   Then read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // OartCase / OartResult (pure C++, safe in a .cu)

// ---- Host wrapper --------------------------------------------------------
// oart_gpu: run the full reduced-scope oART workflow on the GPU.
//   Uploads the case, runs `iters` Demons iterations (warp + force + smooth) with
//   everything resident on the device, warps the dose, copies the fields back,
//   and computes the same metrics as the CPU path (via compute_metrics on the
//   host, for identical arithmetic). Returns the total GPU time of the registration
//   + warp kernels (CUDA-event measured) in *kernel_ms.
//
//   c         : the input case (images + parameters), read-only
//   r         : filled with u, v, warped_dose, warped_moving and metrics
//   kernel_ms : out-param, milliseconds spent in the GPU kernels (not H2D/D2H)
void oart_gpu(const OartCase& c, OartResult& r, float* kernel_ms);
