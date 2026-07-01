// ===========================================================================
// src/kernels.cuh  --  GPU projection kernels + MLEM driver (interface)
// ---------------------------------------------------------------------------
// Project 4.5 : PET Image Reconstruction (MLEM / OS-EM)
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls mlem_gpu(); kernels.cu
//   implements the host driver plus the two device kernels. Included only by
//   .cu translation units (it declares __global__ kernels, which the plain C++
//   host compiler must never see) -- that is why the CPU reference lives in the
//   separate pure-C++ header reference_cpu.h.
//
// THE BIG IDEA (the projection GATHER pattern -- docs/PATTERNS.md, like 4.01)
//   One MLEM iteration is two projections, and BOTH are gathers of independent
//   outputs, so each maps onto a 1 output = 1 thread grid with NO atomics:
//
//     * FORWARD  y_hat = A x   -- one thread per LOR (k,j). The thread sweeps the
//                 image and gathers every pixel whose ray falls in bin j at angle
//                 k (linear split). Output element (k,j) is written by exactly one
//                 thread -> deterministic, no atomics.
//
//     * UPDATE   x <- x * (A^T ratio)/s  -- one thread per pixel. The thread
//                 gathers the back-projected ratio over all K angles (same split
//                 weights, transpose-consistent) and applies the multiplicative
//                 MLEM step in place. Again one output per thread.
//
//   Because both kernels call the SAME pet_geometry.h helpers the CPU reference
//   uses, the arithmetic matches to FMA rounding (verified in main.cu).
//
//   Determinism note (docs/PATTERNS.md §3): we deliberately AVOID the alternative
//   "LOR-parallel backprojection with atomicAdd into the image", because a float
//   atomic sum depends on thread order and would make the demo's stdout vary.
//   Gather keeps every reduction inside one thread, in a fixed order.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, reference_cpu.h. Then
// read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // PetProblem, PetGeom (pure C++ -> safe inside .cu)

// ---- Device kernels ------------------------------------------------------
// forward_project_kernel: thread (k,j) computes y_hat[k*D+j] = SUM over pixels
//   whose linear split lands in detector bin j at angle k. cosv/sinv are the
//   host-precomputed trig tables (device pointers). See kernels.cu for the
//   launch configuration and the thread->LOR mapping.
__global__ void forward_project_kernel(PetGeom g,
                                       const float* __restrict__ image,
                                       const float* __restrict__ cosv,
                                       const float* __restrict__ sinv,
                                       float* __restrict__ yhat);

// update_kernel: thread (px,py) back-projects the ratio over all angles and
//   applies the in-place MLEM multiplicative update
//       image[j] <- image[j] * corr[j] / sens[j]   (guarded when sens==0).
//   ratio is the precomputed y/y_hat sinogram (device pointer).
__global__ void update_kernel(PetGeom g,
                              const float* __restrict__ ratio,
                              const float* __restrict__ cosv,
                              const float* __restrict__ sinv,
                              const float* __restrict__ sens,
                              float* __restrict__ image);

// ---- Host driver ---------------------------------------------------------
// mlem_gpu: run the FULL MLEM reconstruction on the GPU.
//   Uploads the geometry, trig tables, measured counts, and sensitivity once,
//   then loops `iters` times: forward_project_kernel -> compute ratio (a cheap
//   per-LOR kernel) -> update_kernel. Copies the final image back to `image`
//   (resized to N*N) and reports the summed kernel time via *kernel_ms.
//   main.cu calls exactly this; all CUDA bookkeeping is hidden here.
//
//   p         : the loaded problem (geometry, counts, trig)
//   sens      : host sensitivity image A^T 1 (length N*N), precomputed once
//   iters     : number of MLEM iterations
//   image     : host output, resized to N*N (output parameter)
//   kernel_ms : out-param, total milliseconds across all kernel launches
void mlem_gpu(const PetProblem& p, const std::vector<float>& sens, int iters,
              std::vector<float>& image, float* kernel_ms);
