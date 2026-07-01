// ===========================================================================
// src/kernels.cuh  --  GPU fluence-map-optimization interface (cuSPARSE SpMV)
// ---------------------------------------------------------------------------
// Project 5.2 : Radiotherapy Treatment-Plan Optimization
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls optimize_gpu(); kernels.cu
//   implements the projected-gradient optimizer on the device. The two SpMVs
//   (d = D x and g = D^T r) that dominate each iteration are done by cuSPARSE;
//   the cheap per-voxel residual and per-beamlet projected-update steps are two
//   small hand-written kernels declared here.
//
// THE BIG IDEA (library pattern: cuSPARSE SpMV on a GPU-resident CSR matrix)
//   The dose-influence matrix D (n_vox x n_beam) is uploaded ONCE to the device
//   in CSR form and stays there. Every optimizer iteration then reads it twice:
//     * cusparseSpMV(op = NON_TRANSPOSE) computes dose = D * fluence.
//     * cusparseSpMV(op = TRANSPOSE)     computes grad = D^T * residual.
//   cuSPARSE's SpMV is the canonical GPU sparse kernel -- it parallelizes the
//   per-row dot products across thousands of threads with a tuned load balance
//   we would otherwise hand-roll (see THEORY.md section 4 for what that entails).
//   Because D is resident, each iteration is milliseconds even for millions of
//   voxels, which is what makes real-time adaptive re-planning possible.
//
//   The per-element math (residual, projected update) is shared with the CPU
//   reference through fmo.h, so the only CPU-vs-GPU divergence is float
//   summation order inside the SpMVs -> bounded by a documented tolerance.
//
// READ THIS AFTER: fmo.h, reference_cpu.h, util/cuda_check.cuh, util/timer.cuh.
//   Then read kernels.cu for the cuSPARSE plumbing and the two kernels.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // Problem, PlanStats (pure C++, safe under nvcc)

// ---- Device kernels (defined in kernels.cu) ------------------------------
// residual_kernel: thread v computes r[v] = voxel_residual(spec[v], dose[v]).
//   One thread per voxel; reads the current dose, writes the gradient-driving
//   residual. Fully independent -> no shared memory or atomics. VoxelSpec is a
//   POD from fmo.h so it is safe to pass a device pointer to it.
__global__ void residual_kernel(int n_vox, const VoxelSpec* __restrict__ spec,
                                const float* __restrict__ dose,
                                float* __restrict__ resid);

// update_kernel: thread j applies the projected gradient step to beamlet j:
//   x[j] = max(0, x[j] - step * grad[j]).  One thread per beamlet; independent.
__global__ void update_kernel(int n_beam, float step,
                              const float* __restrict__ grad,
                              float* __restrict__ x);

// ---- Host wrapper --------------------------------------------------------
// optimize_gpu: run the full projected-gradient FMO on the GPU and return the
//   optimized fluence. Uploads the CSR matrix once, then loops `iters` times:
//   cuSPARSE SpMV (D x) -> residual_kernel -> cuSPARSE SpMV (D^T r) ->
//   update_kernel. All CUDA / cuSPARSE bookkeeping is hidden inside.
//     p         : the FMO problem (CSR matrix + specs + hyper-params).
//     x_out     : optimized fluence, resized to n_beam (output parameter).
//     total_ms  : out-param, GPU time for the whole optimization loop (ms,
//                 CUDA-event measured) -- a teaching artifact, not a benchmark.
void optimize_gpu(const Problem& p, std::vector<float>& x_out, float* total_ms);
