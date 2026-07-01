// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for the per-voxel fMRI GLM
// ---------------------------------------------------------------------------
// Project 4.16 : Functional MRI Analysis
//
// THE BIG IDEA
//   Fitting the GLM at V voxels is V INDEPENDENT least-squares solves against
//   the SAME design matrix X (docs/PATTERNS.md §1, the "many identical small
//   solves" pattern, cf. the 9.02 SEIR ensemble). So we give each voxel its own
//   GPU thread; a grid-stride loop lets one modest grid cover an arbitrarily
//   large brain. Each thread:
//     * reads its voxel's time-series row y (length T) from global memory,
//     * calls fit_voxel() -- the SHARED __host__ __device__ core in glm.h, the
//       exact same code the CPU reference runs -- so results match to ~1e-9,
//     * writes one t-statistic and one task-beta.
//
//   Two things are voxel-INDEPENDENT and identical for the whole launch: the
//   design parameters (GlmDesign) and the precomputed (X^T X)^-1. Those go into
//   CONSTANT memory: read by every thread, never written during the launch, so
//   the constant cache broadcasts them warp-wide in one transaction (exactly
//   the role the query fingerprint played in flagship 1.12).
//
//   This header is included only by .cu units. main.cu calls glm_gpu().
//
// READ THIS AFTER: glm.h (the science), reference_cpu.h (the data model),
//   util/cuda_check.cuh, util/timer.cuh. Then read kernels.cu. GPU-mapping
//   reasoning (occupancy, coalescing, the constant-memory choice) is in
//   ../THEORY.md §"GPU mapping".
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // FmriDataset, GlmDesign (pure C++, safe in .cu)

// Device kernel: one thread per voxel computes its GLM t-statistic and task-beta.
//   d_bold : [V * T] device array, voxel-major (row v = voxel v's time-series)
//   V, T   : dimensions
//   d_t    : [V] device output, per-voxel t-statistic (output)
//   d_beta : [V] device output, per-voxel task regression weight (output)
// The design params and (X^T X)^-1 are read from __constant__ symbols set up by
// glm_gpu(), NOT passed as parameters.
__global__ void glm_kernel(const double* __restrict__ d_bold, int V, int T,
                           double* __restrict__ d_t, double* __restrict__ d_beta);

// Host wrapper: uploads the design + inverse to constant memory and the BOLD
// data to global memory, launches the kernel, times ONLY the kernel (CUDA
// events), and returns the per-voxel t-stats and task-betas.
//   ds        : the loaded dataset (V voxels x T scans + design params)
//   XtX_inv   : the precomputed row-major 3x3 (X^T X)^-1 (from compute_XtX_inv)
//   tstat     : resized to V; filled with per-voxel t-statistics
//   beta      : resized to V; filled with per-voxel task weights
//   kernel_ms : out-param, GPU-measured kernel time in milliseconds
void glm_gpu(const FmriDataset& ds, const double XtX_inv[9],
             std::vector<double>& tstat, std::vector<double>& beta, float* kernel_ms);
