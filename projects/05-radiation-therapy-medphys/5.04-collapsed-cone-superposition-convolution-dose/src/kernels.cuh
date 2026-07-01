// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for collapsed-cone SC dose
// ---------------------------------------------------------------------------
// Project 5.4 : Collapsed-Cone / Superposition-Convolution Dose  (2-D teaching model)
//
// THE BIG IDEA (two independent, embarrassingly-parallel stages)
//   STAGE 1  TERMA ray-trace: the beam columns are independent, so we give ONE
//            THREAD PER COLUMN. Each thread marches its column top-to-bottom,
//            accumulating radiological depth and writing TERMA per voxel. No two
//            threads touch the same voxel -> no atomics.
//
//   STAGE 2  Collapsed-cone superposition: every source voxel spreads its TERMA
//            along the n_cones cone rays. We give ONE THREAD PER SOURCE VOXEL;
//            each thread walks all its cones and DEPOSITS into the dose grid.
//            Different source voxels' rays overlap, so deposits use atomicAdd.
//            The trick that keeps it deterministic: we accumulate INTEGER
//            dose-units (ccc_physics.h::dose_to_units), and integer atomicAdd is
//            associative -> the GPU grid is bit-identical to the CPU grid AND
//            reproducible run to run (PATTERNS.md §3). A float atomicAdd would
//            give a different sum every launch.
//
//   Both kernels call the SAME per-voxel physics that the CPU reference calls
//   (ccc_physics.h), which is exactly why main.cu can assert the two dose grids
//   are equal to the last integer.
//
// This header contains a __global__ declaration, so ONLY .cu files may include
// it. The pure-C++ CPU reference uses reference_cpu.h instead.
//
// READ THIS AFTER: ccc_physics.h, util/cuda_check.cuh, util/timer.cuh.
// Then read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // DoseProblem, CccParams (pure C++, safe in a .cu)

// ---- Device kernels (documented fully at their definitions in kernels.cu) ----

// STAGE 1: one thread per irradiated beam column; writes TERMA per voxel.
__global__ void terma_kernel(CccParams P,
                             const float* __restrict__ rho,   // [nx*ny] density map
                             double* __restrict__ terma);      // [nx*ny] output TERMA

// STAGE 2: one thread per source voxel; scatters collapsed-cone dose as integer
//   dose-units into `dose_units` via atomicAdd.
__global__ void ccc_kernel(CccParams P,
                          const float* __restrict__ rho,          // [nx*ny] density map
                          const double* __restrict__ terma,        // [nx*ny] TERMA from stage 1
                          long long* __restrict__ dose_units);      // [nx*ny] integer dose tally

// ---- Host wrapper --------------------------------------------------------
// dose_gpu: run BOTH stages on the device and return the integer dose grid.
//   prob        : the loaded problem (geometry + density map)
//   dose_units  : host output, resized to nx*ny (integer dose-units)
//   terma_out   : host output, resized to nx*ny (the stage-1 TERMA, for reporting
//                 and for the optional stage-1 CPU/GPU cross-check in main.cu)
//   kernel_ms   : out-param, total GPU kernel time (both stages), milliseconds
void dose_gpu(const DoseProblem& prob,
              std::vector<long long>& dose_units,
              std::vector<double>& terma_out,
              float* kernel_ms);
