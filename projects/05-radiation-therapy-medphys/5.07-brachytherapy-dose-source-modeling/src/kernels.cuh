// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 5.7 : Brachytherapy Dose & Source Modeling
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls dose_gpu(); kernels.cu
//   implements both the host wrapper and the device kernel. Included only by
//   .cu translation units (it declares a __global__ kernel, so the plain C++
//   compiler must never see it -- that is why the CPU reference and the shared
//   Plan struct live in the pure-C++ reference_cpu.h).
//
// THE BIG IDEA -- per-voxel threads, inner loop over dwells, tables in constant
//   The dose at each voxel is INDEPENDENT of every other voxel, so we assign
//   ONE GPU THREAD PER VOXEL. Each thread loops over the (few) dwell positions
//   and superposes their TG-43 dose rate (dose_rate_one_dwell from
//   tg43_physics.h -- the same function the CPU reference calls). The source's
//   TG-43 tables and the dwell list are read by EVERY thread but never change
//   during the launch, so they live in __constant__ memory: reads broadcast
//   through the constant cache instead of hammering global memory. This is
//   exactly the catalog's prescribed pattern for 5.7.
//
//   Grid mapping: flatten the 3-D grid to N = nx*ny*nz voxels; launch
//   ceil(N / B) blocks of B threads; thread t owns flat voxel index
//   i = blockIdx.x * blockDim.x + threadIdx.x (decoded back to ix,iy,iz).
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, tg43_physics.h. Then
//   read kernels.cu for the launch + the constant-memory upload.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // Plan, DoseGrid (pure C++, safe to include in .cu)

// ---- Host wrapper --------------------------------------------------------
// dose_gpu: the host-callable "do the whole GPU dose calculation" function.
//   1. Upload the SourceModel + dwell list into __constant__ memory.
//   2. Allocate the device dose buffer (grid.size() floats).
//   3. Launch dose_kernel with one thread per voxel.
//   4. Copy the dose back to the host and report the measured KERNEL time.
//
//   plan      : the complete TG-43 job (source, dwells, grid)
//   dose      : host output, resized to grid.size() (output parameter), cGy/h
//   kernel_ms : out-param, milliseconds in the kernel itself (not the copies)
//
//   main.cu calls exactly this; all CUDA bookkeeping is hidden inside kernels.cu.
void dose_gpu(const Plan& plan, std::vector<float>& dose, float* kernel_ms);
