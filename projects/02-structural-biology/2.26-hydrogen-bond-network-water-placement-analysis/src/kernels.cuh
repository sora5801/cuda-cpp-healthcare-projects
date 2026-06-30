// ===========================================================================
// src/kernels.cuh  --  GPU GIST interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 2.26 : Hydrogen Bond Network & Water Placement Analysis
//
// THE BIG IDEA  (pattern: GRID ACCUMULATION WITH ATOMIC UPDATES)
//   GIST is a SCATTER reduction. There are nframes * waters_per_frame independent
//   (water, frame) samples; each one:
//       1. finds the voxel it occupies                       (read-only lookup),
//       2. computes its water<->solute interaction energy    (independent),
//       3. ATOMICALLY adds 1 to that voxel's occupancy and its energy into the
//          voxel's running sum                               (colliding writes).
//   We give ONE THREAD PER SAMPLE: thread t handles sample t = frame*W + water.
//   Many threads land in the same voxel, so the accumulation uses atomicAdd. To
//   keep the result DETERMINISTIC and CPU-matching, the energy sum is accumulated
//   in FIXED-POINT integers (gist.h) -- integer atomic adds commute, unlike float.
//   This is the same pattern as Monte-Carlo dose (5.01) and k-means accumulate
//   (11.09); see docs/PATTERNS.md.
//
//   After the kernel, the raw tallies are copied back and the SHARED host helper
//   derive_voxels() (reference_cpu.cpp) turns them into the ranked hydration-site
//   list -- the very same function the CPU path uses, so CPU and GPU agree exactly.
//
//   kernels.cu defines the kernel + the host wrapper gist_gpu(). main.cu calls it.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, gist.h, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // Dataset, VoxelResult, gist_fixed_t (pure C++; safe in .cu)

// ---------------------------------------------------------------------------
// gist_accumulate_kernel: one thread per (water, frame) sample.
//   Launch (set in gist_gpu):
//     grid  = ceil(num_samples / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread map: t = blockIdx.x*blockDim.x + threadIdx.x -> sample t (frame, water).
//   Memory: reads waters[t*3..] and all atoms[] from global memory; atomicAdds
//     into counts[voxel] (unsigned int) and esum[voxel] (fixed-point long long).
//   Atomics: yes -- the scatter destination (a voxel) is shared by many samples.
//   The solute atoms array is read by every thread; for the tiny teaching solute
//   it stays hot in the L2/constant-ish caches, so a plain __restrict__ global
//   pointer is fine (production GIST would stage atoms in shared/constant memory).
// ---------------------------------------------------------------------------
__global__ void gist_accumulate_kernel(const float* __restrict__ waters,
                                       long long num_samples,
                                       const float* __restrict__ atoms, int natoms,
                                       GistGrid grid,
                                       unsigned int* __restrict__ counts,
                                       gist_fixed_t* __restrict__ esum);

// ---------------------------------------------------------------------------
// gist_gpu: host wrapper -- the whole GPU computation behind one call.
//   Allocates device buffers, copies the waters + atoms H2D, zeroes the voxel
//   tallies, launches the accumulation kernel (timed with CUDA events), copies the
//   raw tallies back, and runs the SHARED derive_voxels() reduction on the host so
//   the ranked list matches the CPU bit-for-bit.
//
//   d         : the loaded dataset (frames, waters, atoms, grid).
//   counts    : OUT, length grid.num_voxels(), GPU occupancy per voxel.
//   esum      : OUT, length grid.num_voxels(), GPU fixed-point energy sum per voxel.
//   kernel_ms : OUT, milliseconds spent in the accumulation kernel (not copies).
//   returns   : the ranked VoxelResult list (highest GIST dG first).
// ---------------------------------------------------------------------------
std::vector<VoxelResult> gist_gpu(const Dataset& d,
                                  std::vector<unsigned int>& counts,
                                  std::vector<gist_fixed_t>& esum,
                                  float* kernel_ms);
