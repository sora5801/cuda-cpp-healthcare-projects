// ===========================================================================
// src/kernels.cuh  --  GPU bone-remodeling interface (declarations + big idea)
// ---------------------------------------------------------------------------
// Project 6.22 : Bone Remodeling Simulation   (REDUCED-SCOPE teaching version)
//
// THE BIG IDEA (the GPU pattern here: a 2-D STENCIL + ping-pong buffers)
//   The whole simulation is a grid of voxels updated from their nearest
//   neighbours only -- the textbook GPU pattern (PATTERNS.md section 1: "grid
//   PDE / nearest-neighbour update -> stencil + ping-pong", the same shape as
//   flagships 6.04 lattice-Boltzmann and 14.02 reaction-diffusion). We give each
//   voxel its own thread on a 2-D grid of blocks. The host runs the two nested
//   loops (remodeling steps, and Jacobi relaxation sweeps within each step),
//   launching a kernel per sweep/step and PING-PONGING device buffers so a
//   thread never reads a value another thread is mid-writing (no races, no
//   atomics). The per-voxel math is the shared __host__ __device__ code in
//   bone_remodel.h, so the GPU reproduces the CPU reference exactly.
//
//   Two kernels, mirroring the two physics functions:
//     * relax_kernel   : one Jacobi sweep of the mechanical-stimulus field S.
//     * remodel_kernel : one mechanostat density update from a settled S.
//
//   Production voxel-FEM bone codes replace the stimulus relaxation with a real
//   K u = f solve (cuSPARSE assembly + cuSOLVER/PCG); THEORY.md "real world"
//   spells out that difference. This file only declares the two custom kernels
//   and the host wrapper; kernels.cu implements them.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, bone_remodel.h,
//                  reference_cpu.h. Then read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // BoneParams (pure C++, safe to include in a .cu)

// ---- Device kernels --------------------------------------------------------
// relax_kernel: thread (x,y) writes S_new(x,y) = one Jacobi sweep of the
//   stimulus field (bone_relax_point). Reads the read-only S_old + rho buffers.
//     grid  : ceil(nx/TILE) x ceil(ny/TILE) blocks
//     block : TILE x TILE threads
//     map   : x = blockIdx.x*blockDim.x+threadIdx.x, y = blockIdx.y*...+...
__global__ void relax_kernel(int nx, int ny, double load, int load_x0, int load_x1,
                             const double* __restrict__ S_old,
                             const double* __restrict__ rho,
                             double* __restrict__ S_new);

// remodel_kernel: thread (x,y) writes rho_new(x,y) = one mechanostat update
//   (bone_apply_stimulus) from the settled stimulus field S. Same 2-D mapping.
__global__ void remodel_kernel(int nx, int ny, double setpoint, double lazy,
                               double rate, double rho_min,
                               const double* __restrict__ S,
                               const double* __restrict__ rho_old,
                               double* __restrict__ rho_new);

// ---- Host wrapper ----------------------------------------------------------
// bone_gpu: run the full remodeling simulation on the GPU and return the final
//   density field `rho_final` (size nx*ny), the last settled stimulus field
//   `S_final` (size nx*ny, for the report's state histogram), and the total GPU
//   time of all kernel launches via *kernel_ms (CUDA-event measured). main.cu
//   calls exactly this; all device bookkeeping is hidden inside.
void bone_gpu(const BoneParams& p,
              std::vector<double>& rho_final,
              std::vector<double>& S_final,
              float* kernel_ms);
