// ===========================================================================
// src/kernels.cuh  --  GPU reaction-diffusion interface (declarations + idea)
// ---------------------------------------------------------------------------
// Project 6.24 : Reaction-Diffusion Morphogenesis (Turing Patterns)
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls simulate_gpu(); kernels.cu
//   implements both the host time-loop wrapper and the device stencil kernel.
//   Included only by .cu translation units (it declares a __global__, so the
//   plain host C++ compiler must never see it -- that is why the CPU reference
//   scaffolding lives in the pure-C++ reference_cpu.h instead).
//
// THE PATTERN: a 2-D STENCIL with PING-PONG buffers (PATTERNS.md §1)
//   Each grid cell's next (a,h) depends only on itself and its 4 neighbours.
//   We map ONE THREAD PER CELL on a 2-D thread grid (16x16 tiles = 256 threads
//   per block, a solid occupancy default on sm_75..sm_89). The host runs the
//   time loop, launching the kernel once per timestep and swapping ("ping-
//   ponging") two device buffer pairs: read the frozen old state, write the new
//   state, swap. Because each thread writes only its own cell, there are no
//   write races and no atomics. The per-cell math is the SHARED tu_update() in
//   turing.h, so the GPU reproduces the CPU field.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, turing.h. Then kernels.cu.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // TuringParams (pure C++, safe to include in a .cu)

// ---- Device kernel -------------------------------------------------------
// One thread computes one cell's next (a,h) from the input buffers.
//   grid   : ceil(nx/TILE) x ceil(ny/TILE) blocks covering the 2-D domain
//   block  : TILE x TILE threads
//   thread (blockIdx, threadIdx) -> cell (x,y) with
//            x = blockIdx.x*blockDim.x + threadIdx.x,
//            y = blockIdx.y*blockDim.y + threadIdx.y
//   P      : model + grid parameters, passed BY VALUE (small, trivially copyable
//            -> lives in registers/constant param space, no device pointer dance)
//   a, h   : CURRENT activator / inhibitor fields (device, nx*ny) -- read only
//   an, hn : NEXT fields (device, nx*ny) -- written at this thread's cell
// __restrict__ promises the buffers do not alias, so nvcc can cache reads.
__global__ void rd_step_kernel(TuringParams P,
                               const double* __restrict__ a,
                               const double* __restrict__ h,
                               double* __restrict__ an,
                               double* __restrict__ hn);

// ---- Host wrapper --------------------------------------------------------
// simulate_gpu: run the whole GPU simulation and return the KERNEL-loop time.
//   Allocates 4 device buffers (two ping-pong pairs), copies the initial fields
//   H2D, launches rd_step_kernel once per step swapping buffers between launches,
//   copies the final fields D2H, and reports the measured loop time via
//   *kernel_ms (CUDA events). main.cu calls exactly this; all CUDA bookkeeping
//   is hidden here.
//
//   P         : model + grid parameters (defines nx, ny, steps, ...)
//   a, h      : IN = initial fields, OUT = final fields (updated in place)
//   kernel_ms : out-param, milliseconds spent in the stepping loop (not H2D/D2H)
void simulate_gpu(const TuringParams& P, std::vector<double>& a,
                  std::vector<double>& h, float* kernel_ms);
