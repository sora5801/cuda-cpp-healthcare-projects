// ===========================================================================
// src/kernels.cuh  --  GPU reaction-diffusion interface
// ---------------------------------------------------------------------------
// Project 14.02 : Spatial / Whole-Cell Reaction-Diffusion (teaching stencil)
//
// THE PATTERN (a STENCIL, cf. lattice-Boltzmann 6.04)
//   Each grid cell updates from its 4 neighbours only, so we map one thread per
//   cell on a 2-D grid. The host runs the time loop, launching the kernel once
//   per step and PING-PONGING two (U,V) buffer pairs: read the frozen previous
//   state, write the next state, swap. The per-cell update is the shared
//   rd_update() in rd.h, so the GPU reproduces the CPU result.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, rd.h, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // RdParams (pure C++, safe in .cu)

// Device kernel: thread (x,y) computes one cell's next (U,V) from the input buffers.
__global__ void rd_step_kernel(RdParams P, const double* __restrict__ U, const double* __restrict__ V,
                               double* __restrict__ Un, double* __restrict__ Vn);

// Host wrapper: U and V come in holding the INITIAL fields and are updated in
// place to the FINAL fields after `steps` timesteps. Returns total GPU loop time.
void simulate_gpu(const RdParams& P, std::vector<double>& U, std::vector<double>& V, float* kernel_ms);
