// ===========================================================================
// src/kernels.cuh  --  GPU tumor-growth + treatment interface
// ---------------------------------------------------------------------------
// Project 6.8 : Tumor Growth & Treatment-Response Modeling
//
// THE PATTERN (a STENCIL + ping-pong, cf. lattice-Boltzmann 6.04, RD 14.02)
//   Fisher-KPP growth updates each grid cell from its 4 nearest neighbours only,
//   so we map ONE THREAD PER CELL on a 2-D grid. The host runs the time loop,
//   launching the growth kernel once per step and PING-PONGING two density
//   buffers: read the frozen previous field, write the next field, swap. On a
//   scheduled radiotherapy fraction the host first launches a per-cell TREATMENT
//   kernel (a pure multiply by the LQ surviving fraction) in place.
//
//   The per-cell updates are the shared tumor_grow_update / tumor_treat_update
//   in tumor.h, so the GPU reproduces the CPU result (modulo tiny FP-order drift).
//
// READ THIS AFTER: tumor.h, util/cuda_check.cuh, util/timer.cuh, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // TumorParams (pure C++, safe to include in a .cu)

// Device kernel: thread (x,y) computes one cell's next density from the input
// buffer `u` (frozen) into the output buffer `un` -- one Fisher-KPP step.
__global__ void tumor_grow_kernel(TumorParams P, const double* __restrict__ u,
                                   double* __restrict__ un);

// Device kernel: thread i multiplies cell i's density by `survival` in place
// (one radiotherapy fraction; the LQ surviving fraction is precomputed on host).
__global__ void tumor_treat_kernel(int n, double survival, double* __restrict__ u);

// Host wrapper: `u` comes in holding the INITIAL field and is updated in place to
// the FINAL field after `steps` timesteps (growth + scheduled LQ fractions).
// Writes the total GPU loop time (ms) through `kernel_ms`.
void simulate_gpu(const TumorParams& P, std::vector<double>& u, float* kernel_ms);
