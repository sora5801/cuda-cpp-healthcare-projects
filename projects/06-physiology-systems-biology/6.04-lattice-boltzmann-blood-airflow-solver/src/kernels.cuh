// ===========================================================================
// src/kernels.cuh  --  GPU LBM interface
// ---------------------------------------------------------------------------
// Project 6.04 : Lattice-Boltzmann Blood/Airflow Solver
//
// THE BIG IDEA (fifth flagship pattern: a STENCIL)
//   Every lattice node updates from its nearest neighbours only, so we give each
//   node its own thread on a 2-D grid. The host runs the time loop, launching
//   the kernel once per step and PING-PONGING two device buffers (read f_old,
//   write f_new, swap). The per-node math is the shared lbm_collide_stream() in
//   lbm_d2q9.h, so the GPU reproduces the CPU result.
//
//   This is the canonical GPU CFD pattern; production codes add shared-memory
//   tiling of the streaming step and 3-D stencils (D3Q19/D3Q27).
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, lbm_d2q9.h, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // LbmParams (pure C++, safe in .cu)

// Device kernel: thread (x,y) performs one collide+stream update for its node.
__global__ void lbm_step_kernel(int nx, int ny, double tau, double gx,
                                const double* __restrict__ f_old,
                                double* __restrict__ f_new);

// Host wrapper: run the full time loop on the GPU and return the final
// population field (size 9*nx*ny) plus the total GPU time of the loop.
void lbm_gpu(const LbmParams& p, std::vector<double>& f_final, float* kernel_ms);
