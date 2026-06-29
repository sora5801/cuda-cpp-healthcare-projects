// ===========================================================================
// src/kernels.cuh  --  GPU SMD-ensemble interface
// ---------------------------------------------------------------------------
// Project 1.26 : Steered Molecular Dynamics (SMD)
//
// THE BIG IDEA (pattern: ENSEMBLE of independent stochastic trajectories)
//   Jarzynski's free-energy estimate needs MANY independent constant-velocity
//   SMD pulls of the same system, each with a different random thermal history.
//   A single pull is sequential in time (step n+1 depends on step n) but totally
//   independent of the other pulls -- so we give each trajectory its OWN GPU
//   thread. The thread runs the full Langevin time loop in registers and writes
//   one number: that trajectory's external work W_i. This is the same
//   thread-per-trajectory mapping the SEIR ensemble (9.02) and PBPK (13.02) use,
//   combined with a per-thread reproducible RNG (5.01) for the thermal noise.
//
//   The trajectory integrator is shared with the CPU (smd_core.h), so the GPU's
//   W_i match the reference EXACTLY. The Jarzynski reduction over the work array
//   is done once on the host (main.cu), identical for both sides. kernels.cu
//   defines the kernel and its host launcher.
//
// READ THIS AFTER: smd_core.h, util/cuda_check.cuh, util/timer.cuh.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // SmdParams (pure C++, safe to include in a .cu)

// Device kernel: thread `i` runs SMD trajectory i and writes its work W_i.
//   grid  : ceil(n_traj / THREADS_PER_BLOCK) blocks
//   block : THREADS_PER_BLOCK threads
//   thread (blockIdx.x, threadIdx.x) -> trajectory index i -> work[i]
__global__ void smd_kernel(SmdParams p, double* __restrict__ work);

// Host wrapper: allocate device output, launch one thread per trajectory, copy
// the work array back, and report the GPU kernel time (CUDA-event measured).
//   p         : the SMD configuration (passed by value into the kernel)
//   work      : resized to n_traj and filled with the GPU's per-trajectory work
//   kernel_ms : out-param, milliseconds the kernel took (teaching artifact)
void run_gpu(const SmdParams& p, std::vector<double>& work, float* kernel_ms);
