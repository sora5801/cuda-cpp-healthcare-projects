// ===========================================================================
// src/kernels.cuh  --  GPU ensemble-SSA interface (declarations + the idea)
// ---------------------------------------------------------------------------
// Project 6.11 : Stochastic (Gillespie) Biochemical Simulation
//
// THE BIG IDEA (pattern: ENSEMBLE OF INDEPENDENT STOCHASTIC HISTORIES)
//   The Gillespie SSA produces ONE random trajectory of a reaction network.
//   Statistics (means, variances, distributions) need MANY trajectories. Every
//   trajectory is completely independent -- no shared state, no communication,
//   no atomics -- so the mapping is the cleanest possible: ONE GPU THREAD RUNS
//   ONE ENTIRE TRAJECTORY. Thread `idx` seeds RNG stream idx, runs the full
//   event-by-event SSA loop in registers/local memory, and writes one
//   TrajectoryResult. This is "embarrassingly parallel" Monte Carlo -- exactly
//   the workload GPUs excel at, and it needs zero synchronisation.
//
//   The SSA core (RNG + reaction selection + stoichiometry) is shared with the
//   CPU via ssa.h, so the GPU trajectories are BIT-IDENTICAL to the reference.
//   kernels.cu defines the kernel and the host wrapper.
//
// This header contains a __global__ declaration, so ONLY .cu files may include
// it (nvcc). It reuses EnsembleConfig / TrajectoryResult from reference_cpu.h,
// which is pure C++ and therefore safe inside a .cu translation unit.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, ssa.h, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // EnsembleConfig, TrajectoryResult, ReactionNetwork

// ---- Device kernel -------------------------------------------------------
// ssa_kernel: thread `idx` simulates trajectory idx of the given network.
//   net : the reaction network, passed BY VALUE (POD struct rides in the
//         kernel's parameter space -- no device allocation for the network).
//   n_traj : total trajectories (guards the ragged last block).
//   out : device array of n_traj TrajectoryResult (one slot per thread).
// The thread-to-data mapping is idx = blockIdx.x*blockDim.x + threadIdx.x.
__global__ void ssa_kernel(ReactionNetwork net, int n_traj,
                           TrajectoryResult* __restrict__ out);

// ---- Host wrapper --------------------------------------------------------
// simulate_gpu: run the whole ensemble on the GPU.
//   Builds the network from `c`, allocates the device result buffer, launches
//   one thread per trajectory, copies the results back, and reports the measured
//   KERNEL time (CUDA events) via *kernel_ms. main.cu calls exactly this.
//     c        : ensemble configuration (network + trajectory count + seed)
//     results  : host output, resized to c.n_traj (output parameter)
//     kernel_ms: out-param, milliseconds spent in the kernel (not the copies)
void simulate_gpu(const EnsembleConfig& c, std::vector<TrajectoryResult>& results,
                  float* kernel_ms);
