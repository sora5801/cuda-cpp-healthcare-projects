// ===========================================================================
// src/kernels.cuh  --  GPU ensemble-integration interface (the teaching idea)
// ---------------------------------------------------------------------------
// Project 1.35 : QMMM/ML Potential Hybrid MD   (reduced-scope teaching version)
//
// THE BIG IDEA (GPU pattern: ENSEMBLE -- thread per trajectory; PATTERNS.md §1)
//   Active-learning of reactive ML potentials needs MANY short MD trajectories
//   from slightly different starting points, to map out where the model is
//   uncertain. Each trajectory is sequential in TIME but INDEPENDENT of the
//   others, so we give each one its own GPU thread: the thread runs the whole
//   velocity-Verlet loop in registers and writes a single TrajResult summary.
//   No inter-thread communication, no shared memory, no atomics -- pure
//   embarrassing parallelism over ensemble members. (Same shape as flagships
//   9.02 SEIR and 13.02 PBPK.)
//
//   The per-step physics -- the hybrid NNP(ML)+LJ(MM) force/energy and the
//   integrator -- is shared with the CPU reference in nnpmm.h, so the GPU result
//   matches the CPU to floating-point round-off. kernels.cu defines the kernel.
//
//   This header is included only by .cu translation units (it declares a
//   __global__ kernel), so the plain C++ compiler never sees it -- that is why
//   the CPU reference declarations live in the pure-C++ reference_cpu.h.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, nnpmm.h, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // EnsembleConfig, TrajResult (pure C++, safe in .cu)

// ---- Device kernel -------------------------------------------------------
// ensemble_kernel: thread `idx` integrates ensemble member idx and writes its
//   TrajResult.
//   Launch config (set in integrate_gpu):
//     grid  = ceil(M / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: idx = blockIdx.x * blockDim.x + threadIdx.x  -> member.
//   Memory: each thread keeps its small per-atom state (x,v,a of N_ATOMS) in
//   registers/local memory; the only global write is out[idx]. No atomics.
__global__ void ensemble_kernel(EnsembleConfig c, TrajResult* __restrict__ out);

// ---- Host wrapper --------------------------------------------------------
// integrate_gpu: launch one thread per member, copy results back, time the
//   kernel with CUDA events.
//   c         : the ensemble config (passed by value -> copied to every thread)
//   results   : host output, resized to M (output parameter)
//   kernel_ms : out-param, milliseconds spent in the kernel (CUDA-event measured)
void integrate_gpu(const EnsembleConfig& c, std::vector<TrajResult>& results,
                   float* kernel_ms);
