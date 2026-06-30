// ===========================================================================
// src/kernels.cuh  --  GPU ensemble-integration interface (the teaching idea)
// ---------------------------------------------------------------------------
// Project 2.34 : Biophysical Simulation of Biomolecular Condensates
//                (Active Learning Loop)  --  reduced-scope teaching version
//
// THE BIG IDEA (ensemble pattern: ONE THREAD INTEGRATES ONE TRAJECTORY)
//   The active-learning loop needs the SAME coarse-grained MD run for MANY
//   candidate sequences (here: many stickiness values lambda). Each trajectory
//   is sequential IN TIME (step t+1 needs step t) but completely INDEPENDENT of
//   the other trajectories, so the natural GPU mapping is one thread per replica:
//   thread m runs the full Brownian-dynamics loop for candidate m in its own
//   registers/local memory and writes a single ReplicaResult (its D and Rg).
//   No shared memory, no atomics, no inter-thread communication -- embarrassing
//   parallelism over the ensemble. This is the flagship 9.02 / 13.02 pattern
//   (PATTERNS.md section 1, "the same ODE for many parameter sets").
//
//   The integrator itself is the shared __host__ __device__ integrate_replica()
//   in condensate.h, so the GPU result matches the CPU reference to a small,
//   documented float tolerance (the trajectories are hundreds of FMA-bearing
//   steps, so we verify to a physical tolerance, not bit-equality -- PATTERNS s4).
//
//   Included ONLY by .cu translation units (it declares a __global__). The pure-
//   C++ CPU reference + the active-learning step live in reference_cpu.h.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, condensate.h,
//                  reference_cpu.h.  Then read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // EnsembleConfig, ReplicaResult (pure C++, safe in .cu)

// ---- Device kernel -------------------------------------------------------
// ensemble_kernel: thread idx integrates ensemble member idx (one candidate
// sequence's whole CG-MD trajectory) and writes its ReplicaResult.
//   c   : the experiment config, passed BY VALUE so every thread has its own
//         copy in registers/constant-arg space (it is small and read-only)
//   out : device pointer to n_members ReplicaResult slots (one written per thread)
// The launch configuration and thread-to-data mapping are documented in
// kernels.cu where the kernel is defined.
__global__ void ensemble_kernel(EnsembleConfig c, ReplicaResult* __restrict__ out);

// ---- Host wrapper --------------------------------------------------------
// integrate_gpu: "do the whole GPU ensemble" from the host. Allocates the
// device result buffer, launches one thread per member, copies the results back,
// and reports the measured KERNEL time (CUDA events) via *kernel_ms. main.cu
// calls exactly this; all CUDA bookkeeping is hidden here.
//   c         : the ensemble config (host-side; copied into the kernel by value)
//   results   : host output, resized to n_members (output parameter)
//   kernel_ms : out-param, milliseconds spent in the kernel itself (not copies)
void integrate_gpu(const EnsembleConfig& c, std::vector<ReplicaResult>& results,
                   float* kernel_ms);
