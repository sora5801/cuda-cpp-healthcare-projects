// ===========================================================================
// src/kernels.cuh  --  GPU batched-ODE interface (declarations + the big idea)
// ---------------------------------------------------------------------------
// Project 6.10 : Systems-Biology ODE/SDE Network Solver
//
// THE BIG IDEA (batched / ensemble ODE integration -- PATTERNS.md §1 row
// "the same ODE for many parameter sets"; exemplified by flagship 9.02 SEIR and
// 13.02 PBPK)
//   Systems-biology models (gene circuits, signalling cascades, metabolism) are
//   small ODE systems, but the SCIENCE lives in solving them thousands of times:
//   parameter sweeps, uncertainty quantification, per-cell heterogeneity. Each
//   solve is sequential in TIME but completely independent of the others, so we
//   give each ensemble member its OWN GPU THREAD. The thread runs the full RK4
//   time loop in registers (state is only 6 doubles) and writes one summary
//   MemberResult. There is no inter-thread communication -- embarrassingly
//   parallel over members. This is precisely the pattern SUNDIALS/CVODE-GPU and
//   libRoadRunner target with their batch-ODE backends (README "Prior art").
//
//   The RK4 integrator + repressilator RHS are shared with the CPU via grn.h,
//   so the GPU results match the reference to round-off. kernels.cu defines the
//   kernel and the host wrapper.
//
//   (Catalog note: the catalog mentions "one thread-BLOCK per ODE system, shared
//   memory for the Jacobian, cuSPARSE for large sparse Jacobians." That layout
//   pays off only for LARGE stiff systems needing an implicit solve. For the
//   small explicit repressilator, one THREAD per system is faster and simpler;
//   THEORY.md §GPU-mapping explains the trade-off and when to switch.)
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, grn.h, reference_cpu.h.
// Then read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // EnsembleConfig, MemberResult (pure C++, safe in .cu)

// ---- Device kernel -------------------------------------------------------
// ensemble_kernel: thread `idx` integrates ensemble member `idx`.
//   grid  : ceil(M / THREADS_PER_BLOCK) blocks, M = ensemble_size(c)
//   block : THREADS_PER_BLOCK threads (a good occupancy default on sm_75..sm_89)
//   thread (blockIdx.x, threadIdx.x) -> flat member index idx
//   It reads its (alpha,n) from the sweep via member_params(), runs the full RK4
//   loop from grn.h in registers/local memory, and writes one MemberResult.
//   `c` is passed BY VALUE so the whole config lands in the kernel's constant
//   parameter bank (broadcast to every thread); `out` is [M] in global memory.
__global__ void ensemble_kernel(EnsembleConfig c, MemberResult* __restrict__ out);

// ---- Host wrapper --------------------------------------------------------
// integrate_gpu: allocate the [M] result buffer on the device, launch one
//   thread per member, copy results back, and report the measured KERNEL time
//   (CUDA events) via *kernel_ms. main.cu calls exactly this; all CUDA
//   bookkeeping is hidden here.
//     c         : the ensemble config (read-only)
//     results   : host output, resized to M (output parameter)
//     kernel_ms : out-param, milliseconds spent in the kernel itself (not copies)
void integrate_gpu(const EnsembleConfig& c, std::vector<MemberResult>& results,
                   float* kernel_ms);
