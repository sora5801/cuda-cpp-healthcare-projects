// ===========================================================================
// src/kernels.cuh  --  GPU knockout-screen interface
// ---------------------------------------------------------------------------
// Project 6.12 : Metabolic Flux / Constraint-Based Modeling
//
// THE BIG IDEA (pattern: an ENSEMBLE OF INDEPENDENT LPs, one per thread)
//   A gene-essentiality screen asks: "for each reaction, if we delete it, can the
//   cell still grow?" Answering it means solving one FBA linear program per
//   deletion. Those LPs are completely INDEPENDENT -- deleting reaction 3 has
//   nothing to do with deleting reaction 7 -- so we give each knockout its own
//   GPU thread. Thread k solves the LP for "reaction k removed"; one extra thread
//   solves the wild type. This is the same embarrassingly-parallel ensemble shape
//   as flagship 9.02 (one ODE trajectory per thread), but the per-item work here
//   is a whole simplex solve instead of an RK4 integration.
//
//   The simplex solver is shared with the CPU reference (fba.h, __host__
//   __device__), so the GPU screen and the CPU screen return bit-identical
//   objectives -- verification is exact. kernels.cu defines the kernel + wrapper.
//
//   MEMORY NOTE: each thread's simplex tableau lives in per-thread LOCAL memory
//   (fixed-size arrays in fba.h). That keeps threads fully independent -- no
//   shared memory, no atomics, no synchronisation -- at the cost of a sizeable
//   per-thread footprint, which caps occupancy. For a teaching screen of a few
//   dozen reactions that is completely fine; THEORY.md section "GPU mapping"
//   discusses the shared-memory tableau a production batch solver would use.
//
// READ THIS AFTER: fba.h (the solver), reference_cpu.h (the model + CPU screen),
// util/cuda_check.cuh, util/timer.cuh. Then read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // FbaModel, FbaResult (pure C++, safe inside a .cu)

// ---- Device kernel -------------------------------------------------------
// screen_kernel: thread `k` solves the FBA LP with reaction k deleted (or the
//   wild type for k == nrxn) and writes one FbaResult.
//   grid  : ceil((nrxn+1) / block) blocks
//   block : THREADS_PER_BLOCK threads (set in the wrapper)
//   thread (blockIdx.x, threadIdx.x) -> knockout index k = bx*blockDim.x + tx.
//   model is passed BY VALUE (it is plain-old-data, no pointers), so every thread
//   gets its own copy to clamp -- no device allocation of the model needed.
__global__ void screen_kernel(FbaModel model, FbaResult* __restrict__ out, int njobs);

// ---- Host wrapper --------------------------------------------------------
// screen_gpu: run the whole knockout screen on the GPU.
//   model     : the FBA model (by const ref; copied into the kernel launch).
//   results   : filled with (nrxn + 1) FbaResults in the SAME layout as
//               screen_cpu() -- index k = knockout of reaction k, last = wild type.
//   kernel_ms : out-param, milliseconds spent in the kernel (CUDA-event timed).
// All CUDA bookkeeping (allocate result buffer, launch, copy back, free) is
// hidden here; main.cu just calls this and compares with the CPU array.
void screen_gpu(const FbaModel& model, std::vector<FbaResult>& results, float* kernel_ms);
