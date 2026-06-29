// ===========================================================================
// src/kernels.cuh  --  GPU molecular-dynamics interface (declarations + idea)
// ---------------------------------------------------------------------------
// Project 1.1 : Molecular Dynamics Engine  (reduced-scope teaching version)
//
// THE BIG IDEA (the all-pairs N-body force pattern)
//   The cost of MD is dominated by the FORCE EVALUATION: computing, every step,
//   the force on each atom from every other atom. For N atoms that is N*(N-1)
//   pairwise interactions -- an O(N^2) sum that is "embarrassingly parallel" in
//   the index i: the force on atom i is INDEPENDENT of the force on atom k. So we
//   give EACH ATOM ITS OWN GPU THREAD: thread i loops over all j, accumulating its
//   own total force in registers, then we kick/drift it. This is the classic
//   GPU N-body mapping (the same shape as the famous CUDA n-body sample).
//
//   To avoid hammering global memory -- a naive kernel would read all N positions
//   from DRAM for every one of the N threads, i.e. N^2 global loads -- we TILE the
//   j-loop through SHARED MEMORY: each block cooperatively loads a tile of
//   positions once, all its threads reuse that tile, then we advance to the next
//   tile. This cuts global traffic to ~N^2/TILE and is the single most important
//   optimization here (see THEORY §GPU mapping; mirrors PATTERNS.md tiling idea).
//
//   The per-pair physics + the Verlet kicks come from md.h, the SAME header the
//   CPU reference uses, so the GPU trajectory matches the CPU one to round-off.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, md.h, reference_cpu.h.
//   Then read kernels.cu for the implementation.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // MdSystem, MdResult, SimParams (all pure C++)

// ---------------------------------------------------------------------------
// integrate_gpu: run the whole velocity-Verlet simulation on the GPU and return
//   the same MdResult the CPU reference produces, so main.cu can compare them.
//
//   sys       : the initial system (parameters + initial positions/velocities).
//               Not modified (its pos/vel are copied to the device).
//   kernel_ms : out-param; total milliseconds spent in the integration kernels
//               (CUDA-event measured), reported as a teaching artifact on stderr.
//
//   All CUDA bookkeeping (allocate, copy, launch, free) is hidden inside; main.cu
//   stays clean and just calls this. Implementation + the device kernels live in
//   kernels.cu.
// ---------------------------------------------------------------------------
MdResult integrate_gpu(const MdSystem& sys, float* kernel_ms);
