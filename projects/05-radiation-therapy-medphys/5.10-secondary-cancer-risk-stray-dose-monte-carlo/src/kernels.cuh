// ===========================================================================
// src/kernels.cuh  --  GPU Monte Carlo stray-dose interface
// ---------------------------------------------------------------------------
// Project 5.10 : Secondary Cancer Risk & Stray-Dose Monte Carlo
//
// THE BIG IDEA (pattern: per-thread RNG + atomic scoring; exemplar 5.01)
//   Particle histories are INDEPENDENT, so each GPU thread tracks one primary
//   photon (grid-stride over millions of them). Three MC-specific lessons live in
//   the kernel, all in kernels.cu:
//     * PER-THREAD RNG: each thread seeds its own reproducible stream from its
//       history index (rng_seed in stray_physics.h). The shared header means the
//       CPU reproduces the identical histories, so verification is EXACT.
//     * VARIANCE REDUCTION: survival biasing + Russian roulette + forced
//       detection (all inside simulate_history) turn a rare stray-dose signal into
//       a low-variance per-history tally -- the reason stray-dose MC is tractable.
//     * ATOMIC + FIXED-POINT SCORING: many threads deposit into the SAME organ
//       bins, so the tally uses atomicAdd. Deposits are FIXED-POINT INTEGERS, so
//       the atomic adds are order-independent -> the GPU result is deterministic
//       and equals the CPU tally exactly (a float atomic sum would not).
//
//   kernels.cu defines the kernel; main.cu calls dose_gpu().
//
// READ THIS AFTER: stray_physics.h, reference_cpu.h.
// READ NEXT: kernels.cu, then main.cu.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // StrayProblem, SimParams (pure C++, safe in .cu)

// Device kernel: each thread simulates one or more primary histories (grid-stride)
// and scores fixed-point stray dose into the shared `dose` tally via atomicAdd.
//   sp   : phantom + beam + variance-reduction parameters (by value -> registers)
//   dose : device pointer to n_organs 64-bit accumulators (fixed-point)
__global__ void stray_kernel(SimParams sp,
                             unsigned long long* __restrict__ dose);

// Host wrapper: allocate + zero the device tally, launch the histories, copy the
// dose back.
//   prob      : the loaded problem (phantom + organs + history count)
//   dose      : resized to n_organs; filled with per-organ fixed-point dose
//   kernel_ms : out-param, GPU kernel time in milliseconds (teaching artifact)
void dose_gpu(const StrayProblem& prob, std::vector<unsigned long long>& dose,
              float* kernel_ms);
