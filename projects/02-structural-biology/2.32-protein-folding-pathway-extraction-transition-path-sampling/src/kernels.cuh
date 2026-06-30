// ===========================================================================
// src/kernels.cuh  --  GPU Transition-Path-Sampling interface
// ---------------------------------------------------------------------------
// Project 2.32 : Protein Folding Pathway Extraction (Transition Path Sampling)
//
// THE BIG IDEA
//   TPS shooting moves are INDEPENDENT: each shot has its own RNG stream and its
//   own Brownian-dynamics trajectory, with no data shared between shots. That is
//   the textbook "embarrassingly parallel independent shooter array" the catalog
//   names for this project -- so we assign ONE GPU THREAD PER SHOOTER (with a
//   grid-stride loop so a fixed grid covers any n_shooters).
//
//   Two TPS-specific lessons, mirrored from the Monte-Carlo dose flagship (5.01):
//     * PER-THREAD RNG: each thread seeds its own reproducible stream from its
//       shooter index (rng_seed in tps_physics.h). The shared header means the
//       CPU reproduces the IDENTICAL shooting moves for exact verification.
//     * INTEGER ATOMIC SCORING: many threads update the SAME scalar counters and
//       the SAME committor-histogram bins, so the tally uses atomicAdd. Because
//       every increment is an INTEGER (a shot either is/isn't a transition, did/
//       didn't commit to B), the atomic adds are order-independent -> the GPU
//       result is deterministic AND equals the CPU tally exactly. Floating-point
//       sums would NOT have this property (PATTERNS.md §3).
//
//   kernels.cu defines the kernel + host wrapper. main.cu calls tps_gpu().
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, tps_physics.h,
//                  reference_cpu.h.  Then read kernels.cu (the GPU twin of
//                  reference_cpu.cpp).
// ===========================================================================
#pragma once

#include "reference_cpu.h"   // TpsProblem, TpsTally, SimParams (pure C++; safe in .cu)

// ---- Device kernel --------------------------------------------------------
// tps_kernel: each thread runs one or more shooting moves (grid-stride) and
// scores integer results into device counters via atomicAdd.
//   sp                 : the simulation parameters (passed by value -> registers)
//   d_n_transitions    : device scalar; += 1 per accepted transition path
//   d_n_fwd_to_B       : device scalar; += 1 per forward leg committing to B
//   d_shots_per_bin    : device [n_bins]; += 1 in the shot's committor bin
//   d_committed_per_bin: device [n_bins]; += 1 if that shot's fwd leg reached B
// We accumulate into `unsigned long long` because that is the type CUDA's
// 64-bit atomicAdd operates on; the host wrapper converts back to signed for the
// (exact) comparison with the CPU's long long tally.
__global__ void tps_kernel(SimParams sp,
                           unsigned long long* __restrict__ d_n_transitions,
                           unsigned long long* __restrict__ d_n_fwd_to_B,
                           unsigned long long* __restrict__ d_shots_per_bin,
                           unsigned long long* __restrict__ d_committed_per_bin);

// ---- Host wrapper ---------------------------------------------------------
// tps_gpu: run the whole GPU computation. Allocates and zeroes the device
// counters, launches tps_kernel over all shooters, copies the integer tallies
// back, and reports the measured KERNEL time (CUDA events) via *kernel_ms.
//   prob      : the loaded problem (parameters).
//   tally     : output; resized to n_bins and filled with the integer counts.
//   kernel_ms : out-param, milliseconds spent in the kernel itself (not copies).
void tps_gpu(const TpsProblem& prob, TpsTally& tally, float* kernel_ms);
