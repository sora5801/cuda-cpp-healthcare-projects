// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 2.29 : Ion Channel Gating & Permeation Simulation
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls permeation_gpu(); kernels.cu
//   implements both the host wrapper and the device kernel. Included only by .cu
//   translation units (it pulls in CUDA-only declarations), so the plain C++
//   compiler never sees it -- which is why the CPU reference's prototype lives in
//   the separate pure-C++ reference_cpu.h.
//
// THE BIG IDEA  (PATTERNS.md §1: stochastic / Monte-Carlo histories)
//   The ions are INDEPENDENT Brownian walkers, so we assign ONE GPU THREAD PER
//   ION. With n_ions trajectories and a block of B threads we use a grid-stride
//   loop so a fixed grid covers any number of ions: thread
//   i0 = blockIdx.x*blockDim.x + threadIdx.x handles ions i0, i0+stride, ...
//
//   Two Monte-Carlo lessons made concrete here:
//     * PER-THREAD RNG: each thread re-seeds its private splitmix64 stream from
//       the ion index (rng_seed in channel_physics.h). Because the CPU reference
//       seeds the SAME way from the SAME index, it reproduces the identical
//       trajectory -- enabling EXACT verification.
//     * ATOMIC SCORING: many threads add into the SAME occupancy bins and the
//       SAME two crossing counters, so we use atomicAdd. The tallied quantities
//       are INTEGERS, so the atomic adds commute -> the GPU result is
//       deterministic and equals the CPU tally bit-for-bit (a float current sum
//       would not have this property; see THEORY.md "Numerical considerations").
//
//   kernels.cu defines the kernel. main.cu calls permeation_gpu().
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, channel_physics.h,
//                  reference_cpu.h. Then read kernels.cu.
// ===========================================================================
#pragma once

#include "reference_cpu.h"   // PermeationProblem, PermeationResult, ChannelParams

// ---- Device kernel -------------------------------------------------------
// permeation_kernel: each thread integrates one or more ion trajectories and
// scores integer occupancy + crossing counts via atomicAdd.
//   cp        : channel/protocol parameters (passed by value -> per-thread regs)
//   n_ions    : number of independent trajectories (grid-stride loop bound)
//   seed      : base RNG seed (ion i uses the reproducible stream (seed, i))
//   occupancy : device tally [n_bins], one bucket per z-bin (atomicAdd target)
//   crossings : device tally [2] = {forward, reverse} permeation counts
__global__ void permeation_kernel(ChannelParams cp,
                                  unsigned long long n_ions,
                                  unsigned long long seed,
                                  unsigned long long* __restrict__ occupancy,
                                  unsigned long long* __restrict__ crossings);

// ---- Host wrapper --------------------------------------------------------
// permeation_gpu: the host-callable "do the whole GPU computation" function.
//   Allocates + zeroes the device tallies, launches permeation_kernel, copies
//   the integer results back, and reports the measured KERNEL time (CUDA events)
//   via *kernel_ms. main.cu calls exactly this; all CUDA bookkeeping is hidden.
//
//   prob      : the simulation job (channel params + n_ions + seed)
//   out       : filled with occupancy histogram + forward/reverse counts
//   kernel_ms : out-param, milliseconds spent in the kernel itself (not copies)
void permeation_gpu(const PermeationProblem& prob, PermeationResult& out,
                    float* kernel_ms);
