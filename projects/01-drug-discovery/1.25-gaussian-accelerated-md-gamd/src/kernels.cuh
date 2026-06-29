// ===========================================================================
// src/kernels.cuh  --  GPU ensemble interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 1.25 : Gaussian-Accelerated MD (GaMD)   (reduced-scope teaching version)
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls run_ensemble_gpu(); kernels.cu
//   implements both the host wrapper and the device kernel. Included only by .cu
//   translation units (it pulls in gamd.h, whose run_walker() compiles for the
//   device under nvcc).
//
// THE BIG IDEA (PATTERNS.md §1: "the same ODE for many parameter sets ->
//   ENSEMBLE, thread per trajectory"; exemplars 9.02 SEIR, 13.02 PBPK, plus
//   per-thread RNG like 5.01 Monte-Carlo dose)
//   We run n_walkers INDEPENDENT GaMD-boosted Langevin walkers. Each walker is a
//   sequential time loop but independent of the others, so we assign ONE GPU
//   THREAD PER WALKER: thread idx = blockIdx.x*blockDim.x + threadIdx.x owns
//   walker idx and runs the full run_walker() loop (gamd.h) in registers. The
//   only cross-thread interaction is the shared PMF histogram, into which many
//   walkers deposit samples -- done with DETERMINISTIC fixed-point integer
//   atomicAdd (PATTERNS.md §3 rule 2) so the GPU tally matches the serial CPU
//   tally EXACTLY, regardless of thread order.
//
//   Because the per-walker physics, RNG, and tally all live in the shared
//   gamd.h GAMD_HD functions, the device and host run byte-identical math and
//   verification is exact (tolerance 0).
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, gamd.h. Then kernels.cu.
// ===========================================================================
#pragma once

#include <cstdint>          // int64_t accumulators
#include <vector>

#include "reference_cpu.h"  // GamdConfig (pure C++, safe in .cu)

// ---- Device kernel --------------------------------------------------------
// ensemble_kernel: thread `idx` runs walker idx and atomically deposits its
//   per-step (count, dV, dV^2) contributions into the global fixed-point
//   accumulator array d_acc (length acc_total(c) = 3*n_bins int64). It also
//   writes the walker's final position to d_final_x[idx] (a deterministic
//   cross-check). No shared memory: the histogram is small but written from every
//   thread, so a single global fixed-point array with atomics is the simplest
//   correct-and-deterministic choice for a teaching kernel.
//     c          : the run config, passed BY VALUE (POD -> lives in each thread)
//     d_acc      : [3*n_bins] int64 device tally, pre-zeroed by the host wrapper
//     d_final_x  : [n_walkers] device array of final positions
__global__ void ensemble_kernel(GamdConfig c,
                                long long* __restrict__ d_acc,
                                double* __restrict__ d_final_x);

// ---- Host wrapper ---------------------------------------------------------
// run_ensemble_gpu: do the whole GPU computation. Allocates + zeros the device
//   tally, launches one thread per walker, copies the tally and final positions
//   back, and reports the measured KERNEL time (CUDA events) via *kernel_ms.
//   main.cu calls exactly this; all CUDA bookkeeping is hidden here.
//     acc      : OUT, resized to acc_total(c), the fixed-point (count|S1|S2) tally
//     final_x  : OUT, resized to c.n_walkers, each walker's final position
//     kernel_ms: OUT, milliseconds spent in the kernel itself (not H2D/D2H copies)
void run_ensemble_gpu(const GamdConfig& c,
                      std::vector<int64_t>& acc,
                      std::vector<double>& final_x,
                      float* kernel_ms);
