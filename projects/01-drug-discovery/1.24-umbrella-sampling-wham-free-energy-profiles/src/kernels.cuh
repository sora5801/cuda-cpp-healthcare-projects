// ===========================================================================
// src/kernels.cuh  --  GPU umbrella-sampling interface (declarations + the idea)
// ---------------------------------------------------------------------------
// Project 1.24 : Umbrella Sampling / WHAM Free Energy Profiles
//
// THE BIG IDEA (the ENSEMBLE pattern, PATTERNS.md section 1: "the same simulation
// for many independent parameter sets -> thread per simulation")
//   Umbrella sampling runs N independent biased simulations, one per window. Each
//   window is a sequential time-loop (its trajectory depends on its own history)
//   but is COMPLETELY independent of the other windows -- the textbook
//   "embarrassingly parallel across windows" the deep-dive calls out. So we give
//   each window its OWN GPU THREAD: thread k runs the full Langevin trajectory for
//   window k (in registers/local memory) and writes window k's histogram into its
//   own private slice of global memory.
//
//   Because each thread owns a disjoint histogram slice, there is NO cross-thread
//   contention and NO need for atomics here -- a cleaner variant of the
//   histogram-accumulation pattern than the all-threads-into-one-tally case (cf.
//   5.01 Monte Carlo, where many threads DO share bins and must use atomicAdd).
//   Counts are integers, so the result is deterministic and exactly matches the
//   CPU reference (PATTERNS.md section 3).
//
//   The per-step physics is shared with the CPU via umbrella.h, so GPU and CPU
//   histograms are bit-identical. WHAM post-processing (turning the histograms
//   into a PMF) runs on the CPU for both paths -- it is cheap and serial. This
//   header is included only by .cu units (it declares a __global__).
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, umbrella.h. Then kernels.cu.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // UmbrellaConfig, window_spec, total_hist_size (pure C++, safe in .cu)

// ---- Device kernel -------------------------------------------------------
// window_kernel: thread `k` simulates umbrella window k and fills its histogram
// slice hist[k*nbins .. k*nbins+nbins-1].
//   c    : the whole experiment, passed BY VALUE so every thread has its own copy
//          in registers/constant-ish memory (it is a small POD struct -- cheap to
//          copy, and avoids a global-memory indirection on every access).
//   hist : device pointer to the flat [n_windows * nbins] count array (zeroed by
//          the host before launch). __restrict__ promises it does not alias.
// Launch config and thread->window mapping are documented at the definition.
__global__ void window_kernel(UmbrellaConfig c, unsigned int* __restrict__ hist);

// ---- Host wrapper --------------------------------------------------------
// sample_windows_gpu: the host-callable "run all windows on the GPU" function.
//   Allocates and zeroes the device histogram, launches one thread per window,
//   copies the counts back, and reports the measured KERNEL time via *kernel_ms.
//   main.cu calls exactly this; all CUDA bookkeeping is hidden here.
//
//   c         : the experiment (same struct the CPU reference used)
//   hist_out  : host output, resized to total_hist_size(c) (output parameter)
//   kernel_ms : out-param, milliseconds spent in the kernel itself (CUDA events)
void sample_windows_gpu(const UmbrellaConfig& c,
                        std::vector<unsigned int>& hist_out,
                        float* kernel_ms);
