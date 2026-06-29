// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for the FEP/TI ensemble
// ---------------------------------------------------------------------------
// Project 1.5 : Free Energy Perturbation / Thermodynamic Integration
//
// THE BIG IDEA (PATTERNS.md: "same sampler for many parameter sets")
//   Thermodynamic Integration needs one EQUILIBRIUM AVERAGE < dU/dlambda > per
//   lambda-window, and the windows are mutually INDEPENDENT (each is its own MC
//   chain at its own coupling). So we give each window its own GPU thread: the
//   thread runs the whole Metropolis chain (run_chain() in alchemy.h, in
//   registers) and writes one double. There is no inter-thread communication --
//   pure embarrassing parallelism over windows, the same mapping as the SEIR
//   ensemble (9.02). Production FEP scales the SAME way (one GPU per window).
//
//   Because run_chain() is the shared __host__ __device__ sampler and the RNG is
//   counter-based (reproducible regardless of who runs it), the GPU per-window
//   results match the CPU reference to round-off. kernels.cu defines the kernel.
//
// Included only by .cu translation units (it declares a __global__), so the
// pure-C++ reference uses reference_cpu.h instead.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, alchemy.h, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // AlchemyConfig (pure C++, safe inside a .cu)

// ---- Device kernel -------------------------------------------------------
// ti_kernel: thread `w` runs the MC chain for lambda-window w and writes its
//   < dU/dlambda >_lambda estimate to dvals[w] (and its accepted-move count to
//   accepted[w]). Config is passed BY VALUE so each thread has it in registers.
//     grid  : ceil(W / block) blocks         (W = number of windows)
//     block : THREADS_PER_BLOCK threads
//     map   : w = blockIdx.x * blockDim.x + threadIdx.x  (one window per thread)
__global__ void ti_kernel(AlchemyConfig c,
                          double* __restrict__ dvals,
                          long long* __restrict__ accepted);

// ---- Host wrapper --------------------------------------------------------
// integrate_gpu: launch one thread per lambda-window, copy the per-window
//   < dU/dlambda > (and accepted counts) back, and report the KERNEL time
//   (CUDA events) via *kernel_ms. main.cu then trapezoid-integrates dvals over
//   lambda to get DeltaG_TI. All CUDA bookkeeping is hidden here.
void integrate_gpu(const AlchemyConfig& c,
                   std::vector<double>& dvals,
                   std::vector<long long>& accepted,
                   float* kernel_ms);
