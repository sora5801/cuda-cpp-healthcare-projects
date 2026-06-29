// ===========================================================================
// src/kernels.cuh  --  GPU multi-walker metadynamics interface
// ---------------------------------------------------------------------------
// Project 1.6 : Enhanced Sampling -- Metadynamics & Replica Exchange
//
// THE BIG IDEA (the ENSEMBLE / thread-per-trajectory pattern; PATTERNS.md §1)
//   Multi-walker metadynamics runs many independent walkers that all explore the
//   same free-energy landscape. Each walker's trajectory is sequential in time
//   but INDEPENDENT of the others, so we give each walker its own GPU thread:
//   the thread runs the FULL Langevin + hill-deposition history (metad.h) and
//   writes back its private bias grid and a summary. No inter-thread comms ->
//   embarrassingly parallel (just like flagships 9.02 SEIR and 13.02 PBPK).
//
//   Because the integrator (run_walker) is SHARED with the CPU reference via
//   metad.h, the GPU results match the reference to machine precision. The kernel
//   itself is defined in kernels.cu.
//
// MEMORY NOTE
//   Each walker needs its OWN bias grid of nbins doubles. We allocate one big
//   device buffer of (n_walkers * nbins) doubles and give thread `id` the slice
//   [id*nbins, (id+1)*nbins). That slice lives in GLOBAL memory; the walker reads
//   /writes it through metad.h's bias_value/bias_grad/deposit_hill helpers. (A
//   register-resident grid would be ideal but nbins is a runtime value, so a
//   per-thread global slice is the simple, correct choice -- see THEORY.md.)
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, metad.h, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>

#include "metad.h"          // metad::Model, WalkerResult (pure, safe in .cu and .cpp)
#include "reference_cpu.h"   // MetadConfig, ensemble_size, walker_start

// Device kernel: thread `id` integrates walker `id` and writes its WalkerResult.
//   d_bias : device buffer [n_walkers * nbins]; thread id owns slice id*nbins...
//   d_out  : device buffer [n_walkers] of WalkerResult summaries.
__global__ void metad_kernel(MetadConfig c, double* __restrict__ d_bias,
                             metad::WalkerResult* __restrict__ d_out);

// Host wrapper: launch one thread per walker, copy results + form the
// ensemble-average bias grid on the host, and time the kernel.
//   results   : filled with n_walkers WalkerResult summaries.
//   mean_bias : filled with the nbins-long ensemble-average bias grid.
//   kernel_ms : GPU-measured kernel time (teaching artifact, never a benchmark).
void integrate_gpu(const MetadConfig& c,
                  std::vector<metad::WalkerResult>& results,
                  std::vector<double>& mean_bias,
                  float* kernel_ms);
