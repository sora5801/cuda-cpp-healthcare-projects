// ===========================================================================
// src/kernels.cuh  --  GPU ensemble-of-walkers interface
// ---------------------------------------------------------------------------
// Project 1.32 : Alchemical Hydration Free Energy (delta-G_solv)
//
// THE BIG IDEA (the ensemble-over-threads pattern, after flagship 9.02)
//   A free-energy calculation runs many INDEPENDENT Monte Carlo chains: one per
//   (lambda-window, walker). Each chain is sequential in its own steps but has no
//   communication with the others, so we give every walker its OWN GPU THREAD.
//   The thread runs the full Metropolis loop (alchemy.h, in registers) and writes
//   one WalkerResult. There is no inter-thread communication and the per-window
//   averaging happens afterward on the host -- so the result is deterministic and
//   matches the CPU reference (which runs the identical run_walker()) to round-off.
//
//   The only device-memory traffic during sampling is reading the (shared, read-
//   only) solvent bath coordinates; the solute position and accumulators live in
//   registers. kernels.cu copies the bath to the device, launches one thread per
//   walker, and copies the WalkerResults back.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, alchemy.h, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // AlchConfig, BathStorage, WalkerResult (pure C++, safe in .cu)

// Host wrapper: run the WHOLE ensemble of walkers on the GPU (one thread each),
// fill `walkers` (sized total_walkers(c)), and report the kernel time in ms.
//   c       : the calculation config (windows, walkers, steps, system params)
//   bath    : the solvent geometry (copied to the device inside)
//   walkers : output, one WalkerResult per global walker (resized here)
//   kernel_ms : out-param, GPU-measured kernel time (teaching artifact only)
void run_gpu(const AlchConfig& c, const BathStorage& bath,
             std::vector<WalkerResult>& walkers, float* kernel_ms);
