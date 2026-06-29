// ===========================================================================
// src/kernels.cuh  --  GPU Monte Carlo interface
// ---------------------------------------------------------------------------
// Project 5.01 : Monte Carlo Dose Calculation (simplified slab)
//
// THE BIG IDEA
//   Particle histories are INDEPENDENT, so each GPU thread tracks one photon
//   (grid-stride over millions of them). Two MC-specific lessons:
//     * PER-THREAD RNG: each thread seeds its own reproducible stream from its
//       history index (rng_seed in mc_physics.h) -- the shared header means the
//       CPU reproduces the identical histories for exact verification.
//     * ATOMIC SCORING: many threads deposit into the SAME depth bins, so the
//       tally uses atomicAdd. Because energy is INTEGER quanta, the atomic adds
//       are order-independent -> the GPU result is deterministic and equals the
//       CPU tally exactly (float dose would not have this property).
//
//   kernels.cu defines the kernel. main.cu calls dose_gpu().
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, mc_physics.h, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // DoseProblem, SimParams (pure C++, safe in .cu)

// Device kernel: each thread simulates one or more photon histories and scores
// integer dose into the shared `dose` tally via atomicAdd.
__global__ void dose_kernel(SimParams sp, unsigned long long n_photons,
                            unsigned long long seed,
                            unsigned long long* __restrict__ dose);

// Host wrapper: zero the device tally, launch the histories, copy the dose back.
//   dose      : resized to n_bins; filled with per-bin integer dose
//   kernel_ms : out-param, GPU kernel time (ms)
void dose_gpu(const DoseProblem& prob, std::vector<unsigned long long>& dose, float* kernel_ms);
