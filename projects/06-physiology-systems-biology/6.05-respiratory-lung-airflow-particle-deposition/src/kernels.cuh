// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 6.5 : Respiratory / Lung Airflow & Particle Deposition
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls deposition_gpu(); kernels.cu
//   implements both the host wrapper and the device kernel. Included only by
//   .cu translation units (it declares a __global__, so the plain C++ compiler
//   must never see it -- that is why the CPU reference lives in the separate
//   pure-C++ header reference_cpu.h).
//
// THE BIG IDEA -- Lagrangian particle tracking, one thread per particle
//   Inhaled aerosol particles are INDEPENDENT: each one random-walks down the
//   airway tree on its own, deposits somewhere (or is exhaled), and never talks
//   to the others. That is a textbook "embarrassingly parallel" workload, so we
//   give EACH PARTICLE ITS OWN GPU THREAD (a grid-stride loop covers millions of
//   them with a fixed grid). This is the pattern the catalog calls out:
//   "custom CUDA kernels for Lagrangian force integration (one thread per
//   particle) ... with atomic-add deposition counters."
//
//   Two lessons make this a good teaching kernel (docs/PATTERNS.md sections 2-3):
//     * PER-THREAD RNG: each thread seeds its own reproducible stream from its
//       particle index (rng_seed in lung_physics.h). Because that header is
//       SHARED with the CPU reference, both sides replay identical histories.
//     * ATOMIC INTEGER SCORING: many threads deposit into the SAME per-generation
//       counters, so the tally uses atomicAdd on 64-bit INTEGERS. Integer adds
//       commute, so the GPU result is deterministic and equals the CPU tally
//       EXACTLY (a floating-point tally would not have this property).
//
//   kernels.cu defines the kernel. main.cu calls deposition_gpu().
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, lung_physics.h,
//                  reference_cpu.h. Then read kernels.cu.
// ===========================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "reference_cpu.h"   // DepositionProblem, lung::Airway (pure C++, safe in .cu)

// ---- Device kernel -------------------------------------------------------
// deposition_kernel: each thread tracks one or more particle histories (grid-
// stride) and atomically increments the integer counter for the generation each
// particle deposits in.
//   p        : particle properties (by value -> one copy per thread in registers)
//   aw       : airway geometry (by value -> small fixed-size struct, per thread)
//   n        : number of particle histories to run
//   seed     : base RNG seed (particle i uses stream (seed, i))
//   n_gen    : number of generations (tally index n_gen is the "exhaled" bucket)
//   tally    : device array of length n_gen+1, updated with atomicAdd
__global__ void deposition_kernel(lung::Particle p, lung::Airway aw,
                                  uint64_t n, uint64_t seed, int n_gen,
                                  unsigned long long* __restrict__ tally);

// ---- Host wrapper --------------------------------------------------------
// deposition_gpu: the host-callable "do the whole GPU computation" function.
//   Allocates + zeroes the device tally, launches deposition_kernel over all
//   particles, copies the tally back, and reports the measured KERNEL time
//   (CUDA events) via *kernel_ms. main.cu calls exactly this; all CUDA
//   bookkeeping is hidden here.
//
//   prob      : the deposition experiment (aerosol + breathing + MC counts)
//   aw        : the airway geometry built by build_airway() (shared with CPU)
//   tally     : host output, resized to n_gen+1 (output parameter)
//   kernel_ms : out-param, milliseconds spent in the kernel itself (not copies)
void deposition_gpu(const DepositionProblem& prob, const lung::Airway& aw,
                    std::vector<uint64_t>& tally, float* kernel_ms);
