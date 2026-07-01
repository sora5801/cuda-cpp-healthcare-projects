// ===========================================================================
// src/kernels.cuh  --  GPU track-structure interface (declarations + big idea)
// ---------------------------------------------------------------------------
// Project 5.11 : Microdosimetry & Track-Structure Simulation
//
// THE BIG IDEA
//   Particle tracks are INDEPENDENT: one primary's ionizations never influence
//   another's. So each GPU thread simulates ONE primary track (grid-stride over
//   millions of them). Two Monte-Carlo lessons this project teaches:
//
//     * PER-THREAD RNG: each thread seeds its own reproducible counter-based
//       stream from its track index (rng_seed in ts_physics.h). Because the CPU
//       reference seeds the same way and runs the same transport, it reproduces
//       the identical tracks -> the tallies can be verified EXACTLY.
//
//     * ATOMIC SCORING into shared histograms: many threads add to the SAME
//       lineal-energy bins and the same DNA-damage counters, so the tallies use
//       atomicAdd. Because every per-track quantity is an INTEGER count (energy
//       quanta, SSB/DSB counts, histogram increments), the atomic adds are
//       order-independent -> the GPU result is deterministic AND equals the CPU
//       tally to the bit. Floating-point tallies would not have this property
//       (PATTERNS.md §3); that is precisely why we quantise energy.
//
//   The catalog's fuller vision (one warp per track, cross-section tables in
//   constant/shared memory, sorting tracks by interaction type to cut warp
//   divergence) is described in THEORY.md "Where this sits in the real world".
//   Here we ship the readable one-thread-per-track version.
//
//   kernels.cu implements the kernel; main.cu calls track_gpu().
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, ts_physics.h, reference_cpu.h.
// ===========================================================================
#pragma once

#include "reference_cpu.h"   // TrackProblem, TrackParams, TrackTally (pure C++)

// ---- Device kernel -------------------------------------------------------
// Each thread simulates one or more primary tracks (grid-stride) and scores its
// integer results into the shared device tallies via atomicAdd.
//   tp        : simulation parameters, passed by value (lives in each thread's regs)
//   n_tracks  : total primary tracks to simulate
//   seed      : base RNG seed; track i uses the reproducible stream (seed, i)
//   d_quanta  : [1] total energy-quanta accumulator (device)
//   d_ssb     : [1] total single-strand-break accumulator (device)
//   d_dsb     : [1] total double-strand-break accumulator (device)
//   d_yhist   : [n_y_bins] lineal-energy histogram (device)
__global__ void track_kernel(TrackParams tp, unsigned long long n_tracks,
                             unsigned long long seed,
                             unsigned long long* __restrict__ d_quanta,
                             unsigned long long* __restrict__ d_ssb,
                             unsigned long long* __restrict__ d_dsb,
                             unsigned long long* __restrict__ d_yhist);

// ---- Host wrapper --------------------------------------------------------
// track_gpu: allocate + zero the device tallies, launch all tracks, copy the
//   results back into `tally`, and report the measured KERNEL time (CUDA events)
//   via *kernel_ms. main.cu calls exactly this; all CUDA bookkeeping is hidden.
//   tally     : filled with the aggregated integer results (output parameter)
//   kernel_ms : out-param, milliseconds spent in the kernel itself (not copies)
void track_gpu(const TrackProblem& prob, TrackTally& tally, float* kernel_ms);
