// ===========================================================================
// src/kernels.cuh  --  GPU Monte Carlo interface (declarations + the big idea)
// ---------------------------------------------------------------------------
// Project 2.7 : Monte Carlo Protein Structure Sampling (HP lattice model)
//
// THE BIG IDEA
//   Monte Carlo replicas are INDEPENDENT random walks, so this is the textbook
//   "ensemble of independent histories" GPU pattern (PATTERNS.md §1): we give
//   each replica its OWN GPU THREAD. With R replicas and a block of B threads we
//   launch ceil(R / B) blocks; thread (blockIdx.x, threadIdx.x) owns replica
//   r = blockIdx.x * blockDim.x + threadIdx.x and runs the entire walk for it.
//
//   Two MC-specific lessons live here:
//     * PER-THREAD RNG: each thread seeds its own reproducible splitmix64 stream
//       from its replica index (rng_seed in mc_moves.h). Because the stream is
//       shared code, the CPU reproduces each replica's identical walk -- which is
//       what makes the GPU-vs-CPU check EXACT, not approximate.
//     * NO ATOMICS NEEDED: unlike a Monte-Carlo *tally* (where many threads add
//       into shared bins and need atomicAdd, e.g. project 5.01), here each thread
//       writes only its OWN result slot out[r]. Independent outputs => no
//       contention => no atomics. The shared work is read-only (the sequence and
//       the Boltzmann tables), so threads never step on each other.
//
//   kernels.cu implements the kernel + host wrapper. main.cu calls sample_gpu().
//
// This header is included ONLY by .cu units (it declares a __global__). The CPU
// reference uses reference_cpu.h instead. Both pull McProblem/McResult from
// mc_moves.h, so the two paths share one definition of the physics.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, mc_moves.h, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // McProblem, McResult, boltzmann_table_size (pure C++)

// Device kernel: each thread runs ONE replica's full Metropolis walk and writes
// its {best_energy, final_energy} into out[replica]. The Boltzmann tables are
// passed as a flat device array (one table per replica, stride = table_stride).
__global__ void sample_kernel(McProblem prob, const double* __restrict__ tables,
                              int table_stride, McResult* __restrict__ out);

// Host wrapper: upload the prebuilt Boltzmann tables, launch one thread per
// replica, copy the per-replica results back.
//   prob      : the problem (passed by value -> lives in constant/param space)
//   tables    : host flat array, n_replicas * boltzmann_table_size() doubles
//   out       : resized to n_replicas; filled with each replica's result
//   kernel_ms : out-param, GPU kernel time in milliseconds (CUDA events)
void sample_gpu(const McProblem& prob, const std::vector<double>& tables,
                std::vector<McResult>& out, float* kernel_ms);
