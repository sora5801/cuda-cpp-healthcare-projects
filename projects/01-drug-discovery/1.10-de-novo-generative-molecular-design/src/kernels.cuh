// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 1.10 : De Novo Generative Molecular Design (reduced-scope teaching).
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls generate_and_score_gpu();
//   kernels.cu implements both the host wrapper and the device kernel. Included
//   only by .cu translation units (it declares a __global__ kernel, so the plain
//   C++ compiler must never see it -- that is why the CPU reference lives in the
//   separate pure-C++ reference_cpu.h).
//
// THE BIG IDEA
//   Generating N novel molecules is N INDEPENDENT stochastic jobs: molecule i is
//   sampled from its OWN RNG stream rng_seed(seed, i). So we give each molecule
//   its own GPU THREAD -- thread (blockIdx.x, threadIdx.x) generates and scores
//   molecule i = blockIdx.x * blockDim.x + threadIdx.x. This is the per-thread
//   RNG "Monte-Carlo histories" pattern (PATTERNS.md §1; flagship 5.01), here
//   producing thousands of candidate molecules per kernel launch -- the same
//   thing an RL rollout does, only with a Markov model instead of a transformer.
//
//   The transition MODEL is read-only and identical for every thread, so we put
//   it in CONSTANT memory: its broadcast cache serves the whole warp from one
//   fetch (same trick as the query fingerprint in flagship 1.12). The kernel
//   writes only two small per-molecule outputs (score, length), so there are NO
//   atomics and NO inter-thread communication -- embarrassingly parallel.
//
// READ THIS AFTER: generator.h, util/cuda_check.cuh, util/timer.cuh.
// Then read kernels.cu. The science / GPU-mapping is in ../THEORY.md.
// ===========================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "reference_cpu.h"   // Corpus, MarkovModel (pure C++, safe to include here)

// ---- Device kernel -------------------------------------------------------
// One thread == one molecule. The model is read from the __constant__ symbol
// defined in kernels.cu (NOT a parameter), so it is not in the signature.
//   n_gen   : number of molecules to generate (guards the ragged last block)
//   seed    : base RNG seed; this thread uses stream rng_seed(seed, i)
//   scores  : [n_gen] device array, OUT, integer milli-reward per molecule
//   lengths : [n_gen] device array, OUT, character length per molecule
__global__ void generate_kernel(int n_gen, unsigned long long seed,
                                int* __restrict__ scores,
                                int* __restrict__ lengths);

// ---- Host wrapper --------------------------------------------------------
// generate_and_score_gpu: do the whole GPU computation.
//   Uploads the trained model to constant memory, allocates the two device
//   output arrays, launches generate_kernel with a 1-D grid sized to n_gen,
//   copies results back, and reports the measured KERNEL time (CUDA events).
//
//   model     : the trained Markov model (built once on the host)
//   n_gen     : number of molecules to generate
//   seed      : base RNG seed (must match the CPU reference for bit-identity)
//   scores    : host OUT, resized to n_gen
//   lengths   : host OUT, resized to n_gen
//   kernel_ms : OUT, milliseconds spent in the kernel itself (not copies)
void generate_and_score_gpu(const MarkovModel& model, int n_gen, uint64_t seed,
                            std::vector<int>& scores, std::vector<int>& lengths,
                            float* kernel_ms);
