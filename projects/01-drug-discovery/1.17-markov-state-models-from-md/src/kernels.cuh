// ===========================================================================
// src/kernels.cuh  --  GPU MSM interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 1.17 : Markov State Models from MD
//
// THE BIG IDEA (pattern: PARALLEL ASSIGN + ATOMIC INTEGER REDUCTION)
//   Two steps of the MSM pipeline dominate at scale and are data-parallel; the
//   GPU does exactly these, and reuses the host helpers for everything else:
//
//   (A) k-means ASSIGN: one thread per FRAME finds its nearest centroid
//       (microstate). Fully independent -> a 1-D grid over the N frames. This is
//       the same kernel as flagship 11.09.
//   (B) k-means ACCUMULATE: each frame atomically adds its FIXED-POINT
//       coordinates to its microstate's running sum and bumps the count. Integer
//       atomics commute -> deterministic and CPU-matching.
//   (C) transition COUNT: one thread per time index t scatters the pair
//       (labels[t] -> labels[t+lag]) into the K x K count matrix C via an
//       INTEGER atomicAdd. Again integer -> order-independent -> reproducible and
//       bit-identical to the CPU count matrix.
//
//   The centroid divide, the transition-matrix normalization, and the tiny
//   eigen-analysis (pi, slowest timescale) are reused from reference_cpu.cpp, so
//   the only difference between the CPU and GPU paths is WHERE the two hot loops
//   run -- which is why main.cu can verify them as EXACTLY equal.
//
//   kernels.cu defines the kernels + the host wrapper msm_gpu(). main.cu calls
//   msm_gpu(); see docs/PATTERNS.md (clustering / atomic reduce).
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, msm.h, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // Dataset, MsmResult, and the shared host helpers

// (A) ASSIGN: labels[i] = index of the nearest centroid to frame i.
//   grid  : ceil(N / block) blocks ; block : 256 threads
//   thread (blockIdx.x, threadIdx.x) -> frame index i = bx*blockDim.x + tx
__global__ void assign_kernel(const float* __restrict__ x, int N, int D,
                              const float* __restrict__ centroids, int K,
                              int* __restrict__ labels);

// (B) ACCUMULATE: atomically add each frame's fixed-point coordinates to its
//   microstate's coordinate sum, and increment that microstate's count.
__global__ void accumulate_kernel(const float* __restrict__ x, int N, int D,
                                  const int* __restrict__ labels,
                                  unsigned long long* __restrict__ sum,
                                  unsigned int* __restrict__ count);

// (C) COUNT_TRANSITIONS: one thread per time index t in [0, N-lag); atomically
//   increment counts[labels[t]*K + labels[t+lag]] (the K x K transition tally).
__global__ void count_transitions_kernel(const int* __restrict__ labels, int N, int K, int lag,
                                         unsigned int* __restrict__ counts);

// Host wrapper: run the WHOLE MSM on the GPU (k-means for `iters` Lloyd steps +
// transition counting), reusing the host transition-matrix / spectral helpers so
// the result matches msm_cpu() exactly. Fills `out`; returns the GPU loop time
// (assign+accumulate+count kernels) via *kernel_ms.
MsmResult msm_gpu(const Dataset& d, int iters, float* kernel_ms);
