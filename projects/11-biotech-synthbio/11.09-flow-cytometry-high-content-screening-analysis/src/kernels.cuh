// ===========================================================================
// src/kernels.cuh  --  GPU k-means interface
// ---------------------------------------------------------------------------
// Project 11.09 : Flow Cytometry & High-Content Screening Analysis
//
// THE BIG IDEA (tenth flagship pattern: PARALLEL ASSIGN + ATOMIC REDUCTION)
//   k-means alternates two GPU steps:
//     * ASSIGN: one thread per event finds its nearest centroid -> independent.
//     * ACCUMULATE: every event atomically adds its coordinates to its cluster's
//       running sum (and bumps the count) -- a SCATTER-REDUCTION via atomicAdd.
//   The centroid divide (sum/count) is a tiny host step reused from the CPU
//   reference, so CPU and GPU produce identical centroids. To make the atomic
//   reduction DETERMINISTIC and CPU-matching, coordinates are accumulated in
//   FIXED-POINT integers (kmeans.h) -- atomic integer adds commute.
//
//   kernels.cu defines the kernels. main.cu calls kmeans_gpu().
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, kmeans.h, reference_cpu.h.
// ===========================================================================
#pragma once

#include <cstdint>
#include <vector>
#include "reference_cpu.h"   // Dataset + shared helpers (pure C++, safe in .cu)

// ASSIGN: labels[i] = index of the nearest centroid to event i.
__global__ void assign_kernel(const float* __restrict__ x, int N, int D,
                              const float* __restrict__ centroids, int K,
                              int* __restrict__ labels);

// ACCUMULATE: atomically add each event's fixed-point coordinates to its
// cluster's sum, and increment the cluster count.
__global__ void accumulate_kernel(const float* __restrict__ x, int N, int D,
                                  const int* __restrict__ labels,
                                  unsigned long long* __restrict__ sum,
                                  unsigned int* __restrict__ count);

// Host wrapper: run `iters` Lloyd iterations on the GPU. Fills labels, centroids,
// and sizes; returns the final inertia and the GPU loop time via kernel_ms.
double kmeans_gpu(const Dataset& d, int iters, std::vector<float>& centroids,
                  std::vector<int>& labels, std::vector<unsigned int>& sizes, float* kernel_ms);
