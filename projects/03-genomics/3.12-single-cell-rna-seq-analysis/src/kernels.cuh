// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 3.12 : Single-Cell RNA-seq Analysis  (reduced-scope teaching version)
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls run_gpu(); kernels.cu
//   implements the two host wrappers and the two device kernels. Included only
//   by .cu translation units (it declares __global__ kernels, so the plain C++
//   compiler must never see it -- that is why the CPU reference lives in the
//   separate pure-C++ reference_cpu.h).
//
// THE TWO STEPS THE GPU DOES (both embarrassingly parallel over cells)
//   (1) normalize_kernel : one thread per CELL. The thread sums its cell's G
//       counts (the library size) and writes the normalized row (counts-per-
//       target + log1p) using the SHARED math in scrna.h. No communication
//       between threads -> a textbook map.
//   (2) knn_kernel       : one thread per QUERY CELL. The thread scans all N
//       cells, keeps a fixed-size top-k list in registers/local memory, and
//       writes its k nearest neighbours. This is the O(N^2) step the deep dive
//       flags as the GPU win; each query is independent (the "score one item vs
//       N, each independent" pattern -- docs/PATTERNS.md sec 1, exemplar 1.12).
//
//   Both kernels call the SAME inline functions the CPU reference calls, so the
//   results match (CLAUDE.md section 5). main.cu verifies index-for-index.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, scrna.h. Then kernels.cu.
// ===========================================================================
#pragma once

#include "reference_cpu.h"   // Dataset, KnnGraph (shared host types)

// ---- Device kernels (defined in kernels.cu; declared here for documentation) ----
//
// normalize_kernel: thread c normalizes cell c's whole row.
//   counts     : [N*G] device, raw counts, row-major.
//   N, G       : dimensions.
//   target_sum : the fixed total each cell is scaled to.
//   normalized : [N*G] device OUTPUT, the normalized matrix.
__global__ void normalize_kernel(const float* __restrict__ counts, int N, int G,
                                 double target_sum, float* __restrict__ normalized);

// knn_kernel: thread q finds the k nearest neighbours of query cell q.
//   normalized : [N*G] device, the normalized matrix (read by every thread).
//   N, G, k    : dimensions and neighbour count.
//   nbr_idx    : [N*k] device OUTPUT neighbour indices (nearest first).
//   nbr_dist   : [N*k] device OUTPUT Euclidean distances (ascending).
__global__ void knn_kernel(const float* __restrict__ normalized, int N, int G, int k,
                           int* __restrict__ nbr_idx, float* __restrict__ nbr_dist);

// ---- Host wrapper --------------------------------------------------------
// run_gpu: do the WHOLE GPU pipeline (normalize + KNN) and fill `out`.
//   Allocates device buffers, uploads the raw counts, launches the two kernels,
//   copies the normalized matrix + neighbour graph back, and reports the summed
//   KERNEL time (CUDA events) via *kernel_ms. main.cu calls exactly this; all
//   CUDA bookkeeping is hidden here.
//
//   d         : the loaded dataset (host, read-only).
//   out       : host output (normalized + KNN graph), filled here.
//   kernel_ms : out-param, milliseconds spent in the two kernels (not copies).
void run_gpu(const Dataset& d, KnnGraph& out, float* kernel_ms);
