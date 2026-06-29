// ===========================================================================
// src/kernels.cuh  --  GPU aggregation-scan interface (declarations + idea)
// ---------------------------------------------------------------------------
// Project 1.34 : Amyloid / Aggregation Propensity Prediction
//
// THE BIG IDEA (sliding-window 1-D conv, batched over sequences)
//   ONE BLOCK PER PROTEIN, ONE THREAD PER RESIDUE. Each block:
//     1. Looks up its protein's per-residue intrinsic propensities and stages
//        them into SHARED MEMORY -- a TILE of `len` floats plus a HALO of `half`
//        extra on each side (zero-padded past the termini). This is the
//        canonical shared-memory tiling that makes the window mean read on-chip
//        memory instead of re-reading global memory W times (flagship 7.10).
//     2. After __syncthreads, each thread computes the centered windowed mean
//        for its residue by calling windowed_mean() from propensity.h -- the
//        SAME function the CPU reference calls, so results match to fp epsilon.
//     3. The smoothed profile is written to global memory; a tiny per-block
//        reduction (block-stride loop in one thread) turns it into the protein's
//        AggResult (peak score+pos, prone count, longest APR). The reduction is
//        done by a single thread per block on purpose: it is O(len), trivially
//        deterministic, and avoids any float-atomic non-associativity (PATTERNS
//        §3) -- correctness-you-can-see over a marginal speedup on short chains.
//
//   kernels.cu defines the kernel + constant-memory scale upload. main.cu calls
//   scan_dataset_gpu(). Included only by .cu units (it declares a __global__).
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, reference_cpu.h,
//                  propensity.h. Then read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // Dataset, AggResult (pure C++, safe in .cu)

// Threads per block = the maximum protein length we tile in one block. 1024 is
// the hardware ceiling on threads/block (sm_75..sm_89) and comfortably covers
// the proteins/domains this teaching demo handles. Sequences longer than this
// are rejected by the host wrapper with a clear message (see kernels.cu); the
// production fix (multi-block tiling per sequence) is left as an exercise.
static constexpr int AGG_MAX_LEN = 1024;

// Device kernel: one block scans one protein; one thread owns one residue.
//   flat_codes : [num*stride] padded amino-acid indices (row-major)
//   lengths    : [num] real length of each protein
//   stride     : padded row width
//   half       : half-window; full window W = 2*half + 1
//   threshold  : prone if smoothed score >= threshold
//   smoothed   : [num*stride] OUTPUT smoothed profiles (for verify + demo)
//   results    : [num] OUTPUT AggResult per protein
// Launched with `half`-aware dynamic shared memory (see scan_dataset_gpu).
__global__ void agg_scan_kernel(const int* __restrict__ flat_codes,
                                const int* __restrict__ lengths,
                                int stride, int half, float threshold,
                                float* __restrict__ smoothed,
                                AggResult* __restrict__ results);

// Host wrapper: upload the propensity scale (to constant memory) + the flat
// batch, launch one block per protein, copy the smoothed profiles and per-
// protein results back, and report the measured KERNEL time via *kernel_ms.
//   ds        : the loaded batch
//   window    : sliding-window width W (odd)
//   threshold : prone-residue cutoff
//   results   : [num] output AggResults
//   smoothed  : [num*stride] output smoothed profiles
//   kernel_ms : out-param, milliseconds in the kernel itself (not the copies)
void scan_dataset_gpu(const Dataset& ds, int window, float threshold,
                      std::vector<AggResult>& results,
                      std::vector<float>& smoothed, float* kernel_ms);
