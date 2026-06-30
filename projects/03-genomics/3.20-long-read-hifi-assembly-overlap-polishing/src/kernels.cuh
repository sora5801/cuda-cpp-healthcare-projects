// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for all-vs-all overlap chaining
// ---------------------------------------------------------------------------
// Project 3.20 : Long-Read HiFi Assembly Overlap & Polishing
//
// THE BIG IDEA
//   Scoring the overlap of read i against read j is INDEPENDENT of every other
//   pair, and there are N*(N-1)/2 such pairs -- the O(N^2) blow-up that makes
//   all-vs-all overlap the bottleneck of long-read assembly. So we give EACH
//   ORDERED PAIR its own GPU thread: thread `t` decodes its flat pair index back
//   into (i, j), reads the two reads' minimiser slices from global memory, and
//   runs the SAME anchor-merge + collinear-chaining DP the CPU reference runs --
//   in fixed on-thread scratch (no allocation), so the per-thread cost is bounded
//   by OVL_MAX_ANCHORS. This is the "score one item vs many, each independent"
//   pattern (PATTERNS.md sec 1; exemplar flagships 1.12 Tanimoto / 12.01 search).
//
//   Because the chain score is built from INTEGER link scores defined once in
//   overlap_core.h, the GPU result is bit-identical to the CPU's (tolerance 0).
//
//   This header is included by main.cu (host) and kernels.cu (device). It pulls
//   in reference_cpu.h for the ReadSet / OverlapResult / Minimizer types, which
//   are pure C++ and safe to compile under nvcc.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, reference_cpu.h,
// overlap_core.h. Then read kernels.cu. The GPU mapping is in ../THEORY.md.
// ===========================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "reference_cpu.h"   // ReadSet, OverlapResult, Minimizer, pair_index
#include "overlap_core.h"    // OVL_MAX_ANCHORS, ovl_chain_link_score, ...

// ---------------------------------------------------------------------------
// overlap_kernel: one thread per (i<j) read pair.
//   Inputs are the flattened ReadSet arrays already resident in device memory:
//     d_off, d_cnt : per-read [offset, count) into the minimiser arrays
//                    ([n_reads] each)
//     d_min_pos    : all minimiser positions, concatenated  ([total_min])
//     d_min_hash   : all minimiser hashes,    concatenated  ([total_min])
//   (We split Minimizer into two parallel arrays -- a struct-of-arrays layout --
//    so a thread streams the contiguous hash array cache-friendly; see THEORY
//    "GPU mapping".)
//   Outputs (one entry per pair, written at the thread's pair slot):
//     d_score   : best collinear chain score (int)
//     d_nanchor : number of shared-seed anchors used (int)
//   n_reads, n_pairs : sizes; the kernel guards the ragged last block with n_pairs.
// ---------------------------------------------------------------------------
__global__ void overlap_kernel(const int32_t*  __restrict__ d_off,
                               const int32_t*  __restrict__ d_cnt,
                               const int32_t*  __restrict__ d_min_pos,
                               const uint32_t* __restrict__ d_min_hash,
                               int n_reads, long long n_pairs,
                               int32_t* __restrict__ d_score,
                               int32_t* __restrict__ d_nanchor);

// ---------------------------------------------------------------------------
// overlap_gpu: host wrapper. Flattens the ReadSet into device arrays, launches
//   overlap_kernel over all pairs, times ONLY the kernel (CUDA events), and
//   returns one OverlapResult per pair in pair_index order (matching the CPU).
//     rs        : the loaded, sketched dataset.
//     out       : resized to rs.num_pairs(); filled with per-pair results.
//     kernel_ms : out-param, GPU-measured kernel time in milliseconds.
// ---------------------------------------------------------------------------
void overlap_gpu(const ReadSet& rs, std::vector<OverlapResult>& out, float* kernel_ms);
