// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for all-vs-all read overlap
// ---------------------------------------------------------------------------
// Project 3.5 : De Novo Genome Assembly  (read-overlap stage)
//
// THE BIG IDEA
//   All-vs-all overlap of n reads has P = n*(n-1)/2 unordered pairs, and every
//   pair is scored INDEPENDENTLY (count shared minimizers). So we give each
//   PAIR its own GPU thread: thread p decodes its (i,j) coordinate, fetches the
//   two reads' sorted minimizer sketches from the flattened CSR buffers, and
//   runs the SAME count_shared_sorted() the CPU reference runs (from assembly.h)
//   -> identical integer result, no atomics, no shared memory. A grid-stride
//   loop lets one modest grid cover any P. This is the "independent jobs over
//   pairs" pattern (cf. 1.12's independent jobs over a library).
//
//   This header declares a __global__ kernel, so it is included ONLY by .cu
//   units. main.cu calls overlap_gpu(); the pure-C++ data model is in
//   assembly.h / reference_cpu.h.
//
// READ THIS AFTER: assembly.h (the shared math), util/cuda_check.cuh,
// util/timer.cuh. Then read kernels.cu. The GPU mapping is in ../THEORY.md.
// ===========================================================================
#pragma once

#include <vector>

#include "assembly.h"   // ReadSet, Overlap, count_shared_sorted, pair_to_ij

// ---- Device kernel -------------------------------------------------------
// overlap_kernel: score every read pair. out_score[p] = shared-minimizer count
// of the pair whose flat upper-triangle index is p.
//   d_mins   : [total minimizers] concatenated sorted-unique sketches (CSR)
//   d_offset : [n+1] CSR offsets into d_mins (read r -> [offset[r],offset[r+1]))
//   n        : number of reads
//   P        : number of pairs = n*(n-1)/2 (passed in to avoid recompute)
//   out_score: [P] device output, shared count per pair (flat triangle order)
__global__ void overlap_kernel(const minimizer_t* __restrict__ d_mins,
                               const int* __restrict__ d_offset,
                               int n, long long P,
                               int* __restrict__ out_score);

// ---- Host wrapper --------------------------------------------------------
// overlap_gpu: do the whole GPU computation and return the per-pair scores.
//   Uploads the CSR sketch buffers, launches overlap_kernel over all P pairs,
//   times the kernel (CUDA events), copies the [P] scores back, and frees the
//   device memory. main.cu then thresholds the scores into Overlap edges on the
//   host (cheap) so the GPU/CPU score arrays can be diffed exactly.
//     rs        : the sketched reads (CSR ReadSet from sketch_reads()).
//     out_score : host output, resized to num_pairs(rs.n); shared count/pair.
//     kernel_ms : out-param, GPU-measured kernel milliseconds (not copies).
void overlap_gpu(const ReadSet& rs, std::vector<int>& out_score, float* kernel_ms);
