// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 3.26 : GPU BAM Sorting & Deduplication
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls sort_gpu() and markdup_gpu();
//   kernels.cu implements the host wrappers, the device kernels, and the Thrust
//   calls. Included only by .cu translation units (it declares __global__
//   kernels, so the plain host compiler must never see it -- that is why the CPU
//   reference lives in the separate pure-C++ reference_cpu.h).
//
// THE BIG IDEA (two classic data-parallel primitives, via Thrust + CUB)
//   This flagship-of-its-pattern teaches the GPU PRIMITIVES behind genomics
//   pipelines: a RADIX SORT and a SEGMENTED (group-by) REDUCTION.
//
//   (1) sort_gpu  -- COORDINATE SORT as thrust::sort_by_key:
//         key   = coord_key(ref,pos,strand)  (bam.h, packed into one uint64)
//         value = the read's original id
//       Thrust runs a parallel LSD radix sort on the 64-bit keys -- O(n) passes
//       over the data, bandwidth-bound, far faster than a CPU comparison sort on
//       big inputs. A tie-break sort on id first makes the order TOTAL so it
//       matches the CPU std::sort exactly (see kernels.cu for the two-key trick).
//
//   (2) markdup_gpu -- DUPLICATE MARKING as a SEGMENTED REDUCTION:
//         a) one thread per read computes dup_key (a map kernel).
//         b) thrust::sort_by_key brings equal-signature reads together.
//         c) thrust::reduce_by_key finds each group's BEST read (is_better_dup)
//            -- a segmented reduction: the GPU analogue of "GROUP BY signature,
//            keep MAX(quality)".
//         d) one thread per read marks it a duplicate unless it is its group's
//            chosen representative (a second map kernel).
//       All keys/scores are integers, so the reduction is order-independent and
//       the GPU's choice equals the CPU's exactly (deterministic, no epsilon).
//
//   WHY THRUST (no black box -- PATTERNS.md §5): radix sort and segmented
//   reduction are SOLVED primitives. Hand-rolling a correct, fast multi-pass
//   radix sort (per-digit histogram, exclusive scan, scatter) is a project in
//   itself; Thrust/CUB ship a tuned one. We document what each call computes and
//   the data layout it expects, so the learner understands the machinery even
//   though they call a library. Thrust is header-only (ships with CUDA), so NO
//   extra .lib is linked -- just #include <thrust/...>.
//
// READ THIS AFTER: bam.h, util/cuda_check.cuh, util/timer.cuh. Then kernels.cu.
// ===========================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "reference_cpu.h"   // ReadSet, ReadRecord (pure C++, safe in .cu)

// ---- (1) Coordinate sort -------------------------------------------------
// sort_gpu: coordinate-sort the reads on the GPU (thrust::sort_by_key on the
//   packed coord_key, with an id tie-break for a total, CPU-matching order).
//   Inputs : rs (reads in input order).
//   Output : `out` filled with the reads in genome order (resized to n).
//   kernel_ms : out-param, GPU time for the sort path (events), excludes the
//               host<->device copies so the number is comparable to the CPU sort.
void sort_gpu(const ReadSet& rs, std::vector<ReadRecord>& out, float* kernel_ms);

// ---- (2) Duplicate marking ----------------------------------------------
// markdup_gpu: flag PCR/optical duplicates on the GPU (map dup_key -> sort by
//   key -> reduce_by_key for the best per group -> map to mark non-best copies).
//   Inputs : rs (reads, any order).
//   Output : `is_dup` [n] indexed by original id (1 = duplicate, 0 = kept).
//   Returns the number of duplicates flagged (matches markdup_cpu exactly).
//   kernel_ms : out-param, GPU time for the dedup path (events).
int markdup_gpu(const ReadSet& rs, std::vector<uint8_t>& is_dup, float* kernel_ms);
