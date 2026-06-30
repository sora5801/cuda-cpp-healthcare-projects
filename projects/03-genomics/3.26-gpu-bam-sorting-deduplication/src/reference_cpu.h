// ===========================================================================
// src/reference_cpu.h  --  Read set + CPU reference for sort + dedup
// ---------------------------------------------------------------------------
// Project 3.26 : GPU BAM Sorting & Deduplication
//
// Pure C++ (no CUDA) so the plain host compiler can build reference_cpu.cpp.
// The per-record key/compare math lives in bam.h (shared __host__ __device__).
// kernels.cu reuses ReadSet + these declarations so CPU and GPU run the SAME
// algorithm and can be compared exactly.
//
// THE TWO OPERATIONS (mirrored on the GPU in kernels.cu):
//   sort_cpu    : coordinate-sort the reads (ref, pos, strand, then id).
//   markdup_cpu : flag PCR/optical duplicates -- among reads sharing a
//                 duplicate signature, keep the highest base-quality copy.
//
// READ THIS AFTER: bam.h. READ BEFORE: reference_cpu.cpp, kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "bam.h"   // ReadRecord, coord_key, dup_key, coord_less, is_better_dup

// ---------------------------------------------------------------------------
// ReadSet -- the loaded collection of aligned reads plus a little metadata.
//   `reads` is a flat array of ReadRecord in INPUT order (each record's `id`
//   equals its index here). num_refs is how many chromosomes the synthetic
//   genome has (used only for reporting). This is the unit both the CPU
//   reference and the GPU pipeline consume.
// ---------------------------------------------------------------------------
struct ReadSet {
    int num_refs = 0;                  // number of reference sequences (chromosomes)
    std::vector<ReadRecord> reads;     // [n] aligned reads, in input order
    int n() const { return static_cast<int>(reads.size()); }
};

// Load the text read set (format documented in data/README.md):
//   line 1: "<n> <num_refs>"
//   next n lines: "<ref_id> <pos> <strand> <mate_pos> <base_qual_sum>"
//   (the original index `id` is assigned from line order, 0..n-1)
// Throws std::runtime_error on a missing file / bad header / truncation /
// out-of-range field (so demos fail loudly rather than on garbage).
ReadSet load_readset(const std::string& path);

// ---------------------------------------------------------------------------
// sort_cpu -- coordinate-sort a copy of the reads using coord_less (bam.h).
//   Fills `out` with the reads in genome order. This is the trusted baseline
//   the GPU thrust::sort_by_key must reproduce exactly. O(n log n).
// ---------------------------------------------------------------------------
void sort_cpu(const ReadSet& rs, std::vector<ReadRecord>& out);

// ---------------------------------------------------------------------------
// markdup_cpu -- mark PCR/optical duplicates.
//   Input:  the reads (any order).
//   Output: `is_dup` [n], indexed by the read's original `id`: 1 if the read is
//           a duplicate (a non-best copy of some fragment), 0 if it is kept.
//   Returns the number of reads flagged as duplicates.
//   Method: group reads by dup_key; within each group keep the single best copy
//   (is_better_dup) and flag the rest. Deterministic via the id tie-breaks.
//   This is the baseline the GPU reduce_by_key + mark kernel must match exactly.
// ---------------------------------------------------------------------------
int markdup_cpu(const ReadSet& rs, std::vector<uint8_t>& is_dup);
