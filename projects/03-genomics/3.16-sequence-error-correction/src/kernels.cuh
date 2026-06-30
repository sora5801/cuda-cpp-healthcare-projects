// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 3.16 : Sequence Error Correction  (k-mer spectrum / trusted-k-mer)
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls correct_reads_gpu(), which
//   runs BOTH GPU phases and returns the corrected reads. This header is included
//   only by .cu translation units (it declares __global__ kernels, so the plain
//   host C++ compiler must never see it -- the CPU reference + shared physics
//   live in the pure-C++ reference_cpu.h, which this file also includes).
//
// THE TWO GPU PHASES (full mapping in ../THEORY.md sec 4)
//   Phase 1 -- BUILD THE SPECTRUM (a histogram via atomics):
//     Every (read, position) pair yields one k-mer; counting them is a histogram.
//     We launch ONE THREAD PER READ, and each thread walks its read's k-mers and
//     does atomicAdd(&counts[code], 1). Integer atomics COMMUTE, so the final
//     table is identical regardless of thread order -> deterministic and
//     bit-identical to the serial CPU count. This is the GPU-hash-table-for-
//     k-mer-counting pattern from the catalog, simplified to a direct-indexed
//     exact table (no hashing) because 4^9 slots fit in ~1 MB.
//
//   Phase 2 -- CORRECT THE READS (independent jobs):
//     Correcting read i never touches read j's bytes and only READS the (now
//     frozen) spectrum table. So we launch ONE THREAD PER READ again; each thread
//     calls the shared correct_one_read() from reference_cpu.h. Because that is
//     the exact same inline function the CPU reference runs, the GPU output
//     matches the CPU output byte-for-byte (verified by == in main.cu).
//
//   This is PATTERNS.md sec 1 "score one query vs N items, each independent"
//   (here: process N reads against a shared spectrum) + sec 2's HD-core idiom.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, reference_cpu.h.
// Then read kernels.cu.
// ===========================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "reference_cpu.h"   // ReadSet, KMER_TABLE_N, the shared HD physics

// ---- Phase-1 kernel: build the k-mer spectrum ----------------------------
// One thread per read; the thread atomicAdds into the shared count table for
// each valid k-mer in its read.
//   d_bases  : [total bases] all reads concatenated (device)
//   d_offset : [n+1] CSR offsets (device)
//   d_length : [n]   per-read lengths (device)
//   n        : number of reads
//   d_counts : [KMER_TABLE_N] spectrum table (device, pre-zeroed) -- OUTPUT
__global__ void count_kmers_kernel(const char* __restrict__ d_bases,
                                   const int* __restrict__ d_offset,
                                   const int* __restrict__ d_length,
                                   int n,
                                   uint32_t* d_counts);

// ---- Phase-2 kernel: correct the reads -----------------------------------
// One thread per read; the thread runs the shared correct_one_read() physics
// against the frozen spectrum and writes the corrected bytes + its change count.
//   d_bases     : [total bases] raw reads concatenated (device, read-only)
//   d_offset    : [n+1] CSR offsets (device)
//   d_length    : [n]   per-read lengths (device)
//   n           : number of reads
//   d_counts    : [KMER_TABLE_N] frozen spectrum (device, read-only)
//   thresh      : trust threshold T
//   d_corrected : [total bases] corrected reads concatenated (device) -- OUTPUT
//   d_changes   : [n] number of substitutions per read (device) -- OUTPUT
__global__ void correct_reads_kernel(const char* __restrict__ d_bases,
                                     const int* __restrict__ d_offset,
                                     const int* __restrict__ d_length,
                                     int n,
                                     const uint32_t* __restrict__ d_counts,
                                     uint32_t thresh,
                                     char* __restrict__ d_corrected,
                                     int* __restrict__ d_changes);

// ---- Host wrapper: run BOTH phases on the GPU ----------------------------
// correct_reads_gpu: upload the reads, build the spectrum (phase 1), correct the
// reads (phase 2), download results, and return per-phase kernel timings.
//   reads            : the loaded read set (host)
//   thresh           : trust threshold T (counts >= T => trusted)
//   counts_out       : resized to KMER_TABLE_N; the GPU-built spectrum (so main
//                      can verify it against the CPU spectrum)
//   corrected_out    : resized to reads.bases.size(); the corrected bytes
//   changes_out      : resized to reads.n; per-read substitution counts
//   count_ms         : out-param, phase-1 kernel time (ms)
//   correct_ms       : out-param, phase-2 kernel time (ms)
void correct_reads_gpu(const ReadSet& reads, uint32_t thresh,
                       std::vector<uint32_t>& counts_out,
                       std::vector<char>& corrected_out,
                       std::vector<int>& changes_out,
                       float* count_ms, float* correct_ms);
