// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 3.14 : Metagenomic Taxonomic Classification
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls classify_gpu(); kernels.cu
//   implements both the host wrapper and the device kernel. Included only by .cu
//   translation units (it declares a __global__ kernel, so the plain C++ compiler
//   must never see it -- that is why the data model + CPU reference live in the
//   separate pure-C++ reference_cpu.h).
//
// THE BIG IDEA  (docs/PATTERNS.md sec 1: "score one query vs N items, each
//   independent" -- here, classify N reads against one shared reference table)
//   Classifying a read is INDEPENDENT of every other read, so we give each read
//   its own GPU thread (a grid-stride loop lets a modest grid cover millions of
//   reads). Each thread:
//     * slides a k-mer window across ITS read,
//     * probes the shared reference hash table in GLOBAL memory for each k-mer,
//     * tallies a per-taxon vote in a tiny per-thread register array, and
//     * writes the winning taxon id.
//   The probe logic is the SHARED classify_read() from kmer_core.h -- the same
//   function the CPU reference runs -- so GPU and CPU agree EXACTLY (tolerance 0).
//
//   Why the table is in GLOBAL (not constant) memory, unlike 1.12's query: the
//   reference table is large (potentially gigabytes for RefSeq) and read by every
//   thread at data-dependent addresses; that is the classic global-memory +
//   hardware-cache access pattern, not constant memory's broadcast pattern.
//
// READ THIS AFTER: kmer_core.h, util/cuda_check.cuh, util/timer.cuh. Then kernels.cu.
// ===========================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "reference_cpu.h"   // RefDatabase, ReadSet (pure C++ -- safe in a .cu)

// ---- Device kernel -------------------------------------------------------
// classify_kernel: one thread classifies one read (grid-stride over all reads).
//   bases   : [total_bases] all reads concatenated (device)
//   offset  : [n_reads] start of each read within `bases` (device)
//   length  : [n_reads] length of each read (device)
//   n_reads : number of reads (guards the grid-stride loop)
//   keys    : [capacity] reference hash-table k-mers (device)
//   taxa    : [capacity] reference hash-table taxon ids, 0 = empty (device)
//   capacity: number of table slots (power of two)
//   out     : [n_reads] winning taxon id per read (output, device)
// No atomics or shared memory: every read's output is independent.
__global__ void classify_kernel(const char* __restrict__ bases,
                                const int* __restrict__ offset,
                                const int* __restrict__ length,
                                int n_reads,
                                const uint64_t* __restrict__ keys,
                                const uint32_t* __restrict__ taxa,
                                uint64_t capacity,
                                uint32_t* __restrict__ out);

// ---- Host wrapper --------------------------------------------------------
// classify_gpu: the host-callable "do the whole GPU classification" function.
//   Uploads the reads and the reference table, launches classify_kernel, copies
//   the per-read taxon ids back, and reports the measured KERNEL time (CUDA
//   events) via *kernel_ms. main.cu calls exactly this; all CUDA bookkeeping is
//   hidden here.
//   reads      : the metagenomic sample (host)
//   db         : the reference hash table (host)
//   out        : host output, resized to reads.n_reads (output parameter)
//   kernel_ms  : out-param, milliseconds spent in the kernel itself (not copies)
void classify_gpu(const ReadSet& reads, const RefDatabase& db,
                  std::vector<uint32_t>& out, float* kernel_ms);
