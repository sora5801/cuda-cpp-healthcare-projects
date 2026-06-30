// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 3.2 : Short-Read Mapping / Alignment
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls map_reads_gpu(); kernels.cu
//   implements both the host wrapper and the device kernel. Included only by .cu
//   translation units (it declares a __global__ kernel, so the plain C++
//   compiler must never see it -- that is why the CPU reference lives in the
//   separate pure-C++ header reference_cpu.h).
//
// THE BIG IDEA -- "independent jobs", one read per thread
//   Short-read mapping is embarrassingly parallel ACROSS READS: the best
//   position for read r does not depend on any other read. So we assign ONE GPU
//   THREAD PER READ. Thread (blockIdx.x, threadIdx.x) -- via a grid-stride loop
//   so a fixed grid covers any R -- owns read index `r` and runs the WHOLE
//   seed-and-extend pipeline for that read:
//
//     1. compute the read's leading k-mer code         (kmer_code, shared header)
//     2. binary-search the sorted reference index      (kmer_equal_range)
//     3. score the read at every candidate offset      (score_window)
//     4. keep the best (highest score, lowest offset)  (per-thread max-reduction)
//
//   Each thread keeps its running best in REGISTERS -- no atomics, no shared
//   memory, no inter-thread communication -- the simplest correct GPU structure
//   and a faithful miniature of how Parabricks/Bowtie batch millions of reads.
//   (Contrast 3.01, where ONE alignment is parallelized across an anti-diagonal
//   wavefront; here MANY whole alignments run in parallel, one per thread.)
//
//   Because every per-element function (kmer_code, kmer_equal_range,
//   score_window) is the SAME `__host__ __device__` code the CPU reference runs,
//   the GPU's chosen (pos, score) for each read is BIT-IDENTICAL to the CPU's,
//   so main.cu can verify with exact equality.
//
// READ THIS AFTER: reference_cpu.h (the shared core), util/cuda_check.cuh,
//                  util/timer.cuh. Then read kernels.cu.
// ===========================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "reference_cpu.h"   // MappingProblem, KmerIndex, MapResult, shared core

// ---- Device kernel --------------------------------------------------------
// map_reads_kernel: one thread maps one read (grid-stride over all reads).
//   ref        : [ref_len] device reference bases (0..3)
//   ref_len    : reference length
//   reads      : [n_reads * read_len] device reads, row-major
//   read_len   : per-read length
//   n_reads    : number of reads (guards the grid-stride loop)
//   sorted_codes / sorted_offsets : [n_kmers] the device-side k-mer index
//   n_kmers    : index length
//   out_pos / out_score / out_mism : [n_reads] device outputs (one per read)
// All pointers are __restrict__ (they do not alias) so the compiler may keep
// loads in registers. The kernel touches only global memory + registers.
__global__ void map_reads_kernel(const uint8_t* __restrict__ ref, int ref_len,
                                 const uint8_t* __restrict__ reads, int read_len,
                                 int n_reads,
                                 const uint64_t* __restrict__ sorted_codes,
                                 const int* __restrict__ sorted_offsets,
                                 int n_kmers,
                                 int* __restrict__ out_pos,
                                 int* __restrict__ out_score,
                                 int* __restrict__ out_mism);

// ---- Host wrapper ---------------------------------------------------------
// map_reads_gpu: the host-callable "do the whole GPU mapping" function. It
// uploads the reference, reads, and prebuilt index, launches map_reads_kernel,
// copies the per-read results back, and reports the measured KERNEL time (CUDA
// events) via *kernel_ms. main.cu calls exactly this; all CUDA bookkeeping is
// hidden here.
//   prob      : the loaded problem (reference + reads), host-side
//   index     : the prebuilt k-mer index (same one the CPU used -> identical seeds)
//   results   : resized to n_reads; filled with the GPU's per-read mapping
//   kernel_ms : out-param, milliseconds spent in the kernel itself (not copies)
void map_reads_gpu(const MappingProblem& prob, const KmerIndex& index,
                   std::vector<MapResult>& results, float* kernel_ms);
