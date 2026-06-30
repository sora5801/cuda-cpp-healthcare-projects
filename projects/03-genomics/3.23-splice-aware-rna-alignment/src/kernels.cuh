// ===========================================================================
// src/kernels.cuh  --  GPU interface for batched splice-aware alignment
// ---------------------------------------------------------------------------
// Project 3.23 : Splice-Aware RNA Alignment   (REDUCED-SCOPE teaching version)
//
// THE BIG IDEA (contrast with 3.01's single-pair wavefront!)
//   3.01 parallelised ONE alignment across anti-diagonals. Here we parallelise
//   ACROSS READS: a sequencing run yields millions of short reads, each aligned
//   INDEPENDENTLY against the same reference gene model. That is the natural,
//   embarrassingly-parallel axis a real spliced aligner exploits.
//
//   Mapping: ONE THREAD BLOCK PER READ. A read is short (tens to a few hundred
//   bases), so one block fills that read's entire (M+1)x(N+1) DP table by
//   itself -- the block walks the table row by row, and the block's threads
//   cooperate on the columns WITHIN a row. Between rows the block synchronises
//   (__syncthreads) so row i is fully final before row i+1 starts; that mirrors
//   the CPU's top-to-bottom sweep and keeps the integers identical.
//
//        block b  ->  read b      (independent DP table; no cross-block comms)
//        thread t ->  helps fill columns j = t, t+blockDim.x, ... of each row
//
//   The per-cell math is the SHARED cell_recurrence() from reference_cpu.h, so
//   the kernel and the CPU reference cannot disagree (docs/PATTERNS.md §2).
//
//   This header is included only by .cu units (it declares a __global__). It
//   pulls in reference_cpu.h for the scoring + ReadBatch/AlignResult types.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, reference_cpu.h.
// ===========================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "reference_cpu.h"   // HD math, scoring constants, ReadBatch, AlignResult

// ---- Device kernel -------------------------------------------------------
// align_batch_kernel: one BLOCK aligns one read.
//   d_ref      : [N] encoded reference bases (read by every block)
//   N          : reference length
//   d_reads    : [R*M] encoded reads, row-major (M = padded read length)
//   d_read_lens: [R] true length of each read (padding is ignored)
//   M          : padded read length (rows per table)
//   R          : number of reads (== gridDim.x; one block each)
//   d_H        : [R*(M+1)*(N+1)] scratch for all DP tables (also returned so the
//                host can traceback). Block b writes only its own slot.
//   d_score    : [R] out: best cell value per read
//   d_end_i    : [R] out: 1-based read row of the best cell
//   d_end_j    : [R] out: 1-based ref  column of the best cell
// The DP table for a read lives in GLOBAL memory (it can be larger than shared
// memory for long references); the cooperating threads read/write it directly.
__global__ void align_batch_kernel(const uint8_t* __restrict__ d_ref, int N,
                                   const uint8_t* __restrict__ d_reads,
                                   const int* __restrict__ d_read_lens,
                                   int M, int R,
                                   int* __restrict__ d_H,
                                   int* __restrict__ d_score,
                                   int* __restrict__ d_end_i,
                                   int* __restrict__ d_end_j);

// ---- Host wrapper --------------------------------------------------------
// align_batch_gpu: do the whole GPU computation for a batch.
//   Uploads the reference + reads, launches R blocks (one per read), copies back
//   each read's best AlignResult AND the full DP tables (so main.cu can run the
//   SAME host traceback it runs on the CPU result -- proving they match cell for
//   cell). Reports the measured KERNEL time via *kernel_ms (CUDA events).
//     b         : the loaded batch
//     out       : resized to R; out[r] = GPU AlignResult for read r
//     H_all     : resized to R*(M+1)*(N+1); the GPU-filled DP tables
//     kernel_ms : out-param, milliseconds in the kernel itself (not copies)
void align_batch_gpu(const ReadBatch& b,
                     std::vector<AlignResult>& out,
                     std::vector<int>& H_all,
                     float* kernel_ms);
