// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 3.21 : Structural Variant (SV) Calling
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls sv_call_gpu(); kernels.cu
//   implements the host wrapper and the device kernel. Included only by .cu
//   translation units (it has a __global__ declaration, so the plain C++ compiler
//   must never see it -- that is why the CPU reference lives in a separate
//   pure-C++ header, reference_cpu.h).
//
// THE BIG IDEA (two patterns in one pipeline)
//   PATTERN A -- independent jobs (PATTERNS.md §1, exemplar 1.12):
//     Each candidate split read is RE-ALIGNED independently to refine its
//     breakpoint (banded Smith-Waterman over a +/- window, sv.h). One read = one
//     GPU thread: thread i owns read i = blockIdx.x*blockDim.x + threadIdx.x.
//
//   PATTERN B -- parallel assign + atomic reduce (PATTERNS.md §1, exemplar 11.09):
//     Each thread then VOTES its refined breakpoint into a shared histogram with
//     atomicAdd (support count) and a parallel length-sum (atomicAdd on the
//     deletion length). Integer atomics COMMUTE, so the GPU histogram equals the
//     CPU histogram EXACTLY -- determinism without a tolerance (PATTERNS.md §3/§4).
//
//   The histogram->calls merge is a tiny host step reused verbatim from the CPU
//   reference (histogram_to_calls), so CPU and GPU emit identical calls.
//
//   kernels.cu defines the kernel + sv_call_gpu(). main.cu calls sv_call_gpu().
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, sv.h, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // SvDataset, SvCall, histogram_to_calls (pure C++, safe in .cu)

// ---- Device kernel -------------------------------------------------------
// refine_and_vote_kernel: one thread per read. Refines the read's breakpoint by
// banded SW (sv.h) and atomically votes it into the histogram + length-sum.
//   reads_guess : [N] raw breakpoint guesses (ref coords)
//   reads_dellen: [N] estimated deletion lengths (bp)
//   reads_flank : [N*SV_FLANK] read left-flank base codes, row-major
//   N           : number of reads (guards the ragged last block)
//   ref         : [ref_len] reference base codes (read-only)
//   ref_len     : reference length
//   hist        : [ref_len] output support histogram (atomicAdd target)
//   len_sum     : [ref_len] output deletion-length sum per bin (atomicAdd target)
__global__ void refine_and_vote_kernel(const int* __restrict__ reads_guess,
                                       const int* __restrict__ reads_dellen,
                                       const signed char* __restrict__ reads_flank,
                                       int N,
                                       const signed char* __restrict__ ref, int ref_len,
                                       unsigned int* __restrict__ hist,
                                       unsigned long long* __restrict__ len_sum);

// ---- Host wrapper --------------------------------------------------------
// sv_call_gpu: run the whole GPU pipeline.
//   Allocates device buffers, uploads reads + reference, launches the kernel
//   (refine + atomic vote), copies the histogram back, then merges it into SV
//   calls on the host (histogram_to_calls, shared with the CPU reference).
//
//   d           : the loaded problem (reference + reads)
//   min_support : noise floor for emitting a call (same as the CPU)
//   hist        : out-param, the GPU per-bin support histogram (for verification)
//   len_sum     : out-param, the GPU per-bin deletion-length sum
//   kernel_ms   : out-param, milliseconds in the kernel (CUDA events, not copies)
//   Returns the SV calls (identical to the CPU's, by construction).
std::vector<SvCall> sv_call_gpu(const SvDataset& d, unsigned int min_support,
                                std::vector<unsigned int>& hist,
                                std::vector<unsigned long long>& len_sum,
                                float* kernel_ms);
