// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for batched CTC basecalling
// ---------------------------------------------------------------------------
// Project 3.4 : Nanopore Basecalling  (REDUCED-SCOPE: CTC greedy decode)
//
// THE BIG IDEA  (PATTERNS.md sec 1: independent jobs, one thread per item)
//   Decoding read r's posterior matrix is INDEPENDENT of every other read, so
//   we give each read its OWN GPU thread. With n_reads reads and a block of B
//   threads, we launch ceil(n_reads / B) blocks; thread
//   i = blockIdx.x*blockDim.x + threadIdx.x decodes read i. A grid-stride loop
//   lets a modest grid cover an arbitrarily large batch (a real run has
//   millions of reads). This is the SAME "one independent job per thread"
//   pattern as 1.12 Tanimoto -- here the "job" is a CTC collapse instead of a
//   popcount similarity.
//
//   Each thread runs the SHARED ctc_greedy_decode() from ctc_core.h -- the very
//   same code the CPU reference loops -- so the GPU and CPU results are
//   bit-identical (verification tolerance == 0; see ../THEORY.md "verify").
//
//   This header is included only by .cu units (it declares a __global__). The
//   CPU reference uses the pure-C++ reference_cpu.h instead. main.cu calls
//   basecall_gpu().
//
// READ THIS AFTER: ctc_core.h, util/cuda_check.cuh, util/timer.cuh,
//   reference_cpu.h. Then read kernels.cu.
// ===========================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "reference_cpu.h"   // ReadSet, DecodedRead, CTC_NUM_CLASSES (pure C++)

// ---------------------------------------------------------------------------
// basecall_kernel: decode a BATCH of reads, one read per thread.
//   d_probs   : [total_steps * C] all reads' posteriors, row-major, contiguous
//               (read r's slice starts at d_offset[r]*C). __restrict__ promises
//               no aliasing so the compiler can keep loads in registers.
//   d_offset  : [n_reads+1] per-read start step (prefix sum of T); read r spans
//               steps [d_offset[r], d_offset[r+1]).
//   d_T       : [n_reads] time-step count per read (== d_offset[r+1]-d_offset[r]).
//   n_reads   : number of reads.
//   max_T     : padded stride of the output base buffer (max read length).
//   d_bases   : [n_reads * max_T] output; read r's bases land at row r
//               (d_bases[r*max_T ...]); only the first d_len[r] are valid.
//   d_len     : [n_reads] output decoded length per read.
//   d_checksum: [n_reads] output deterministic integer checksum per read.
// ---------------------------------------------------------------------------
__global__ void basecall_kernel(const float* __restrict__ d_probs,
                                const int*   __restrict__ d_offset,
                                const int*   __restrict__ d_T,
                                int n_reads, int max_T,
                                char*     __restrict__ d_bases,
                                int*      __restrict__ d_len,
                                uint32_t* __restrict__ d_checksum);

// ---------------------------------------------------------------------------
// basecall_gpu: host wrapper -- the whole GPU computation behind one call.
//   Uploads the ReadSet, launches basecall_kernel, copies the decoded bases /
//   lengths / checksums back, reconstructs DecodedRead strings on the host, and
//   reports the measured KERNEL time (CUDA events) via *kernel_ms.
//   rs        : the loaded batch of reads (input).
//   out       : resized to rs.n_reads; out[r] is read r's DecodedRead (output).
//   kernel_ms : out-param, GPU-measured kernel time in milliseconds (no copies).
// ---------------------------------------------------------------------------
void basecall_gpu(const ReadSet& rs, std::vector<DecodedRead>& out, float* kernel_ms);
