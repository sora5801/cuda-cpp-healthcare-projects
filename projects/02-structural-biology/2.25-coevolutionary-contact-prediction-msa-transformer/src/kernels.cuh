// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 2.25 : Coevolutionary Contact Prediction & MSA Transformer
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls coevolution_mi_gpu(), which
//   computes the raw L x L Mutual-Information matrix on the device. kernels.cu
//   implements both the host wrapper and the __global__ kernel. This header is
//   included only by .cu translation units (it declares a __global__ kernel, so
//   the plain C++ compiler must never see it -- that is why the CPU reference
//   lives in the separate pure-C++ reference_cpu.h).
//
// THE BIG IDEA  ("score all L x L column pairs, each independent")
//   The coevolution matrix has one entry per ordered column pair (i, j). Every
//   entry is an INDEPENDENT reduction over the N sequences, so we assign ONE GPU
//   THREAD PER PAIR. A 2-D block/grid maps thread (i, j) -> matrix cell (i, j):
//   thread i = blockIdx.x*blockDim.x + threadIdx.x (column index along x),
//   thread j = blockIdx.y*blockDim.y + threadIdx.y (column index along y). Each
//   thread independently builds a small Q*Q joint-count table in its REGISTERS/
//   local memory, then calls the shared cv_mi_from_counts (coevolution.h) -- the
//   exact same function the CPU reference uses -- so the result matches bit-for-
//   bit up to a 1-ulp log() difference. This is the "many independent jobs"
//   pattern (PATTERNS.md section 1; exemplars 1.12 Tanimoto, 12.01 spectral).
//
//   We only compute the upper triangle (i < j) per thread and write BOTH (i,j)
//   and (j,i) since MI is symmetric; threads with i >= j simply return. The
//   diagonal stays 0 (initialized on the host).
//
//   The downstream APC correction is a cheap O(L^2) host reduction done in
//   main.cu from this matrix (same as project 11.09 finishing its reduction on
//   the host), so it is shared with the CPU path and needs no kernel.
//
// READ THIS AFTER: coevolution.h, util/cuda_check.cuh, util/timer.cuh. Then
// read kernels.cu, then main.cu.
// ===========================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "reference_cpu.h"   // Msa (the input the GPU consumes)

// ---- Device kernel -------------------------------------------------------
// mi_pairs_kernel: each thread computes MI for one column pair (i, j), i < j.
//   tokens : device pointer to the [N*L] MSA token matrix (row-major, uint8).
//   single : device pointer to the [L*CV_Q] precomputed column marginals
//            (single[c*CV_Q + a] = #sequences with token a in column c). We
//            precompute marginals on the host and upload them so each thread
//            only has to build the JOINT table -- the marginals are reused by
//            every pair, so recomputing them per thread would be wasteful.
//   N, L   : MSA dimensions.
//   mi     : device pointer to the [L*L] output matrix (double). Thread (i,j)
//            writes mi[i*L+j] and mi[j*L+i]; the diagonal is left as the host
//            initialized it (0).
// Launch config and thread->data mapping are documented at the definition in
// kernels.cu.
__global__ void mi_pairs_kernel(const uint8_t* __restrict__ tokens,
                                const uint32_t* __restrict__ single,
                                int N, int L,
                                double* __restrict__ mi);

// ---- Host wrapper --------------------------------------------------------
// coevolution_mi_gpu: the host-callable "compute the raw MI matrix on the GPU".
//   Builds the column marginals on the host, uploads tokens + marginals,
//   launches mi_pairs_kernel over a 2-D grid of column pairs, copies the L x L
//   MI matrix back, and reports the measured KERNEL time (CUDA events) via
//   *kernel_ms. All CUDA bookkeeping is hidden here; main.cu just calls this and
//   then runs the shared apc_correct() on the result.
//
//   msa       : the input alignment (host).
//   mi        : host output, resized to L*L (output parameter), row-major.
//   kernel_ms : out-param, milliseconds spent in the kernel itself (not copies).
void coevolution_mi_gpu(const Msa& msa, std::vector<double>& mi, float* kernel_ms);
