// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for one self-attention head
// ---------------------------------------------------------------------------
// Project 2.1 : Protein Structure Prediction Inference (AlphaFold-class)
//               REDUCED-SCOPE TEACHING VERSION.
//
// THE BIG IDEA  (the GPU mapping this project teaches)
//   Self-attention computes, for every residue i,
//       Out[i] = sum_j softmax_j( (Q[i].K[j]) / sqrt(d) ) * V[j].
//   The L output rows are INDEPENDENT of each other, so we give each query
//   residue its OWN THREAD BLOCK. Inside a block, the threads cooperate on row i:
//     * each thread computes a slice of the L scores Q[i].K[j],
//     * a SHARED-MEMORY reduction finds the row max and the softmax denominator
//       (the parallel pattern for "max then sum over a row"), and
//     * the threads jointly accumulate the weighted value sum into Out[i].
//   This "one block per output row, cooperate via shared memory" layout is the
//   schoolbook attention kernel and the conceptual ancestor of FlashAttention.
//
//   This header is included only by .cu units (it declares a __global__). The
//   pure-C++ data model lives in reference_cpu.h; the shared math in
//   attention_core.h. main.cu calls attention_gpu().
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, reference_cpu.h,
// attention_core.h. Then read kernels.cu. Science/GPU-mapping is in ../THEORY.md.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // AttentionProblem, D_MODEL (pure C++, safe in .cu)

// ---- Device kernel -------------------------------------------------------
// attention_kernel: one self-attention output row per BLOCK.
//   Block b handles query residue i = b. Its threads share the work of scoring
//   residue i against all L residues, the softmax reduction, and the weighted
//   value sum. Grid = L blocks; block = THREADS_PER_BLOCK threads (chosen in
//   kernels.cu). Uses dynamic shared memory for the per-row scores + reduction.
//     q,k,v : [L * D_MODEL] row-major device matrices
//     L     : number of residues (== gridDim.x)
//     out   : [L * D_MODEL] device output, row-major
__global__ void attention_kernel(const float* __restrict__ q,
                                 const float* __restrict__ k,
                                 const float* __restrict__ v,
                                 int L,
                                 float* __restrict__ out);

// ---- Host wrapper --------------------------------------------------------
// attention_gpu: the host-callable "do the whole GPU attention" function.
//   Allocates device buffers, copies Q/K/V H2D, launches attention_kernel,
//   copies Out D2H, and reports the measured KERNEL time (CUDA events) via
//   *kernel_ms. main.cu calls exactly this; all CUDA bookkeeping is hidden here.
//
//   prob      : the loaded problem (Q, K, V and dimensions L, d)
//   out       : host output, resized to L*D_MODEL (output parameter)
//   kernel_ms : out-param, milliseconds spent in the kernel itself (not copies)
void attention_gpu(const AttentionProblem& prob, std::vector<float>& out,
                   float* kernel_ms);
