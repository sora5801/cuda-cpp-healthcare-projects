// ===========================================================================
// src/kernels.cuh  --  GPU self-attention interface (declarations + the idea)
// ---------------------------------------------------------------------------
// Project 3.18 : Protein Language Model Inference
//
// THE BIG IDEA (GPU pattern: ONE BLOCK PER (HEAD, QUERY ROW), thread per key)
//   Multi-head self-attention is a stack of dense matrix products and per-row
//   softmaxes. The most teachable parallel decomposition keeps each attention
//   ROW independent:
//
//     * We launch a 2-D grid of blocks: grid.x = L query residues, grid.y = H
//       heads. Block (i, h) computes attention row i for head h.
//     * Inside a block, the THREADS cooperate over the L keys: thread `t`
//       handles key columns j = t, t+blockDim, ...  It (a) computes the scaled
//       logit q_i·k_j, (b) the block does a stable softmax over the L logits in
//       SHARED MEMORY (parallel max-reduction then sum-reduction), and (c) the
//       threads accumulate the value blend  sum_j A[i,j]·V_h[j]  into this
//       residue's d_head output slice.
//     * A second small kernel applies the output projection Wo to the
//       concatenated head outputs and computes each residue's output norm.
//
//   This mirrors the CPU reference exactly (same attention_math.h helpers), and
//   it is the same shape FlashAttention optimizes -- we keep the whole softmax
//   row resident instead of streaming it, which is fine at teaching scale (small
//   L) and far easier to read. See ../THEORY.md "GPU mapping".
//
//   kernels.cu implements everything; main.cu calls attention_gpu().
//
//   Included only by .cu units (it declares __global__ kernels). The CPU
//   reference uses the pure-C++ reference_cpu.h instead.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // AttnConfig, AttnResult (pure C++, safe in .cu)

// Threads per attention-row block. 128 gives good occupancy on sm_75..sm_89 and
// comfortably covers the small L of the teaching sample (one thread strides over
// the L keys). The softmax shared-memory reduction assumes a power-of-two count.
static constexpr int ATTN_THREADS = 128;

// ---- Device kernels (defined in kernels.cu) ------------------------------

// attention_rows_kernel: block (blockIdx.x = query i, blockIdx.y = head h)
//   computes attention row i of head h and accumulates that head's output slice
//   for residue i into Z. Also writes head 0's attention row into `attn` for
//   reporting. Q/K/V are recomputed on the fly from X via proj_one().
//     X    : [L*D] input embeddings (device)
//     Z    : [L*D] concatenated head outputs (device, written)
//     attn : [L*L] head-0 attention map (device, written)
__global__ void attention_rows_kernel(const float* __restrict__ X, AttnConfig cfg,
                                      float* __restrict__ Z, float* __restrict__ attn);

// output_proj_kernel: one thread per output element Y[i,j] = (Z row i)·Wo[:,j].
//   The output norms are computed by a separate 1-thread-per-row kernel below so
//   each norm is a single deterministic sequential sum (no atomics), matching
//   the CPU bit-for-bit closely. This kernel only fills Y.
//     Z   : [L*D] (device)   Y : [L*D] (device, written)
__global__ void output_proj_kernel(const float* __restrict__ Z, AttnConfig cfg,
                                   float* __restrict__ Y);

// row_norm_kernel: one thread per residue i computes out_norm[i] = ||Y row i||.
//   Sequential per-row sum in one thread => deterministic, matches the CPU.
__global__ void row_norm_kernel(const float* __restrict__ Y, AttnConfig cfg,
                               float* __restrict__ out_norm);

// ---- Host wrapper --------------------------------------------------------
// attention_gpu: run the whole block forward pass on the GPU and fill `r`.
//   Uploads X, launches the three kernels, copies out/attn/out_norm back, and
//   computes top_attn on the host (a tiny argmax). Reports the summed KERNEL
//   time (CUDA events) via *kernel_ms. main.cu calls exactly this.
void attention_gpu(const std::vector<float>& X, const AttnConfig& cfg,
                   AttnResult& r, float* kernel_ms);
