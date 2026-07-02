// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 7.3 : Clinical NLP over Notes & Records
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls the single host wrapper
//   declared here; kernels.cu implements it (the device kernels + the cuBLAS
//   batched-GEMM calls live there). Included only by .cu translation units (it
//   would name __global__ kernels internally), so the plain host compiler never
//   sees it -- that is why the CPU reference uses a separate pure-C++ header
//   (reference_cpu.h).
//
// THE GPU JOBS (and the pattern each uses -- see ../THEORY.md, PATTERNS.md)
//
//   A transformer self-attention encoder block is GEMM-DOMINATED (the catalog's
//   own words). We split it the way a real implementation does:
//
//   1. Q = X Wq, K = X Wk, V = X Wv     -- three dense matmuls over ALL tokens.
//        X is [(B*S) x D], each W is [D x D]. One cuBLAS DGEMM per projection
//        (PATTERNS.md §1: "dense linear algebra -> use cuBLAS"). DGEMM is the
//        single most optimized GPU routine in existence; kernels.cu explains
//        what hand-rolling it would take (shared-memory tiling, register
//        blocking, bank-conflict-free loads) and why we don't.
//
//   2. scores = Q_h K_hᵀ / sqrt(dh)     -- B*H independent [S x S] matmuls.
//        We use cublasDgemmStridedBatched: ONE call does all B*H head-matmuls,
//        striding through the Q/K buffers. This is the O(n²) attention cost the
//        deep-dive names -- exactly the multi-head GEMM tensor cores accelerate.
//
//   3. softmax(scores) with PAD masking  -- a hand-written kernel. This is the
//        one step that is NOT a GEMM. One block per (note, head, query row); the
//        block cooperatively finds the row max, exponentiates (stable form from
//        attn_core.h), sums, and normalizes. The "reduction within a block"
//        pattern.
//
//   4. O_h = A_h V_h                     -- another B*H batched DGEMM, then the
//        heads are already laid out contiguously so O is just [(B*S) x D].
//
//   Everything numeric that BOTH sides must agree on lives in attn_core.h, so
//   the GPU output matches the CPU reference to near machine precision (main.cu).
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, attn_core.h.
// THEN READ kernels.cu.
// ===========================================================================
#pragma once

#include "reference_cpu.h"   // NoteBatch, AttnResult (shared POD shapes)

// ---------------------------------------------------------------------------
// GpuAttnTimings: a small breakdown of where GPU time went, for the stderr
//   teaching report. All in milliseconds, measured with CUDA events.
// ---------------------------------------------------------------------------
struct GpuAttnTimings {
    float proj_ms  = 0.0f;   // the three projection DGEMMs (Q,K,V)
    float score_ms = 0.0f;   // the batched QKᵀ DGEMM
    float soft_ms  = 0.0f;   // the softmax + masking kernel
    float ctx_ms   = 0.0f;   // the batched A·V DGEMM
    float total_ms = 0.0f;   // sum of the above (compute only, excludes copies)
};

// ---------------------------------------------------------------------------
// gpu_attention: run ONE self-attention encoder block over the whole batch on
//   the GPU, filling `res` with the SAME shapes the CPU reference produces
//   (res.out [B*S*D], res.weights [B*H*S*S]) so main.cu can verify entrywise.
//     nb  : the loaded batch (token ids, embeddings, dims)
//     res : output (resized inside), contextualized outputs + attention probs
//     t   : out-param timing breakdown (CUDA-event measured)
//   The function owns all device memory + the cuBLAS handle; main.cu just sees
//   host structs in and out.
void gpu_attention(const NoteBatch& nb, AttnResult& res, GpuAttnTimings* t);
