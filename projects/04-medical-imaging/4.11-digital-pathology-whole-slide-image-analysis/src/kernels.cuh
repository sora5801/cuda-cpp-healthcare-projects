// ===========================================================================
// src/kernels.cuh  --  GPU attention-MIL interface (the teaching idea)
// ---------------------------------------------------------------------------
// Project 4.11 : Digital Pathology / Whole-Slide Image Analysis
//
// THE BIG IDEA  (pattern: PER-TILE PROJECTION + SOFTMAX + ATOMIC POOL)
//   A slide is a BAG of N tile feature vectors (produced upstream by a frozen
//   CNN/ViT -- NOT reimplemented here; see THEORY.md). Attention-MIL turns that
//   bag into one slide prediction in four steps, mapped to the GPU like this:
//
//     1. LOGITS  kernel: one thread per TILE computes that tile's attention logit
//                e_i = w . (tanh(V h_i) * sigmoid(U h_i))   -- embarrassingly
//                parallel, each tile independent (the wsi.h math).      [per-tile]
//     2. SOFTMAX (host): a tiny deterministic reduction over the N logits ->
//                attention weights a_i (numerically-stable, subtract-max). N is
//                small and this keeps stdout bit-reproducible.          [host]
//     3. POOL    kernel: one thread per TILE atomically adds its weighted feature
//                a_i * h_i into D shared accumulators, IN FIXED-POINT integers so
//                the atomic sum is order-independent and matches the CPU exactly.
//                                                             [atomic reduction]
//     4. CLASSIFY(host): s = w_c . z + b_c, probability = sigmoid(s).   [host]
//
//   Steps 1 and 3 are the GPU kernels (below); steps 2 and 4 are tiny host
//   reductions REUSED from the CPU reference so the CPU and GPU produce identical
//   numbers. This split -- heavy parallel work on the device, a small exact
//   reduction on the host -- is the same shape as flagship 11.09 (k-means).
//
//   kernels.cu implements the kernels + the host wrapper mil_forward_gpu().
//   main.cu calls mil_forward_gpu() and compares it to mil_forward_cpu().
//
// READ THIS AFTER: wsi.h, util/cuda_check.cuh, util/timer.cuh, reference_cpu.h.
// ===========================================================================
#pragma once

#include "reference_cpu.h"   // SlideBag, MilResult, AttnParams (pure C++, safe in .cu)

// ---- Kernel 1: per-tile attention logits ---------------------------------
// LOGITS: logits[i] = wsi_attention_logit(features + i*FEAT_DIM, params).
//   grid  : ceil(N / block) blocks
//   block : 256 threads (a good occupancy default on sm_75..sm_89)
//   thread (blockIdx.x, threadIdx.x) -> tile index i = bx*blockDim.x + tx.
// The frozen model parameters live in CONSTANT memory (see kernels.cu): every
// thread reads the same weights, which is exactly what constant memory's
// broadcast cache is for.
__global__ void attention_logits_kernel(const double* __restrict__ features,
                                        int N,
                                        double* __restrict__ logits);

// ---- Kernel 2: attention-weighted fixed-point pooling --------------------
// POOL: for each tile i, atomically add its fixed-point weighted feature
//   wsi_quantize(attn[i] * features[i][d]) into fixed_embed[d]. Integer atomic
//   adds commute -> deterministic and CPU-matching (wsi.h explains the trick).
//   Same thread-to-tile mapping as kernel 1.
__global__ void attention_pool_kernel(const double* __restrict__ features,
                                      const double* __restrict__ attn,
                                      int N,
                                      unsigned long long* __restrict__ fixed_embed);

// ---- Host wrapper --------------------------------------------------------
// mil_forward_gpu: run the whole attention-MIL forward pass on the GPU.
//   Uploads the bag, runs the LOGITS kernel, does the softmax on the host,
//   uploads the weights, runs the POOL kernel, brings back the fixed-point
//   embedding, then classifies on the host. Fills a MilResult identical to the
//   CPU one. *kernel_ms returns the summed GPU kernel time (CUDA events).
//     bag       : input slide bag (N tiles x FEAT_DIM)
//     p         : frozen attention model (copied into constant memory)
//     kernel_ms : out-param, milliseconds spent in the two kernels
MilResult mil_forward_gpu(const SlideBag& bag, const AttnParams& p, float* kernel_ms);
