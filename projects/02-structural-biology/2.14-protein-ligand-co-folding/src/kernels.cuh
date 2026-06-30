// ===========================================================================
// src/kernels.cuh  --  GPU co-folding interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 2.14 : Protein-Ligand Co-Folding (reduced-scope teaching version)
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls simulate_gpu(); kernels.cu
//   implements the host time loop plus the device attention kernel. Included
//   only by .cu units (it declares a __global__), so the plain C++ compiler
//   never sees it -- that is why the CPU reference lives in a separate pure-C++
//   header (reference_cpu.h) and the shared math lives in cofold.h.
//
// THE PATTERN (per-step ATTENTION, the real co-folding bottleneck)
//   Co-folding is a reverse-diffusion loop; EACH step is a self-attention pass
//   over the joint protein+ligand token sequence (deep-dive: "50-200 denoising
//   steps, each requiring a full attention forward pass"). We map it as:
//     * the HOST runs the T-step loop, launching one kernel per step and
//       PING-PONGING two position buffers (read frozen state, write next, swap)
//       -- the same double-buffer discipline as the stencil flagship 14.02;
//     * the KERNEL assigns ONE BLOCK PER QUERY TOKEN. The block's threads
//       cooperatively stream over all key tokens, doing a two-pass online
//       softmax (max, then exp-weighted target sum) with a shared-memory
//       reduction. This is the shape of FlashAttention -- parallel over the key
//       dimension, O(1) extra storage per query -- taught at toy scale.
//   The per-token math itself is denoise_token() from cofold.h, so the GPU
//   reproduces the CPU result.
//
// READ THIS AFTER: cofold.h, util/cuda_check.cuh, util/timer.cuh. Then kernels.cu.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // Complex, CofoldParams (pure C++, safe in a .cu)

// ---- Device kernel -------------------------------------------------------
// attention_step_kernel: advance EVERY token one denoising step.
//   Launch config: grid = n_tokens blocks, block = THREADS_PER_TOKEN threads.
//   Block b updates query token b; its threads split the key loop and reduce in
//   shared memory. Reads `pos` (frozen), `target`, `types`; writes `pos_next`.
//   Passing CofoldParams by value puts the small schedule in constant-arg space.
__global__ void attention_step_kernel(CofoldParams P,
                                       const double* __restrict__ pos,
                                       const double* __restrict__ target,
                                       const int* __restrict__ types,
                                       double* __restrict__ pos_next);

// ---- Host wrapper --------------------------------------------------------
// simulate_gpu: run the whole reverse diffusion on the GPU.
//   Allocates device buffers, copies the initial positions / target / types up,
//   runs the T-step ping-pong loop launching attention_step_kernel each step,
//   copies the final positions back, and reports the measured KERNEL-loop time
//   (CUDA events) via *kernel_ms.
//
//   C        : the complex (provides P, target, types).
//   pos      : in = initial noised positions; out = final predicted positions
//              (resized to n_tokens * D_POS by the caller, updated in place).
//   kernel_ms: out-param, milliseconds spent in the denoising loop (not copies).
void simulate_gpu(const Complex& C, std::vector<double>& pos, float* kernel_ms);
