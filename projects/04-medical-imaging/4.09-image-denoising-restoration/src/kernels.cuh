// ===========================================================================
// src/kernels.cuh  --  GPU denoising interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 4.9 : Image Denoising & Restoration  (Non-Local Means)
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls denoise_gpu(); kernels.cu
//   implements both the host wrapper and the device kernel. Included only by
//   .cu translation units (it declares a __global__ kernel, so the plain C++
//   compiler must never see it -- that is why the CPU reference and the shared
//   NLM math live in separate pure-C++ headers).
//
// THE BIG IDEA -- NLM as a per-output-pixel GATHER
//   Non-Local Means replaces each pixel with a patch-similarity-weighted average
//   of pixels in a search window around it (the math is in nlm_core.h). Every
//   output pixel is INDEPENDENT: it only reads the noisy input and writes its own
//   result -- no output pixel depends on another. So we map the 2-D image onto a
//   2-D thread grid and give ONE THREAD ONE OUTPUT PIXEL:
//
//       thread (blockIdx, threadIdx) -> pixel (col, row)
//       col = blockIdx.x * blockDim.x + threadIdx.x
//       row = blockIdx.y * blockDim.y + threadIdx.y
//
//   Each thread runs the *same* nlm_pixel() the CPU reference runs, so GPU and
//   CPU agree to a very tight tolerance. There are no atomics and no shared
//   memory in this teaching version: the per-pixel reduction (Σ weights, Σ
//   weight*value) is entirely local to the thread's registers. THEORY.md
//   "GPU mapping" discusses the obvious next optimisation -- staging the input
//   tile + its halo into shared memory so neighbouring threads reuse loads.
//
//   The noisy image is uploaded ONCE to global memory and read by every thread;
//   because many threads read overlapping regions, the L1/L2 caches do a lot of
//   work for us even without explicit tiling.
//
// READ THIS AFTER: nlm_core.h, util/cuda_check.cuh, util/timer.cuh. Then kernels.cu.
// ===========================================================================
#pragma once

#include "reference_cpu.h"   // Image, NlmParams  (pure C++, safe inside a .cu)

// ---- Device kernel -------------------------------------------------------
// nlm_kernel: thread (col,row) denoises output pixel (row,col).
//   in     : device pointer to the [height*width] noisy image (row-major, [0,1])
//   params : all NLM parameters, passed BY VALUE so each thread gets its own copy
//            in registers (the struct is tiny -- 6 scalars) with no global reads
//   out    : device pointer to the [height*width] denoised image
// The kernel body simply calls nlm_pixel() -- the exact function the CPU uses.
__global__ void nlm_kernel(const float* __restrict__ in, NlmParams params,
                           float* __restrict__ out);

// ---- Host wrapper --------------------------------------------------------
// denoise_gpu: the host-callable "do the whole GPU denoise" function.
//   Uploads the noisy image, launches the 2-D grid of nlm_kernel, copies the
//   denoised image back, and reports the measured KERNEL time (CUDA events) via
//   *kernel_ms (kernel only -- host<->device copies are excluded so the timing
//   reflects compute, the teaching point). main.cu calls exactly this; every
//   cudaMalloc/cudaMemcpy is hidden and error-checked inside.
//     in        : host noisy image
//     params    : NLM parameters (must match what the CPU reference used)
//     out       : host denoised image, resized to match `in` (output parameter)
//     kernel_ms : out-param, milliseconds spent in the kernel itself (not copies)
void denoise_gpu(const Image& in, const NlmParams& params, Image& out, float* kernel_ms);
