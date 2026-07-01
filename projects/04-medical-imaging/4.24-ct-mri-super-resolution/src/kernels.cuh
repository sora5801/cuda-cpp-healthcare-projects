// ===========================================================================
// src/kernels.cuh  --  GPU super-resolution interface (the gather pattern)
// ---------------------------------------------------------------------------
// Project 4.24 : CT/MRI Super-Resolution   (reduced-scope teaching version)
//
// THE BIG IDEA (imaging GATHER pattern; PATTERNS.md §1, exemplar 4.01)
//   Super-resolution turns one LR image into an R x larger HR image. Every HR
//   output pixel is computed INDEPENDENTLY: it looks up which LR cell + sub-pixel
//   phase it belongs to, GATHERS the surrounding 3x3 LR neighbourhood, and runs
//   a tiny two-layer conv network (feature conv + ReLU, then a sub-pixel conv
//   selected by pixel-shuffle). No two output pixels talk to each other -> we map
//   ONE GPU THREAD TO ONE HR OUTPUT PIXEL and let thousands run at once.
//
//   The per-pixel arithmetic is defined ONCE in sr_core.h (sr_hr_pixel), which
//   is decorated __host__ __device__ so the CPU reference and this kernel run the
//   identical math -> exact CPU/GPU agreement (PATTERNS.md §2).
//
//   The network WEIGHTS are read by every thread but never change during the
//   launch, so we stage them in CONSTANT memory (broadcast cache) -- see
//   kernels.cu. main.cu calls super_resolve_gpu().
//
// READ THIS AFTER: sr_core.h, reference_cpu.h, util/cuda_check.cuh, util/timer.cuh.
// ===========================================================================
#pragma once

#include "reference_cpu.h"   // Image (pure C++, safe to include in a .cu)
#include "sr_core.h"         // SrWeights, SR_SCALE (shared with the CPU side)

// Threads per block along each axis: a 16x16 = 256-thread tile is a solid
// occupancy default on sm_75..sm_89 for a 2-D output map (matches 4.01's choice).
static constexpr int SR_BLOCK_X = 16;
static constexpr int SR_BLOCK_Y = 16;

// ---------------------------------------------------------------------------
// super_resolve_gpu: host wrapper around the SR kernel.
//   Uploads the LR image + weights, launches one thread per HR pixel, copies the
//   HR result back. Fills *kernel_ms with the GPU-measured kernel time.
//   Params:
//     lr        : the low-res input image (host).
//     W         : the network weights (host copy; uploaded to constant memory).
//     scale     : upscale factor R (must equal SR_SCALE; asserted in the .cu).
//     out       : receives the super-resolved HR image (size lr.w*R x lr.h*R).
//     kernel_ms : out-param, GPU kernel time in milliseconds (teaching artifact).
//   The output is deterministic: sr_hr_pixel does the same ops in the same order
//   as the CPU reference, so out matches super_resolve_cpu() to (near) the bit.
// ---------------------------------------------------------------------------
void super_resolve_gpu(const Image& lr, const SrWeights& W, int scale,
                       Image& out, float* kernel_ms);
