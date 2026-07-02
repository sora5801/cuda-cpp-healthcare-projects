// ===========================================================================
// src/kernels.cuh  --  GPU forward-pass interface (declarations + the big idea)
// ---------------------------------------------------------------------------
// Project 7.18 : Retinal Fundus AI Screening
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls forward_gpu(); kernels.cu
//   implements the host orchestration plus the device kernels. Included only by
//   .cu translation units (it contains __global__ declarations, so the plain
//   C++ compiler must never see it -- that is why the CPU reference and the
//   shared math live in pure-C++/HD headers).
//
// THE BIG IDEA (the sixth flagship pattern, in 2-D: SHARED-MEMORY TILING)
//   A CNN's cost is its convolution layers. Each output pixel of each feature
//   map reads a K x K neighbourhood across every input channel. Adjacent output
//   pixels overlap almost entirely, so the naive "one thread per output pixel,
//   re-read the window from global memory" re-reads each input pixel ~K*K times.
//   The optimized conv kernel TILES a (blockDim.y+2*halo) x (blockDim.x+2*halo)
//   patch of every input channel into SHARED MEMORY once; then each thread reads
//   its window from that fast on-chip tile. This is the 2-D version of the 1-D
//   tiling lesson in flagship 7.10. The pooling and classifier kernels are
//   simpler element-wise / reduction kernels.
//
//   Thread-to-data mapping (conv/pool kernels): a 2-D block of threads covers a
//   TILE of one output feature map; thread (tx,ty) in block (bx,by) owns output
//   pixel (x = bx*TILE_W + tx, y = by*TILE_H + ty) of the feature map selected
//   by blockIdx.z. See kernels.cu for the exact tile-loading dance.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, cnn_core.h. Then kernels.cu.
// ===========================================================================
#pragma once

#include "reference_cpu.h"   // FundusImage, CnnModel, ForwardResult (pure C++)

// ---- Tile geometry for the convolution kernel ----------------------------
// A 16x16 block of threads computes a 16x16 output tile. With halo=1 the shared
// tile is 18x18 per input channel. 256 threads/block is a solid occupancy
// default on sm_75..sm_89 (matches the 1-D flagship's 256-thread choice).
static constexpr int TILE = 16;

// ---------------------------------------------------------------------------
// forward_gpu: run the ENTIRE forward pass on the GPU and fill `out`.
//   Mirrors forward_cpu() exactly (same math via cnn_core.h) so main.cu can
//   verify the two agree within tolerance.
//     img   : the loaded fundus image (host); uploaded internally
//     model : the fixed weights (host); uploaded internally
//     out   : ForwardResult filled with logits/probs/pred_grade/cam
//     conv_ms : out-param, milliseconds spent in the two conv+relu kernels
//               (the dominant cost) measured with CUDA events. Reported to
//               stderr as a teaching artifact (never a benchmark claim).
// All device allocation, H2D/D2H copies, and error checking are hidden here.
// ---------------------------------------------------------------------------
void forward_gpu(const FundusImage& img, const CnnModel& model,
                 ForwardResult& out, float* conv_ms);
