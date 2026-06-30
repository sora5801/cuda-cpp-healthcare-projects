// ===========================================================================
// src/kernels.cuh  --  GPU 3D-convolution segmentation interface
// ---------------------------------------------------------------------------
// Project 4.7 : Medical Image Segmentation (Deep Learning)   [REDUCED SCOPE]
//
// THE BIG IDEA (pattern: 3D STENCIL / GATHER, one thread per output voxel)
//   A convolution layer turns an input volume into an output volume; each
//   output voxel is an independent dot product of a tiny weight stencil with a
//   3x3x3 neighbourhood. We assign ONE GPU THREAD PER OUTPUT VOXEL: thread
//   (blockIdx, threadIdx) -> flat voxel index i; it gathers its neighbourhood
//   from global memory and reduces it against the filter. This is exactly the
//   3D-convolution workload cuDNN accelerates inside nnU-Net / MONAI.
//
//   The network weights are TINY (a few hundred floats) and read by EVERY
//   thread but never change during a launch, so they live in __constant__
//   memory (defined in kernels.cu): the constant cache broadcasts a weight to a
//   whole warp in one transaction -- ideal for shared, read-only filter taps.
//
//   Two kernels run back-to-back, mirroring the CPU reference:
//     seg_layer1_kernel : 1ch input  -> C_HID hidden maps (conv + ReLU)
//     seg_layer2_kernel : C_HID maps -> per-voxel argmax label + lesion logit
//   Both call the SHARED conv3x3x3_at() from reference_cpu.h, so GPU and CPU run
//   identical math (PATTERNS.md §2).
//
//   kernels.cu defines the kernels + the host wrapper. main.cu calls
//   segment_gpu(). The per-voxel physics is in reference_cpu.h.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // Volume, SegNet, conv3x3x3_at, relu (pure C++; safe in .cu)

// Threads per block for the voxel-parallel kernels. 256 is a solid default on
// sm_75..sm_89: a multiple of the 32-lane warp, 8 warps to hide global-memory
// latency, and small enough to keep many blocks resident (good occupancy).
static constexpr int SEG_BLOCK = 256;

// ---- Device kernels (documented fully in kernels.cu) ---------------------

// Layer 1: conv (1 input channel) + ReLU. One thread per output voxel.
//   d_in     : [D*H*W]        input intensity volume (device)
//   D,H,W    : volume dims
//   d_hidden : [C_HID*D*H*W]  output hidden feature maps (device)
// Weights (w1,b1) are read from __constant__ memory set by the host wrapper.
__global__ void seg_layer1_kernel(const float* __restrict__ d_in, int D, int H, int W,
                                  float* __restrict__ d_hidden);

// Layer 2: conv (C_HID input channels) + per-voxel argmax. One thread per voxel.
//   d_hidden : [C_HID*D*H*W]  hidden maps from layer 1 (device)
//   d_label  : [D*H*W] int    output class label 0/1 (device)
//   d_logit1 : [D*H*W] float  output lesion-class logit (for tolerance check)
// Weights (w2,b2) are read from __constant__ memory.
__global__ void seg_layer2_kernel(const float* __restrict__ d_hidden, int D, int H, int W,
                                  int* __restrict__ d_label, float* __restrict__ d_logit1);

// ---- Host wrapper --------------------------------------------------------
// segment_gpu: upload weights (to constant memory) + the volume, launch both
//   layer kernels, copy the label map and lesion logits back, and report the
//   summed kernel time (CUDA events) via *kernel_ms.
//     vol      : input volume (host)
//     net      : fixed weights (host) -> copied to constant memory
//     label    : [D*H*W] output integer mask (resized here)
//     logit1   : [D*H*W] output lesion logits (resized here)
//     kernel_ms: out-param, total ms in the two kernels (not the H2D/D2H copies)
void segment_gpu(const Volume& vol, const SegNet& net,
                 std::vector<int>& label, std::vector<float>& logit1,
                 float* kernel_ms);
