// ===========================================================================
// src/kernels.cuh  --  GPU forward-pass interface (the teaching idea)
// ---------------------------------------------------------------------------
// Project 7.1 : Diagnostic Imaging Classifier   (REDUCED-SCOPE teaching version)
//
// THE BIG IDEA (pattern: INDEPENDENT per-output-pixel GATHER + CONSTANT weights)
//   A CNN forward pass over a batch is EMBARRASSINGLY PARALLEL: every output
//   pixel of every feature map of every image is an independent K*K dot product
//   (a "gather" that reads a small local window). So we assign ONE GPU THREAD
//   PER OUTPUT ELEMENT -- exactly the mapping the deep-dive means by "backbone
//   convolutions map directly onto tensor cores", written by hand so it is not a
//   black box (CLAUDE.md section 6).
//
//   The network weights are TINY (a few hundred floats) and READ BY EVERY
//   THREAD but never change during a launch -> we stage them in CONSTANT memory,
//   whose broadcast cache serves the same address to a whole warp in one shot
//   (same trick as the query in flagship 1.12). See kernels.cu.
//
//   Two kernels, matching the two arithmetic-heavy layers:
//     1. conv_pool_kernel : per (image, filter, pooled y, pooled x) thread,
//        does CONV+ReLU over the 2x2 pooling window and writes ONE pooled
//        feature. This is the dominant cost.
//     2. dense_kernel     : per (image, class) thread, dots the FLAT feature
//        vector with that class's dense weights -> one logit.
//   main.cu then applies softmax/argmax with the SAME shared helpers as the CPU.
//
//   The per-element math (conv_pixel, pool_pixel, dense_logit) is SHARED with
//   the CPU via reference_cpu.h's __host__ __device__ functions, so GPU==CPU
//   exactly (tolerance 0). kernels.cu only adds the thread indexing + memory.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, reference_cpu.h.
//                  Then read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // Weights, Dataset, geometry constants (pure C++)

// ---- Device kernels (declarations; defined in kernels.cu) ----------------

// conv_pool_kernel: one thread computes one POOLED feature.
//   d_images : [n * IMG_SIZE] device batch (row-major images)
//   n        : number of images (guards the ragged last block)
//   d_feat   : [n * FLAT] device output; the flattened pooled feature vectors
//   Conv filters + biases are read from __constant__ memory (see kernels.cu).
//   Thread -> data map: a flat index t in [0, n*FLAT) is unpacked into
//   (image i, filter f, pooled py, pooled px); the thread convolves the four
//   underlying conv pixels (via conv_pixel) and max-pools them (pool logic).
__global__ void conv_pool_kernel(const float* __restrict__ d_images, int n,
                                 float* __restrict__ d_feat);

// dense_kernel: one thread computes one class logit for one image.
//   d_feat   : [n * FLAT] pooled features from conv_pool_kernel
//   n        : number of images
//   d_logits : [n * NUM_CLS] device output logits
//   Dense weights + biases are read from __constant__ memory.
//   Thread -> data map: flat index t in [0, n*NUM_CLS) -> (image i, class c).
__global__ void dense_kernel(const float* __restrict__ d_feat, int n,
                             float* __restrict__ d_logits);

// ---- Host wrapper --------------------------------------------------------
// classify_gpu: the host-callable "do the whole forward pass on the GPU".
//   Uploads weights to constant memory and the image batch to global memory,
//   launches conv_pool_kernel then dense_kernel, copies logits back, and derives
//   the per-image argmax prediction with the shared helper. Reports the summed
//   KERNEL time (CUDA events) via *kernel_ms (transfers excluded).
//
//   w        : model weights (host)
//   d        : image batch (host)
//   logits   : host output, resized to n*NUM_CLS  (output parameter)
//   pred     : host output, resized to n (argmax class per image)
//   kernel_ms: out-param, total milliseconds in the two kernels
void classify_gpu(const Weights& w, const Dataset& d,
                  std::vector<float>& logits, std::vector<int>& pred,
                  float* kernel_ms);
