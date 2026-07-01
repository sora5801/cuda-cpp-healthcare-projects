// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 4.27 : Radiomics Feature Extraction
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls extract_features_gpu();
//   kernels.cu implements the host wrapper and the device kernel. Included only
//   by .cu translation units (it declares a __global__ kernel, so the plain C++
//   compiler must never see it -- that is why the CPU reference lives in the
//   separate pure-C++ reference_cpu.h).
//
// THE BIG IDEA (parallel histogram / atomic co-occurrence scatter)
//   The GLCM is a HISTOGRAM in disguise: for every ROI voxel and each of the 13
//   neighbour directions, we want to add 1 to matrix cell (gray_i, gray_j). The
//   voxels are independent, so we launch ONE THREAD PER VOXEL. Many voxels land
//   on the same (i,j) cell, so the increments COLLIDE -> we use atomicAdd. The
//   GLCM is tiny (Ng x Ng = 64 unsigned ints for Ng=8), so each block keeps its
//   own copy in SHARED MEMORY, all threads in the block atomic-add into that fast
//   on-chip copy, and one final atomic flush merges it into the global matrix.
//   This "privatized histogram" slashes contention on the global cells.
//
//   DETERMINISM: the accumulators are INTEGERS. Integer atomicAdd is associative
//   and commutative, so the summed counts are identical regardless of thread
//   order -- reproducible AND exactly equal to the serial CPU counts. (Float
//   atomics would NOT be; see docs/PATTERNS.md section 3.) The GLCM->features and
//   histogram->features reductions are then the SAME host code the CPU uses
//   (reference_cpu.cpp), so the whole feature vector matches.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, radiomics.h,
//                  reference_cpu.h. Then read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // Volume, Features, shared reductions (pure C++, safe in .cu)

// ---- Device kernels ------------------------------------------------------

// GLCM kernel: one thread per ROI-candidate voxel scatters co-occurrence pairs
// into a block-private shared-memory GLCM, then flushes it to the global matrix.
//   intensity, mask : device copies of the volume arrays ([nx*ny*nz])
//   nx,ny,nz        : grid dimensions
//   Ng              : gray levels; vmin,vmax : quantization range
//   glcm            : [Ng*Ng] global unsigned-int count matrix (atomic target)
__global__ void glcm_kernel(const float* __restrict__ intensity,
                            const uint8_t* __restrict__ mask,
                            int nx, int ny, int nz, int Ng,
                            float vmin, float vmax,
                            unsigned int* __restrict__ glcm);

// Histogram kernel: one thread per voxel; ROI voxels atomic-add into the ROI
// gray-level histogram (length Ng). Same parallel-histogram idea, one axis.
__global__ void histogram_kernel(const float* __restrict__ intensity,
                                 const uint8_t* __restrict__ mask,
                                 int nx, int ny, int nz, int Ng,
                                 float vmin, float vmax,
                                 unsigned int* __restrict__ hist);

// ---- Host wrapper --------------------------------------------------------
// extract_features_gpu: run the whole GPU pipeline -- upload the volume, launch
//   the histogram + GLCM kernels, copy the tiny matrices back, then reuse the
//   SHARED host reductions (first_order_from_histogram, haralick_from_glcm) so
//   the features match the CPU exactly. Returns the feature bundle; reports the
//   kernel time (CUDA events, GLCM + histogram launches) via *kernel_ms.
Features extract_features_gpu(const Volume& v, float* kernel_ms);
