// ===========================================================================
// src/kernels.cuh  --  GPU vesselness interface
// ---------------------------------------------------------------------------
// Project 4.26 : Vessel Segmentation & Centerline Extraction
//
// THE BIG IDEA (the "map" pattern)
//   The Frangi vesselness of a voxel depends only on its local 3x3x3
//   neighbourhood (to form the Hessian by finite differences). Every voxel's
//   score is therefore INDEPENDENT -- a textbook "map": one GPU thread computes
//   one voxel. We launch a 3-D grid of 3-D thread blocks over the volume; thread
//   (x,y,z) owns output voxel (x,y,z). No atomics, no cross-thread communication.
//
//   The per-voxel math (Hessian eigenvalues + Frangi score) is the SHARED
//   frangi.h, so the kernel reproduces the CPU reference to ~1e-9.
//
//   Note: the Gaussian pre-smoothing is done ONCE on the host (it is a separable
//   convolution both paths share); the kernel takes the already-smoothed volume.
//   Smoothing on the GPU too is a natural exercise (THEORY / README).
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, frangi.h, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // Volume, FrangiParams (pure C++, safe in a .cu)

// ---- Device kernel -------------------------------------------------------
// vesselness_kernel: thread (x,y,z) computes the Frangi score of its voxel.
//   nx,ny,nz : volume dimensions (guard the ragged edge blocks)
//   s        : device pointer to the SMOOTHED volume (nx*ny*nz floats)
//   fp       : Frangi parameters (passed by value -> in each thread's registers)
//   vness    : device pointer to the nx*ny*nz output scores
__global__ void vesselness_kernel(int nx, int ny, int nz,
                                  const float* __restrict__ s,
                                  FrangiParams fp,
                                  float* __restrict__ vness);

// ---- Host wrapper --------------------------------------------------------
// vesselness_gpu: run the whole GPU vesselness computation.
//   Takes the host-side SMOOTHED volume, uploads it, launches the kernel over a
//   3-D grid, downloads the score field, and reports the measured KERNEL time
//   (CUDA events) via *kernel_ms. main.cu calls exactly this.
//     smoothed : host input (already Gaussian-smoothed on the host)
//     fp       : Frangi parameters
//     vness    : host output, resized to nx*ny*nz (output parameter)
//     kernel_ms: out-param, milliseconds spent in the kernel (not copies)
void vesselness_gpu(const Volume& smoothed, const FrangiParams& fp,
                    std::vector<float>& vness, float* kernel_ms);
