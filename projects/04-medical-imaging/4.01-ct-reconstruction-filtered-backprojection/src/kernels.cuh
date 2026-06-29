// ===========================================================================
// src/kernels.cuh  --  GPU backprojection interface
// ---------------------------------------------------------------------------
// Project 4.01 : CT Reconstruction (Filtered Backprojection)
//
// THE BIG IDEA
//   Backprojection is a per-PIXEL GATHER: every output pixel is independent, so
//   we give each pixel its own GPU thread (a 2-D grid over the image). Each
//   thread loops over all projection angles, finds where its ray hits the
//   detector (s = x*cos + y*sin), linearly interpolates the filtered projection
//   there, and accumulates. This is the canonical CT GPU kernel; in production
//   the interpolation is done by texture hardware essentially for free.
//
//   We pass HOST-precomputed cos/sin so the GPU and CPU use identical trig (so
//   their results match within tight tolerance). kernels.cu defines the kernel.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // CTProblem (pure C++, safe in .cu)

// Device kernel: thread (px,py) reconstructs one image pixel.
//   filtered : [n_angles*n_det] ramp-filtered sinogram (device)
//   cosv,sinv: [n_angles] precomputed trig (device)
//   center   : detector index of s=0  ((n_det-1)/2)
//   pix      : world units per pixel ; scale = pi/n_angles ; W = world_half
__global__ void backproject_kernel(const float* __restrict__ filtered,
                                   const float* __restrict__ cosv,
                                   const float* __restrict__ sinv,
                                   int n_angles, int n_det, int N,
                                   float ds, float center, float W, float pix, float scale,
                                   float* __restrict__ image);

// Host wrapper: upload filtered sinogram + trig, launch the 2-D grid, copy the
// reconstructed image back, and report the kernel time.
//   image     : resized to img*img; filled with the reconstruction
//   kernel_ms : out-param, GPU kernel time (ms)
void backproject_gpu(const CTProblem& ct, const std::vector<float>& filtered,
                     const std::vector<float>& cosv, const std::vector<float>& sinv,
                     std::vector<float>& image, float* kernel_ms);
