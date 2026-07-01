// ===========================================================================
// src/kernels.cuh  --  GPU reconstruction interface (declarations + the idea)
// ---------------------------------------------------------------------------
// Project 4.13 : Photoacoustic Image Reconstruction
//
// THE BIG IDEA
//   Delay-and-sum (DAS) backprojection is a per-PIXEL GATHER: every output pixel
//   is computed independently, reading one interpolated sample from every sensor
//   trace. So we lay a 2-D thread grid over the 2-D image and give each pixel its
//   own thread. Thread (px, py) computes exactly the value the CPU reference
//   computes for pixel (px, py) -- because both call the SAME pa_pixel_das()
//   from pa_core.h. No shared memory or atomics are needed: pixels never write
//   to each other, so this is an embarrassingly-parallel gather (the ideal GPU
//   workload). This is the same shape as CT backprojection (project 4.01); the
//   difference is that the "projection" is now a time-series and the geometry is
//   a travel-time delay instead of a fan/parallel ray.
//
//   The sensor geometry (sx, sy) and traces are small/read-only and read by
//   every thread; we keep them in plain global memory here for clarity (the
//   L2/texture cache serves the shared re-reads). THEORY.md §GPU-mapping discusses
//   the texture-memory and constant-memory optimizations production PA does.
//
// READ THIS AFTER: pa_core.h (the physics), util/cuda_check.cuh, util/timer.cuh.
//                  Then read kernels.cu (the launch + memory bookkeeping).
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // PAProblem (pure C++, safe to include in a .cu)

// ---- Device kernel -------------------------------------------------------
// das_kernel: thread (px, py) reconstructs one image pixel via delay-and-sum.
//   Launch config (set in reconstruct_gpu):
//     block = 16 x 16 threads  (a 2-D tile matching the 2-D image)
//     grid  = ceil(img/16) x ceil(img/16) blocks covering the whole image
//   Thread-to-data map: px = blockIdx.x*blockDim.x + threadIdx.x (image column),
//                       py = blockIdx.y*blockDim.y + threadIdx.y (image row).
//   Device pointers (all in global memory):
//     d_sx, d_sy : [n_sensors] sensor coordinates [m]
//     d_sig      : [n_sensors * n_samples] pressure traces (sensor-major)
//     d_image    : [img*img] output, row-major (written once per thread)
//   Scalars precomputed on the host (identically to the CPU) so the arithmetic
//   matches bit-for-bit: pix (metres/pixel), inv_c, inv_dt, inv_ns.
__global__ void das_kernel(const float* __restrict__ d_sx,
                           const float* __restrict__ d_sy,
                           const float* __restrict__ d_sig,
                           int n_sensors, int n_samples, int img,
                           float world_half, float pix,
                           float inv_c, float inv_dt, float inv_ns,
                           float* __restrict__ d_image);

// ---- Host wrapper --------------------------------------------------------
// reconstruct_gpu: the host-callable "do the whole GPU reconstruction". It
// uploads the sensor geometry + traces, launches the 2-D grid of das_kernel,
// copies the reconstructed image back, and reports the KERNEL time (CUDA events,
// not counting the PCIe copies) via *kernel_ms. main.cu calls exactly this.
//   image     : resized to img*img and filled with the reconstruction (out param)
//   kernel_ms : out-param, milliseconds spent in the kernel itself
void reconstruct_gpu(const PAProblem& pa, std::vector<float>& image,
                     float* kernel_ms);
