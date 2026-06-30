// ===========================================================================
// src/kernels.cuh  --  GPU Delay-and-Sum beamforming interface
// ---------------------------------------------------------------------------
// Project 4.6 : Ultrasound Beamforming (Delay-and-Sum)
//
// THE BIG IDEA
//   DAS beamforming is a per-PIXEL GATHER: every output pixel is independent, so
//   we give each pixel its own GPU thread (a 2-D grid over the (x,z) image).
//   Each thread loops over all transducer elements, computes the round-trip
//   focal delay to its pixel, linearly interpolates that element's RF trace at
//   the delay, and accumulates. This is the canonical ultrasound GPU kernel; in
//   production the interpolation is often handed to texture hardware (free
//   bilinear fetch) and the RF data lives in textures -- noted in THEORY.md.
//
//   The per-(pixel,element) math is NOT duplicated here: the kernel calls
//   das_pixel() from beamform.h, the very same __host__ __device__ function the
//   CPU reference calls (PATTERNS.md §2). That is why GPU==CPU to tight
//   tolerance: identical operations, identical order.
//
// READ THIS AFTER: beamform.h, util/cuda_check.cuh, util/timer.cuh.
//                  Then read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>
#include "beamform.h"        // BeamformGeom + das_pixel (shared HD core)
#include "reference_cpu.h"   // BeamformProblem (pure C++, safe inside a .cu)

// ---- Device kernel -------------------------------------------------------
// das_kernel: thread (ix, iz) reconstructs one image pixel.
//   g   : geometry, passed BY VALUE (lands in constant/param space; every thread
//         reads the same small struct -- ideal, no global traffic for scalars)
//   rf  : [n_elements * n_samples] RF data in device global memory (__restrict__
//         promises no aliasing so the compiler can keep the interpolated loads
//         in registers)
//   img : [nx * nz] output image (signed coherent sum per pixel)
// Launch: 2-D grid of 16x16 blocks covering the nx-by-nz image (see kernels.cu).
__global__ void das_kernel(BeamformGeom g,
                           const float* __restrict__ rf,
                           float* __restrict__ img);

// ---- Host wrapper --------------------------------------------------------
// beamform_gpu: upload RF data, launch the 2-D grid, copy the image back, and
//   report the measured KERNEL time (CUDA events) via *kernel_ms. main.cu calls
//   exactly this; all CUDA bookkeeping (malloc/memcpy/free) is hidden inside.
//   image     : host output, resized to nx*nz (output parameter)
//   kernel_ms : out-param, milliseconds spent in the kernel itself (not copies)
void beamform_gpu(const BeamformProblem& p, std::vector<float>& image,
                  float* kernel_ms);
