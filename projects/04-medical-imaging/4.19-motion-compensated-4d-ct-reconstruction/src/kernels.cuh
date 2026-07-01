// ===========================================================================
// src/kernels.cuh  --  GPU motion-compensated reconstruction interface
// ---------------------------------------------------------------------------
// Project 4.19 : Motion-Compensated 4D-CT Reconstruction (2-D teaching version)
//
// THE BIG IDEA
//   Reconstruction is a per-PIXEL GATHER: every output pixel is independent, so
//   we give each pixel its own GPU thread on a 2-D grid over the image (exactly
//   like flagship 4.01). Each thread loops over every (phase, angle) projection;
//   for the MOTION-COMPENSATED reconstruction it first DISPLACES the pixel by
//   that phase's Deformation Vector Field (DVF), so every phase's rays are
//   sampled where the tissue actually was -- collapsing P under-sampled phase
//   images into ONE sharp reference image.
//
//   The per-pixel math is NOT written here: it lives in mc4dct.h as the shared
//   __host__ __device__ function mc_pixel(), which both this kernel and the CPU
//   reference call. That is what makes the GPU and CPU results bit-identical
//   (PATTERNS.md section 2). This header only declares the launch wrapper and
//   the thin kernel that calls mc_pixel().
//
//   We pass HOST-precomputed cos/sin so GPU and CPU use identical trig.
//
// READ THIS AFTER: mc4dct.h, util/cuda_check.cuh, util/timer.cuh. Then kernels.cu.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // FourDCTProblem, Geom (pure C++, safe in .cu)

// ---- Device kernel -------------------------------------------------------
// reconstruct_kernel: thread (px,py) reconstructs one image pixel by calling the
//   shared mc_pixel() from mc4dct.h.
//   filtered : [total_angles * n_det] ramp-filtered sinogram (device)
//   cosv,sinv: [total_angles] precomputed trig (device)
//   motion_comp: 0 = naive 4D-FBP, 1 = motion-compensated (DVF warp per phase)
//   image    : [img*img] output (device)
//   The Geom is passed BY VALUE so every thread has its tiny geometry struct in
//   registers/constant space -- no pointer chasing for scalars.
__global__ void reconstruct_kernel(Geom g,
                                   const float* __restrict__ cosv,
                                   const float* __restrict__ sinv,
                                   const float* __restrict__ filtered,
                                   int motion_comp,
                                   float* __restrict__ image);

// ---- Host wrapper --------------------------------------------------------
// reconstruct_gpu: upload filtered sinogram + trig, launch the 2-D grid, copy
//   the reconstructed image back, and report the kernel time.
//   prob      : geometry + (only geom is used here; sinogram already filtered)
//   filtered  : host ramp-filtered sinogram [total_angles*n_det]
//   cosv,sinv : host trig [total_angles]
//   motion_comp: 0 naive, 1 motion-compensated
//   image     : resized to img*img and filled with the reconstruction
//   kernel_ms : out-param, GPU kernel time (ms), measured with CUDA events
void reconstruct_gpu(const FourDCTProblem& prob, const std::vector<float>& filtered,
                     const std::vector<float>& cosv, const std::vector<float>& sinv,
                     int motion_comp, std::vector<float>& image, float* kernel_ms);
