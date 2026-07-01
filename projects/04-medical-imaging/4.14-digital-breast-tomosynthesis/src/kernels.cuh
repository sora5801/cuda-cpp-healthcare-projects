// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 4.14 : Digital Breast Tomosynthesis
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls reconstruct_sart_gpu();
//   kernels.cu implements the host driver and the two device kernels. This
//   header is included ONLY by .cu translation units (it declares __global__
//   kernels, which the plain host C++ compiler must never see -- that is why the
//   CPU reference lives in its own pure-C++ header, reference_cpu.h).
//
// THE TWO KERNELS = THE TWO HALVES OF SART (docs/PATTERNS.md gather pattern)
//   SART alternates two independent, embarrassingly-parallel gathers, and each
//   maps to one kernel:
//
//   1. FORWARD PROJECTION  (forward_project_kernel)
//        one thread per detector RAY (angle k, bin j).
//        The thread marches along its ray, bilinearly sampling the current image
//        estimate (dbt_geometry.h::forward_ray_integral) and summing -> the
//        simulated line integral for that ray. Pure read of the image, one write
//        to sim[k*n_det+j]. No atomics, no shared memory: a textbook gather.
//
//   2. BACKPROJECTION + UPDATE  (backproject_update_kernel)
//        one thread per output PIXEL (px, py) on a 2-D thread grid.
//        The thread gathers the residual from every angle (projecting its world
//        position onto each detector, linearly interpolating), averages, applies
//        the relaxed SART correction, and clamps to >= 0. One read of many
//        residual samples, one write to image[py*N+px]. Again: independent
//        outputs -> no atomics needed.
//
//   Because both kernels call the SAME dbt_geometry.h helpers the CPU reference
//   uses, GPU and CPU produce matching results (verified in main.cu within a
//   small float tolerance -- see THEORY.md, verification).
//
// READ THIS AFTER: dbt_geometry.h, util/cuda_check.cuh, util/timer.cuh.
//                  Then read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // DBTProblem (shared struct; pure C++, safe in .cu)

// ---- Device kernels ------------------------------------------------------
// Declared here so the launch configuration lives next to the interface; the
// full launch-config reasoning is in kernels.cu at each kernel's definition.

// Forward projection: sim[k*n_det+j] = line integral of `image` along ray (k,j).
//   grid  : covers n_angles*n_det rays (1-D)
//   thread: global index -> (k, j)
__global__ void forward_project_kernel(const float* __restrict__ image,
                                       const float* __restrict__ cosv,
                                       const float* __restrict__ sinv,
                                       int n_angles, int n_det, int N,
                                       float ds, float center, float W, float pix,
                                       int steps, float dt,
                                       float* __restrict__ sim);

// Backprojection + relaxed SART update, in place on `image`.
//   grid  : 2-D, covers the N x N image
//   thread: (px, py) owns image pixel (px, py)
__global__ void backproject_update_kernel(const float* __restrict__ residual,
                                          const float* __restrict__ cosv,
                                          const float* __restrict__ sinv,
                                          int n_angles, int n_det, int N,
                                          float ds, float center, float W, float pix,
                                          float lambda, float inv_na,
                                          float* __restrict__ image);

// ---- Host driver ---------------------------------------------------------
// reconstruct_sart_gpu: run the whole GPU SART reconstruction.
//   Uploads the measured projections + angle tables once, keeps the image and
//   scratch buffers device-resident across all iterations (no per-iteration H2D
//   traffic), loops {forward, residual, backproject-update} n_iters times, and
//   copies the final image back. Reports the total device kernel time (CUDA
//   events, summed over all launches) via *kernel_ms.
//
//   p     : the problem (geometry + measured projections + SART schedule).
//   cosv, sinv : host angle tables from compute_angles() (uploaded once).
//   image : host output, resized to img*img (output parameter).
//   kernel_ms  : out-param, total ms spent inside the kernels (not copies).
void reconstruct_sart_gpu(const DBTProblem& p,
                          const std::vector<float>& cosv,
                          const std::vector<float>& sinv,
                          std::vector<float>& image,
                          float* kernel_ms);
