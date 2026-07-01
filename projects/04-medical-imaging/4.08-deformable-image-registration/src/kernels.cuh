// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 4.8 : Deformable Image Registration (reduced-scope teaching version)
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls register_gpu(); kernels.cu
//   implements the host wrapper plus three device kernels. Included only by .cu
//   translation units (it declares __global__ kernels, so the plain C++ compiler
//   must never see it -- that is why the CPU reference lives in reference_cpu.h).
//
// THE BIG IDEA (Thirion's Demons on the GPU)
//   One Demons iteration is three data-parallel passes, each ONE THREAD PER
//   PIXEL over the nx*ny image (a 2-D grid of 2-D blocks, cf. 14.02/6.04):
//
//     1. FORCE  : demons_force_kernel -- each thread computes du at its pixel
//                 (warp + gradient + Thirion normalization, all in demons.h) and
//                 adds it to the displacement field. Pure gather, no races: a
//                 thread only writes its own u[i].
//     2. SMOOTH-X : gauss_x_kernel -- separable Gaussian blur along x (stencil).
//     3. SMOOTH-Y : gauss_y_kernel -- separable Gaussian blur along y (stencil).
//
//   Steps 2/3 write into a SECOND pair of buffers (ping-pong) so no thread reads
//   a field another thread is mid-writing. The host wrapper loops these kernels
//   P.iters times, swapping buffers, then copies the final field back.
//
//   All per-pixel math is the SAME demons.h code the CPU reference runs, so the
//   GPU and CPU displacement fields agree within the tolerance in ../THEORY.md.
//
// READ THIS AFTER: demons.h, util/cuda_check.cuh, util/timer.cuh. Then kernels.cu.
// ===========================================================================
#pragma once

#include <vector>

#include "demons.h"           // DemonsParams + the shared __host__ __device__ core
#include "reference_cpu.h"    // DirImages (the loaded fixed/moving pair)

// ---- Device kernels (documented in full in kernels.cu) -------------------
// demons_force_kernel: one thread per pixel adds the Demons update du to (ux,uy).
//   F,M      : device images [ny*nx].
//   ux,uy    : device displacement field, updated in place.
//   P        : parameters (by value -> parameter memory, read by all threads).
__global__ void demons_force_kernel(const double* __restrict__ F,
                                    const double* __restrict__ M,
                                    double* __restrict__ ux,
                                    double* __restrict__ uy,
                                    DemonsParams P);

// gauss_x_kernel / gauss_y_kernel: one thread per pixel, separable Gaussian blur
//   of one displacement component from `src` into `dst` (never in place).
__global__ void gauss_x_kernel(const double* __restrict__ src,
                               double* __restrict__ dst,
                               DemonsParams P);
__global__ void gauss_y_kernel(const double* __restrict__ src,
                               double* __restrict__ dst,
                               DemonsParams P);

// ---- Host wrapper --------------------------------------------------------
// register_gpu: run the full Demons solver on the GPU and return the final
//   displacement field.
//   im        : the fixed/moving image pair (host); copied to the device once.
//   P         : run parameters.
//   ux,uy     : host output displacement field, each resized to nx*ny.
//   kernel_ms : out-param, milliseconds spent in the iteration loop (CUDA-event
//               timed), excluding the one-time H2D/D2H copies.
// main.cu calls exactly this; all CUDA bookkeeping is hidden inside.
void register_gpu(const DirImages& im, const DemonsParams& P,
                  std::vector<double>& ux, std::vector<double>& uy,
                  float* kernel_ms);
