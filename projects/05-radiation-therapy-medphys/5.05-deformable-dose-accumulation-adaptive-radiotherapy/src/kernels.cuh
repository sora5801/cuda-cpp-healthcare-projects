// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 5.5 : Deformable Dose Accumulation & Adaptive Radiotherapy
//               (reduced-scope 2-D teaching version)
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls two host wrappers:
//     register_gpu()      -- run Demons DIR on the GPU  -> the DVF (ux,uy).
//     accumulate_dose_gpu()-- warp the delivered dose by the DVF, accumulate it,
//                             and build the dose-volume histogram, all on the GPU.
//   kernels.cu implements those wrappers plus the device kernels. This header is
//   included only by .cu translation units (it declares __global__ kernels, so
//   the plain C++ compiler must never see it -- that is why the CPU reference
//   lives in reference_cpu.h).
//
// THE BIG IDEA (two GPU patterns, one pipeline)
//   Stage A -- DIR (the same three-pass Demons as project 4.8):
//     one thread per pixel over nx*ny; per iteration run
//       force -> smooth-x -> smooth-y (ping-pong buffers).  [gather + stencil]
//   Stage B -- deformable dose accumulation (the NEW part of 5.5):
//     warp_dose_kernel   : one thread per voxel GATHERS the delivered dose at the
//                          deformed position (x+ux,y+uy) via bilinear -> a race-
//                          free per-voxel write (cf. 4.01 CT backprojection).
//     accumulate_kernel  : total[i] += warped[i], one thread per voxel (race-free).
//     dvh_kernel         : one thread per voxel atomicAdd's 1 into its dose bin --
//                          an INTEGER histogram, so the parallel reduction is
//                          DETERMINISTIC and matches the CPU exactly (PATTERNS.md §3).
//
//   All per-voxel math is the SAME demons.h / dose.h code the CPU reference runs,
//   so GPU and CPU agree within the tolerances in ../THEORY.md.
//
// READ THIS AFTER: demons.h, dose.h, util/cuda_check.cuh, util/timer.cuh.
//                  Then kernels.cu.
// ===========================================================================
#pragma once

#include <vector>

#include "demons.h"          // DemonsParams + shared DIR per-pixel core
#include "dose.h"            // DVH_BINS + shared dose warp / bin core
#include "reference_cpu.h"   // ArtCase (the loaded planning/daily images + doses)

// ---- Stage A: DIR device kernels (documented in full in kernels.cu) -------
// demons_force_kernel: one thread per pixel adds the Demons update du to (ux,uy).
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

// ---- Stage B: dose warp / accumulate / histogram device kernels -----------
// warp_dose_kernel: one thread per voxel gathers the deformed delivered dose.
//   dose    : delivered dose on today's grid [ny*nx].
//   ux,uy   : the DVF from Stage A.
//   out     : warped dose in the planning frame [ny*nx] (one write per thread).
__global__ void warp_dose_kernel(const double* __restrict__ dose,
                                 const double* __restrict__ ux,
                                 const double* __restrict__ uy,
                                 double* __restrict__ out,
                                 int nx, int ny);

// accumulate_kernel: total[i] += add[i], one thread per voxel (race-free: each
//   thread owns a distinct index).
__global__ void accumulate_kernel(double* __restrict__ total,
                                  const double* __restrict__ add,
                                  int n);

// dvh_kernel: one thread per voxel atomicAdd's 1 into hist[dvh_bin(dose[i])].
//   INTEGER atomics -> the histogram is order-independent and deterministic.
__global__ void dvh_kernel(const double* __restrict__ dose,
                           unsigned* __restrict__ hist,
                           int n);

// ---- Host wrappers --------------------------------------------------------
// register_gpu: run the full Demons solver on the GPU, return the DVF (ux,uy).
//   c         : the ART case (uses plan_img as FIXED, daily_img as MOVING).
//   P         : run parameters.
//   ux,uy     : host output displacement field, each resized to nx*ny.
//   kernel_ms : out-param, ms spent in the DIR iteration loop (CUDA-event timed).
void register_gpu(const ArtCase& c, const DemonsParams& P,
                  std::vector<double>& ux, std::vector<double>& uy,
                  float* kernel_ms);

// accumulate_dose_gpu: Stage B end to end on the GPU. Warps c.daily_dose by the
//   DVF, accumulates `nfractions` copies of it into a running total (the demo
//   delivers the same fraction nfractions times), and histograms the result.
//   ux,uy       : the DVF (host; copied to device once).
//   nfractions  : how many identical fractions to accumulate (>=1).
//   total_out   : host output accumulated dose [nx*ny] (planning frame, Gy).
//   dvh_out     : host output DVH counts [DVH_BINS] (of the accumulated dose).
//   kernel_ms   : out-param, ms spent in the Stage-B kernels (CUDA-event timed).
void accumulate_dose_gpu(const ArtCase& c,
                         const std::vector<double>& ux,
                         const std::vector<double>& uy,
                         int nfractions,
                         std::vector<double>& total_out,
                         std::vector<unsigned>& dvh_out,
                         float* kernel_ms);
