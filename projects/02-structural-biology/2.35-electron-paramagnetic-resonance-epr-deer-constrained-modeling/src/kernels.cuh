// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 2.35 : Electron Paramagnetic Resonance (EPR/DEER) Constrained Modeling
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls deer_backcalc_gpu() to get
//   the per-frame DEER histograms on the device; kernels.cu implements both the
//   host wrapper and the device kernel. Included only by .cu translation units
//   (it contains a __global__ declaration, so the plain C++ compiler must never
//   see it -- that is why the CPU reference lives in a separate pure-C++ header).
//
// THE BIG IDEA  (PATTERNS.md §1: "the same expensive job for many members")
//   The DEER pipeline has two stages with very different shapes:
//     1. BACK-CALCULATION (heavy, embarrassingly parallel):  each MD frame's
//        P_m(r) needs a ROTAMERS^2 rotamer convolution. The M frames are fully
//        independent, so we map ONE FRAME PER GPU THREAD and compute all M
//        histograms in a single launch. No two threads touch the same output
//        row -> no atomics, no races. This is what the GPU accelerates.
//     2. REWEIGHTING (light, sequential-ish):  a gradient descent over the tiny
//        M-vector of weights (O(M*NBINS) per step). This is cheap and is run as
//        SHARED HOST CODE (reference_cpu.cpp's reweight_cpu) for BOTH paths, so
//        the GPU and CPU reweighting trajectories are identical. We deliberately
//        do NOT GPU-ify it -- see ../THEORY.md "GPU mapping" for the why.
//
//   Thread-to-data mapping for stage 1: frame m = blockIdx.x*blockDim.x +
//   threadIdx.x; that thread reads the m-th slice of the rotamer arrays and
//   writes the m-th row of the [M x NBINS] histogram matrix via the SHARED
//   deer_member_histogram() in deer.h (identical math to the CPU).
//
// READ THIS AFTER: deer.h, util/cuda_check.cuh, util/timer.cuh. Then kernels.cu.
// ===========================================================================
#pragma once

#include <vector>

#include "deer.h"          // Spin3, deer_member_histogram (shared host/device)
#include "deer_params.h"   // NBINS, ROTAMERS_PER_SITE

// ---- Device kernel -------------------------------------------------------
// deer_backcalc_kernel: one thread back-calculates ONE frame's P_m(r).
//   M     : number of frames (guards the ragged last block)
//   siteA : device ptr, [M*ROTAMERS_PER_SITE] site-1 rotamer endpoints (nm)
//   siteB : device ptr, [M*ROTAMERS_PER_SITE] site-2 rotamer endpoints (nm)
//   hist  : device ptr, [M*NBINS] output histograms (row m = frame m's P_m(r))
//   __restrict__ promises the pointers do not alias so the compiler can keep
//   loads in registers. Each thread writes only its own NBINS-long output row.
__global__ void deer_backcalc_kernel(int M,
                                     const Spin3* __restrict__ siteA,
                                     const Spin3* __restrict__ siteB,
                                     double* __restrict__ hist);

// ---- Host wrapper --------------------------------------------------------
// deer_backcalc_gpu: run stage-1 back-calculation on the GPU.
//   Allocates device buffers, copies the rotamer clouds H2D, launches
//   deer_backcalc_kernel over all M frames, copies the [M*NBINS] histograms D2H,
//   and reports the measured KERNEL time (CUDA events) via *kernel_ms. main.cu
//   then feeds these histograms into the shared reweighting. All CUDA bookkeeping
//   is hidden here.
//
//   siteA, siteB : host rotamer clouds, each length M*ROTAMERS_PER_SITE
//   hist         : host output, resized to M*NBINS (output parameter)
//   kernel_ms    : out-param, milliseconds spent in the kernel itself (not copies)
void deer_backcalc_gpu(int M,
                       const std::vector<Spin3>& siteA,
                       const std::vector<Spin3>& siteB,
                       std::vector<double>& hist,
                       float* kernel_ms);
