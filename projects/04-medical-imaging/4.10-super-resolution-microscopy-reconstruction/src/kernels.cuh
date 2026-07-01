// ===========================================================================
// src/kernels.cuh  --  GPU SMLM interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 4.10 : Super-Resolution Microscopy Reconstruction  (STORM / PALM SMLM)
//
// THE BIG IDEA (flagship pattern: INDEPENDENT JOBS + ATOMIC REDUCTION)
//   Two GPU phases, mirroring the CPU reference exactly:
//
//     LOCALIZE : one GPU THREAD PER INTERIOR PIXEL tests whether that pixel is a
//                strict local maximum above threshold; if so it runs the SAME
//                smlm_localize() fit the CPU uses (from smlm.h) on its own 7x7
//                patch and writes the result into the OUTPUT SLOT indexed by its
//                scan position. Every fit is independent (reads only its patch,
//                writes only its slot) -> embarrassingly parallel, no atomics.
//                Because a slot's index IS its (frame,row,col) scan position, a
//                host compaction that keeps slots in index order reproduces the
//                CPU's canonical localization order EXACTLY.
//
//     RENDER   : one GPU thread per localization scatters its photons into the
//                super-resolution image bin its (x,y) falls in, via atomicAdd on
//                FIXED-POINT integers (smlm.h). Integer atomics commute, so the
//                rendered image is order-independent and bit-identical to the CPU
//                render (docs/PATTERNS.md §3). This is the atomic-reduction half.
//
//   kernels.cu implements both kernels + the host wrapper smlm_gpu(). main.cu
//   calls smlm_gpu() and compares its ResultSummary with the CPU's.
//
//   Included ONLY by .cu files (it declares __global__ kernels). The CPU
//   reference cannot see this header -- it uses reference_cpu.h + smlm.h instead.
//
// READ THIS AFTER: smlm.h, reference_cpu.h, util/cuda_check.cuh, util/timer.cuh.
// ===========================================================================
#pragma once

#include <cstddef>
#include <vector>

#include "reference_cpu.h"   // FrameStack, Localization, ResultSummary (pure C++)

// ---- Device kernels (documented at their definitions in kernels.cu) --------

// LOCALIZE: thread per interior pixel of one frame. Writes a Localization into
// slot[i] and sets valid[i]=1 iff pixel i is a detected+localized emitter.
__global__ void localize_kernel(const float* __restrict__ frame, int H, int W,
                                double background, double threshold, int frame_idx,
                                Localization* __restrict__ slot,
                                unsigned char* __restrict__ valid);

// RENDER: thread per localization. Atomically adds each emitter's fixed-point
// photons into its super-resolution bin.
__global__ void render_kernel(const Localization* __restrict__ locs, int n,
                              int srH, int srW,
                              unsigned long long* __restrict__ img_fixed);

// ---- Host wrapper ----------------------------------------------------------
// smlm_gpu: run the whole GPU pipeline on a frame stack.
//   Detects + localizes every frame (localize_kernel), compacts the results into
//   the canonical (frame,row,col) order on the host, renders them into a
//   fixed-point super-resolution image (render_kernel), and fills:
//     out_locs  : the localization list, in canonical order (matches the CPU)
//     img_fixed : the fixed-point super-resolution image (srH*srW)
//     srH, srW  : render dimensions
//     kernel_ms : total GPU kernel time (localize + render), CUDA-event measured
//   Returns the ResultSummary (same fields the CPU produces) for verification.
ResultSummary smlm_gpu(const FrameStack& stack,
                       std::vector<Localization>& out_locs,
                       std::vector<unsigned long long>& img_fixed,
                       int& srH, int& srW, float* kernel_ms);
