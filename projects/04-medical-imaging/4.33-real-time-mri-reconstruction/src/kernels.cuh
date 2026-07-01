// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 4.33 : Real-Time MRI Reconstruction
//
// WHAT THE GPU DOES HERE
//   reconstruct_frames_gpu() reconstructs the WHOLE sliding-window movie on the
//   device: for each frame it grids that frame's window of radial samples onto a
//   Cartesian grid, inverse-FFTs with cuFFT, and deapodizes -- the exact parallel
//   twin of reconstruct_frame_cpu() in reference_cpu.cpp.
//
//   THE KEY GPU IDEA (PATTERNS.md "scatter + atomic reduce", section 1 row
//   "clustering / centroid accumulation"):
//     * GRIDDING is a SCATTER: each of the window's (win * n_ro) k-space samples is
//       independent and spreads onto ~(W+1)^2 nearby grid cells. We give each sample
//       its OWN thread; the thread computes the sample's position + density weight
//       and atomically adds its Kaiser-Bessel-weighted contribution into the grid.
//     * Many threads hit the SAME grid cell, so we accumulate in FIXED-POINT
//       INTEGERS with atomicAdd (grid_core.h to_fixed). Integer atomics are
//       associative -> the result is DETERMINISTIC and bit-identical to the CPU loop,
//       unlike a nondeterministic float atomicAdd (PATTERNS.md section 3).
//     * The inverse FFT is a solved problem, so we use cuFFT (kernels.cu documents
//       exactly what it computes -- "no black box", CLAUDE.md section 6.1.6).
//     * The FFT-shift and deapodization are per-pixel maps -> one thread per pixel.
//
// READ THIS AFTER: reference_cpu.h (the data model + the CPU twin). The math is in
// grid_core.h; the "why" is in ../THEORY.md.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // RadialData, GriddingParams (shared problem model)

// ---------------------------------------------------------------------------
// reconstruct_frames_gpu: reconstruct every sliding-window frame on the GPU.
//   For frame f (f = 0..n_frames-1) the window starts at spoke f*stride and spans
//   `win` spokes. The output is the concatenation of all frames' magnitude images:
//   out_frames has size n_frames * n * n, row-major per frame.
//     * d          : the loaded radial acquisition (samples, geometry, window plan)
//     * out_frames : filled with n_frames magnitude images (each n*n), back to back
//     * kernel_ms  : OUT, GPU time (ms, CUDA-event measured) for the whole movie --
//                    a teaching artifact, printed to stderr (CLAUDE.md section 12)
//
//   Mirrors reconstruct_frame_cpu() per frame; main.cu runs both and asserts they
//   agree. Because the grid accumulator is fixed-point, the agreement is EXACT for
//   the gridding step; the only residual difference is cuFFT vs our radix-2 FFT.
// ---------------------------------------------------------------------------
void reconstruct_frames_gpu(const RadialData& d,
                            std::vector<float>& out_frames,
                            float* kernel_ms);
