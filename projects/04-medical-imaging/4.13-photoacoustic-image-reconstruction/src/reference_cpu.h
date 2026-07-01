// ===========================================================================
// src/reference_cpu.h  --  Loader + serial CPU reconstruction (the baseline)
// ---------------------------------------------------------------------------
// Project 4.13 : Photoacoustic Image Reconstruction
//
// WHAT THIS PROJECT COMPUTES
//   Reconstruct a 2-D map of optical absorbers p0(x,y) inside tissue from the
//   ultrasound pressure traces that a ring of point sensors recorded after a
//   laser pulse. The reconstruction is DELAY-AND-SUM (DAS) backprojection:
//     for each image pixel x:
//        b(x) = (1/S) * sum over sensors s of  g_s( |x - p_s| / c )
//   i.e. look up, in every sensor trace, the sample that would have arrived from
//   x, and sum them. Constructive interference recovers the true source.
//
// WHY A GPU  (the catalog "Deep dive")
//   DAS is a per-pixel GATHER: every output pixel independently reads one
//   interpolated sample from every sensor. A clinical 3-D volume of 256^3 voxels
//   with 1024 sensors is ~68 billion delay-and-sum operations PER FRAME -- only
//   a GPU makes real-time (multi-frame/second) interventional PA imaging
//   feasible. We do the 2-D ring-array case so the geometry stays legible; the
//   3-D extension is the same loop with a z coordinate (THEORY.md §real-world).
//
// FILE ROLES
//   * reference_cpu.h  (this file) : the PAProblem loader + serial DAS declared.
//   * reference_cpu.cpp            : their implementations (host compiler only).
//   * pa_core.h                    : the shared per-pixel physics used by BOTH
//                                    this reference AND the GPU kernel.
//
// This header is pure C++ (it pulls in pa_core.h which is CUDA-clean), so it is
// safe to #include from kernels.cu as well. READ pa_core.h FIRST.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "pa_core.h"   // PAProblem + the shared __host__ __device__ DAS core

// ---------------------------------------------------------------------------
// load_pa: parse a PAProblem from the text format documented in data/README.md:
//   header : "<n_sensors> <n_samples> <dt> <c> <img> <world_half>"
//   then   : n_sensors lines, each "<sx> <sy>" (sensor position, metres)
//   then   : n_sensors lines, each n_samples floats (that sensor's trace)
// Throws std::runtime_error on a missing file or malformed/truncated content so
// demos fail loudly rather than reconstructing garbage.
// ---------------------------------------------------------------------------
PAProblem load_pa(const std::string& path);

// ---------------------------------------------------------------------------
// reconstruct_cpu: the trusted serial baseline. Fills `image` (sized img*img,
// row-major, image[py*img + px]) by calling pa_pixel_das() for every pixel.
// This is the exact computation the GPU kernel parallelizes; the demo runs both
// and asserts they agree to a tiny tolerance (identical PA_HD code, but the GPU
// contracts FMAs so they match to ~1e-5, not bit-for-bit -- PATTERNS.md §4).
// image is resized inside.
// ---------------------------------------------------------------------------
void reconstruct_cpu(const PAProblem& pa, std::vector<float>& image);
