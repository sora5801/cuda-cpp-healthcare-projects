// ===========================================================================
// src/reference_cpu.h  --  RF-data loader + serial DAS beamforming reference
// ---------------------------------------------------------------------------
// Project 4.6 : Ultrasound Beamforming (Delay-and-Sum)
//
// WHAT THIS PROJECT COMPUTES
//   Reconstruct a B-mode ultrasound image from raw per-element RF echo data
//   using Delay-And-Sum (DAS) beamforming. For every image pixel we:
//     1. compute the round-trip travel time from the transmit source, to the
//        pixel, back to each receive element (the focal "delay");
//     2. look up (interpolate) that element's RF signal at that delay;
//     3. sum across all elements (the coherent "sum").
//   The actual per-pixel/per-element physics lives in beamform.h so the CPU and
//   GPU share one formula (PATTERNS.md §2). THIS file is the trusted SERIAL
//   baseline that the GPU kernel is verified against.
//
// WHY A GPU
//   DAS is a per-pixel GATHER: each output pixel reads one interpolated sample
//   from EVERY element, and all pixels are independent. A clinical frame can be
//   512x512 pixels x 128 elements x thousands of frames/sec -> ~10^10-10^11
//   multiply-accumulates per second, far past real-time CPU capability but a
//   natural fit for one GPU thread per pixel. Here we use a small grid so the
//   geometry stays legible; the GPU pattern is identical at clinical scale.
//
//   Pure-C++ header (no CUDA). kernels.cu reuses BeamformGeom from beamform.h.
//
// READ THIS AFTER: beamform.h.   READ THIS BEFORE: reference_cpu.cpp, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "beamform.h"   // BeamformGeom + the shared __host__ __device__ physics

// One beamforming problem: the geometry plus the raw RF echo data.
//   rf[e * n_samples + t] = element e's recorded signal at sample index t
//   (time t/fs + t0 seconds after transmit). Layout is element-major so each
//   element's fast-time trace is contiguous -- the layout the kernel wants too.
struct BeamformProblem {
    BeamformGeom geom;          // all geometry/units (see beamform.h)
    std::vector<float> rf;      // [n_elements * n_samples] raw RF data
};

// Load a BeamformProblem from the text format in data/README.md:
//   header: "<n_elements> <n_samples> <nx> <nz> <fs> <c> <pitch> "
//           "<x_min> <z_min> <dx> <dz> <t0>"
//   then n_elements rows of n_samples floats (the RF data).
// Throws std::runtime_error on a missing/short/garbled file so demos fail loud.
BeamformProblem load_beamform(const std::string& path);

// CPU reference DAS: image[iz*nx + ix] = das_pixel(...) for every pixel, by
// calling the SAME beamform.h core the kernel uses. `image` is sized to nx*nz
// and holds the signed coherent sum (envelope/|.| is taken later in main.cu).
// This is the baseline the GPU result is checked against to tight tolerance.
void beamform_cpu(const BeamformProblem& p, std::vector<float>& image);
