// ===========================================================================
// src/reference_cpu.h  --  Prototypes for the CPU reference decoder + data load
// ---------------------------------------------------------------------------
// Project 4.32 : GPU-Accelerated Landmark Detection
//
// WHY A SEPARATE PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the plain host compiler (cl.exe / g++) and
//   must never see CUDA/__global__ syntax, so its prototypes cannot live in
//   kernels.cuh. main.cu (nvcc) and reference_cpu.cpp both include THIS header
//   plus the shared, CUDA-free landmark.h, so they agree on the data types and
//   on the decode math.
//
// THE CONTRACT
//   * HeatmapSet  -- an in-memory batch of L landmark heatmaps over one grid.
//   * load_heatmaps  -- parse the tiny text sample in data/sample/ into a set.
//   * decode_cpu  -- the trusted reference: argmax + soft-argmax per landmark,
//                    the SAME computation the GPU kernels perform. main.cu runs
//                    both and asserts they agree exactly (integer decode) /
//                    within a tiny tolerance (the final centroid division).
//
// READ THIS BEFORE: reference_cpu.cpp, main.cu. Read landmark.h first.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "landmark.h"   // VolumeDims, Landmark, the shared decode helpers

// ---------------------------------------------------------------------------
// HeatmapSet: L heatmaps, each an nx*ny*nz grid, concatenated in `data`.
//   dims        : the shared grid geometry (nx,ny,nz) every heatmap uses.
//   num_landmarks (L) : how many landmarks / how many volumes are stacked.
//   data        : flat float array of length L * nx*ny*nz, landmark-major
//                 (all of landmark 0's voxels, then landmark 1's, ...). This is
//                 exactly how a network's output tensor [L,Z,Y,X] sits in memory.
//   truth_x/y/z : the KNOWN ground-truth coordinate each synthetic heatmap was
//                 built around (length L). Used only to report decode error --
//                 it validates the SCIENCE (did we recover the planted point?),
//                 separate from the CPU-vs-GPU agreement check. Empty if unknown.
// ---------------------------------------------------------------------------
struct HeatmapSet {
    VolumeDims          dims{0, 0, 0};
    int                 num_landmarks = 0;
    std::vector<float>  data;                 // L * nx*ny*nz intensities
    std::vector<double> truth_x, truth_y, truth_z;  // planted coords (optional)
};

// Load a heatmap set from the whitespace-delimited sample format documented in
// data/README.md. Throws std::runtime_error if the file is missing or malformed
// so demos fail loudly rather than silently decoding garbage.
HeatmapSet load_heatmaps(const std::string& path);

// The CPU reference decoder. Fills `out` with one Landmark per heatmap:
//   1. scan the whole volume for the argmax voxel (max intensity),
//   2. accumulate the soft-argmax integer weight sums over the RADIUS window,
//   3. finalize into a sub-voxel (x,y,z) via finalize_softargmax().
// This is the baseline the GPU must match. out is resized to num_landmarks.
void decode_cpu(const HeatmapSet& hs, std::vector<Landmark>& out);
