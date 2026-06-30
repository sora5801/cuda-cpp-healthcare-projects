// ===========================================================================
// src/reference_cpu.h  --  Dataset + shared GIST host helpers + CPU reference
// ---------------------------------------------------------------------------
// Project 2.26 : Hydrogen Bond Network & Water Placement Analysis
//
// Pure C++ (no CUDA). The per-sample physics and the per-voxel thermodynamics
// live in gist.h as __host__ __device__ functions, so the CPU reference here and
// the GPU kernels (kernels.cu) compute byte-for-byte identical numbers. This
// header declares:
//   * Dataset            -- the loaded MD frames + solute + grid.
//   * gist_cpu()         -- the trusted serial GIST baseline.
//   * derive_voxels()    -- tallies -> ranked VoxelResult list (shared by CPU/GPU).
//   * load_dataset()     -- read the data/sample text file.
//
// kernels.cu reuses Dataset + derive_voxels() so both paths share the reduction.
// READ THIS AFTER: gist.h.  BEFORE: reference_cpu.cpp, kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "gist.h"   // GistGrid, VoxelResult, gist_* physics + fixed-point

// ---------------------------------------------------------------------------
// Dataset: everything loaded from data/sample/.
//   Water-oxygen positions are stored as a FLAT array: for frame f and water w,
//   the (x,y,z) live at waters[((f*waters_per_frame)+w)*3 + {0,1,2}]. The solute
//   is a fixed set of atoms (position + partial charge). The grid is the box we
//   accumulate over.
// ---------------------------------------------------------------------------
struct Dataset {
    GistGrid grid{};                 // voxel grid over the binding pocket
    int nframes = 0;                 // number of MD snapshots
    int waters_per_frame = 0;        // water oxygens recorded per frame (fixed)
    int natoms = 0;                  // solute atom count
    std::vector<float> waters;       // [nframes * waters_per_frame * 3] xyz, row-major
    std::vector<float> atoms;        // [natoms * 4] (x, y, z, charge)

    // Total number of (water, frame) scatter samples = the kernel's thread count.
    long long num_samples() const {
        return static_cast<long long>(nframes) * waters_per_frame;
    }
};

// Load the text sample (format documented in data/README.md). Throws
// std::runtime_error on any malformed/short input so the demo fails loudly.
Dataset load_dataset(const std::string& path);

// ---------------------------------------------------------------------------
// derive_voxels: SHARED reduction step (host code, called by BOTH paths).
//   Given per-voxel raw tallies (occupancy counts + fixed-point energy sums),
//   produce a VoxelResult for every sufficiently-OCCUPIED voxel (under-sampled
//   voxels are dropped as noise; see GIST_MIN_OCCUPANCY_FRACTION) and sort them so
//   the output is deterministic and the strongest hydration sites come first.
//   Sort key: occupancy (count) descending -- the robust "where do waters cluster"
//   signal that identifies a site -- then dG descending (displaceability), then
//   flat voxel index. A strict total order, so CPU and GPU rankings are identical.
// ---------------------------------------------------------------------------
std::vector<VoxelResult> derive_voxels(const Dataset& d,
                                       const std::vector<unsigned int>& counts,
                                       const std::vector<gist_fixed_t>& esum);

// ---------------------------------------------------------------------------
// gist_cpu: the trusted serial reference. Streams every (water, frame), finds its
//   voxel, accumulates occupancy + fixed-point energy, then calls derive_voxels.
//   Fills `counts` and `esum` (length = grid.num_voxels()) with the raw tallies
//   and returns the ranked voxel list. main.cu compares these against the GPU's.
// ---------------------------------------------------------------------------
std::vector<VoxelResult> gist_cpu(const Dataset& d,
                                  std::vector<unsigned int>& counts,
                                  std::vector<gist_fixed_t>& esum);
