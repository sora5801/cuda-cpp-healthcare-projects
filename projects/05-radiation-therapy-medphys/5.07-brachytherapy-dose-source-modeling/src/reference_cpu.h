// ===========================================================================
// src/reference_cpu.h  --  Plan definition + CPU TG-43 dose reference
// ---------------------------------------------------------------------------
// Project 5.7 : Brachytherapy Dose & Source Modeling
//
// This header defines the whole brachytherapy PLAN (source model + dwell
// positions + the 3-D dose grid) and declares the pure-C++ reference dose
// calculator. It is CUDA-free: kernels.cu can #include it to reuse the same
// Plan struct, and reference_cpu.cpp compiles it with the plain host compiler.
//
// The per-voxel physics is NOT here -- it lives in tg43_physics.h so the CPU
// reference and the GPU kernel share byte-identical math (PATTERNS.md section 2).
//
// READ THIS AFTER: tg43_physics.h.  BEFORE: reference_cpu.cpp, kernels.cuh.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "tg43_physics.h"   // SourceModel, Dwell, dose_rate_one_dwell (HD-safe)

// ---------------------------------------------------------------------------
// DoseGrid: a regular Cartesian grid of voxel centers on which we evaluate dose.
//   The grid is axis-aligned; voxel (ix,iy,iz) has its CENTER at
//     origin + (ix*spacing, iy*spacing, iz*spacing).
//   We flatten to 1-D in x-fastest (row-major over z,y,x) order:
//     index(ix,iy,iz) = (iz*ny + iy)*nx + ix.
//   Keeping a single flat vector<float> makes the CPU<->GPU copy a trivial
//   contiguous memcpy and matches how the kernel indexes voxels.
// ---------------------------------------------------------------------------
struct DoseGrid {
    int nx = 0, ny = 0, nz = 0;     // voxel counts per axis
    double ox = 0, oy = 0, oz = 0;  // world position [cm] of voxel (0,0,0)'s center
    double spacing = 0.1;           // isotropic voxel pitch [cm] (1 mm here)

    int   size()  const { return nx * ny * nz; }               // total voxels
    // World-space center of voxel (ix,iy,iz).
    double cx(int ix) const { return ox + ix * spacing; }
    double cy(int iy) const { return oy + iy * spacing; }
    double cz(int iz) const { return oz + iz * spacing; }
};

// ---------------------------------------------------------------------------
// Plan: a complete, self-contained TG-43 calculation job. This is exactly what
// gets handed to both dose_cpu() and the GPU path (dose_gpu()). One SourceModel
// (a single source TYPE stepped to many dwells, the HDR afterloader model), the
// list of dwell positions/weights, and the output grid.
// ---------------------------------------------------------------------------
struct Plan {
    SourceModel        source;   // TG-43 consensus dataset for the source type
    std::vector<Dwell> dwells;   // dwell positions + weights (the implant plan)
    DoseGrid           grid;     // where to compute dose
};

// ---------------------------------------------------------------------------
// load_plan: parse a Plan from the tiny text format in data/README.md.
//   Throws std::runtime_error on a malformed / missing file so demos fail loudly
//   rather than silently computing on garbage. Path is the sample file.
// ---------------------------------------------------------------------------
Plan load_plan(const std::string& path);

// ---------------------------------------------------------------------------
// dose_cpu: the reference implementation. For every voxel, sum the TG-43 dose
// rate over all dwell positions (dose_rate_one_dwell from tg43_physics.h). This
// is the SAME nested computation the GPU does, just serial -- so the two dose
// arrays must agree within floating-point tolerance.
//   `dose` is resized to grid.size() and filled with cGy/h per voxel.
//   Complexity: O(n_voxels * n_dwells) -- embarrassingly parallel over voxels.
// ---------------------------------------------------------------------------
void dose_cpu(const Plan& plan, std::vector<float>& dose);
