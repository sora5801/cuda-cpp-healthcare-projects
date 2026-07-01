// ===========================================================================
// src/reference_cpu.h  --  Plan definition + CPU pencil-beam dose reference
// ---------------------------------------------------------------------------
// Project 5.3 : Proton & Heavy-Ion Therapy Dose
//
// The CPU reference computes the SAME dose the GPU computes -- same grid, same
// beam model, same per-(voxel,spot) formula from proton_physics.h -- just
// serially, one voxel at a time. main.cu runs both and checks they agree to a
// documented tolerance. This header is pure C++ (no CUDA), so kernels.cu can
// reuse the Plan struct and the loader without dragging CUDA into a host-only TU.
//
// READ THIS AFTER: proton_physics.h (the physics core).  Then kernels.cuh.
// ===========================================================================
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "proton_physics.h"   // Spot, Grid, BeamModel, dose_from_spot (host+device safe)

// ---------------------------------------------------------------------------
// A complete treatment "plan" to compute dose for: the scoring grid, the beam
// model, the list of spots, and the depth at which the beam enters the patient
// (the patient surface). Everything the dose engine needs lives here.
// ---------------------------------------------------------------------------
struct Plan {
    Grid              grid;             // the voxel box we score dose onto
    BeamModel         beam;             // lateral/depth beam-shape parameters
    std::vector<Spot> spots;            // the PBS spot list (>= 1 spot)
    float             z_entry = 0.0f;   // world z of the patient surface (cm); beam enters here
};

// Load a Plan from the tiny text format described in data/README.md. The format
// is line-oriented and whitespace-tolerant (comments start with '#'):
//   nx ny nz dx ox oy oz                          (grid: counts, spacing, origin)
//   sigma0 sigma_grow peak_width p_exp z_entry     (beam model + surface depth)
//   n_spots
//   x0 y0 range weight        (one line per spot, repeated n_spots times)
// Throws std::runtime_error on a malformed/empty file so demos fail loudly.
Plan load_plan(const std::string& path);

// Total number of voxels in a grid (nx*ny*nz), as size_t to avoid int overflow
// on larger grids. Used to size the dose vector and the device buffer.
inline std::size_t voxel_count(const Grid& g) {
    return static_cast<std::size_t>(g.nx) *
           static_cast<std::size_t>(g.ny) *
           static_cast<std::size_t>(g.nz);
}

// Linear index of voxel (i,j,k) in the flat dose array. x is fastest-varying so
// neighbouring-in-x voxels are adjacent in memory -- which is exactly how we map
// threads to voxels on the GPU for coalesced writes (THEORY.md §GPU mapping).
inline std::size_t voxel_index(const Grid& g, int i, int j, int k) {
    return (static_cast<std::size_t>(k) * g.ny + j) * g.nx + i;
}

// CPU reference: fill `dose` (sized nx*ny*nz, x-fastest then y then z) with the
// summed pencil-beam dose at every voxel centre. Serial triple loop over voxels,
// inner loop over spots -- the exact computation the GPU parallelises. Because
// both sides call dose_from_spot() in FP32, summing spots in the SAME order per
// voxel, the results match to a small floating-point tolerance (THEORY.md §7).
void dose_cpu(const Plan& plan, std::vector<float>& dose);
