// ===========================================================================
// src/reference_cpu.h  --  Problem definition + CPU (serial) SC dose reference
// ---------------------------------------------------------------------------
// Project 5.4 : Collapsed-Cone / Superposition-Convolution Dose  (2-D teaching model)
//
// The CPU reference runs the SAME arithmetic as the GPU kernels (both call the
// shared per-voxel physics in ccc_physics.h), so the two dose grids must be
// IDENTICAL down to the last integer dose-unit. This header is pure C++ (no
// CUDA); kernels.cu / main.cu reuse DoseProblem and the loader.
//
// Read this AFTER ccc_physics.h (the physics) and BEFORE reference_cpu.cpp.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ccc_physics.h"   // CccParams + all per-voxel formulas (host+device safe)

// A complete dose-calculation job: the geometry/physics plus the per-voxel
// density map (the "CT"). rho is row-major, length nx*ny, relative to water.
struct DoseProblem {
    CccParams          P{};    // geometry + beam + kernel parameters
    std::vector<float> rho;    // density map rho[y*nx + x], water-relative (>=0)
};

// Load a DoseProblem from the committed text sample (layout documented in
// data/README.md). Throws std::runtime_error on any malformed / missing field so
// the demo fails loudly instead of silently computing garbage.
DoseProblem load_dose_problem(const std::string& path);

// ---- The two computation stages, exposed so main.cu can time them and so the
//      GPU twins in kernels.cu can be checked stage-by-stage. -----------------

// STAGE 1 (CPU): ray-trace TERMA down each beam column into `terma` (size nx*ny,
//   0 outside the beam columns). One vertical Siddon march per irradiated column.
void terma_cpu(const DoseProblem& prob, std::vector<double>& terma);

// STAGE 2 (CPU): collapse-cone superpose `terma` into the integer dose grid
//   `dose_units` (size nx*ny). Each source voxel spreads its TERMA along all
//   n_cones cone rays with density-scaled exponential decay; every deposit is
//   quantized to integer dose-units (dose_to_units) so the result is exact and
//   order-independent -- identical to the GPU's atomicAdd tally.
void dose_cpu(const DoseProblem& prob, const std::vector<double>& terma,
              std::vector<long long>& dose_units);
