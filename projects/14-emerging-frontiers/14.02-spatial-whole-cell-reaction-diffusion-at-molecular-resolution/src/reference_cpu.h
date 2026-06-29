// ===========================================================================
// src/reference_cpu.h  --  RD parameters, field init, CPU reference
// ---------------------------------------------------------------------------
// Project 14.02 : Spatial / Whole-Cell Reaction-Diffusion (teaching stencil)
//
// Pure C++ (no CUDA). The per-cell update is in rd.h; kernels.cu reuses RdParams.
// The CPU reference runs the identical stencil as the GPU, so the final fields
// must match (within the float-accumulation tolerance discussed in THEORY).
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "rd.h"   // RdParams, rd_update

// Load RdParams from the one-line text format (data/README.md):
//   "nx ny Du Dv F k dt steps seed_half"
RdParams load_rd(const std::string& path);

// Initialize the grid: U = 1, V = 0 everywhere, except a central square of side
// (2*seed_half+1) seeded with U = 0.5, V = 0.25 -- the perturbation that grows
// into patterns. U and V are sized to nx*ny.
void init_fields(const RdParams& P, std::vector<double>& U, std::vector<double>& V);

// CPU reference: advance the fields `steps` timesteps in place (double-buffered
// explicit Euler). The trusted baseline the GPU stencil is checked against.
void simulate_cpu(const RdParams& P, std::vector<double>& U, std::vector<double>& V);
