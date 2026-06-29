// ===========================================================================
// src/reference_cpu.h  --  Mesh init + CPU PBD reference
// ---------------------------------------------------------------------------
// Project 10.02 : Real-Time Soft-Tissue Deformation for Surgical Simulation
//
// Pure C++ (no CUDA). The per-particle physics is in pbd.h; kernels.cu reuses
// PbdParams and the mesh. The CPU reference runs the identical PBD steps as the
// GPU, so the final particle positions must match.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "pbd.h"   // Vec3, PbdParams, pbd_predict/correction/new_velocity

// Load PbdParams from the one-line text format (data/README.md):
//   "R C spacing dt gravity stiffness omega iters steps"
PbdParams load_pbd(const std::string& path);

// Build the initial mesh: an R x C grid laid flat in the x-z plane (y = 0),
// spacing apart. The top row (r = 0) is PINNED (inverse mass 0); all others are
// free (inverse mass 1). Velocities start at zero.
void init_mesh(const PbdParams& P, std::vector<Vec3>& x, std::vector<Vec3>& v,
               std::vector<double>& w);

// CPU reference: advance the mesh `steps` timesteps in place (predict ->
// Jacobi constraint projection x iters -> velocity update). Trusted baseline.
void simulate_cpu(const PbdParams& P, std::vector<Vec3>& x, std::vector<Vec3>& v,
                  const std::vector<double>& w);
