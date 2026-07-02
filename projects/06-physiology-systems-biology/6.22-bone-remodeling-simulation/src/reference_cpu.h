// ===========================================================================
// src/reference_cpu.h  --  Bone-remodeling parameters + CPU reference solver
// ---------------------------------------------------------------------------
// Project 6.22 : Bone Remodeling Simulation   (REDUCED-SCOPE teaching version)
//
// Pure C++ (NO CUDA syntax) so the host compiler can build reference_cpu.cpp and
// so kernels.cu can reuse BoneParams. The actual per-voxel physics lives in the
// shared bone_remodel.h (__host__ __device__) so the CPU reference here and the
// GPU kernels run byte-for-byte identical math (PATTERNS.md section 2).
//
// READ THIS BEFORE: reference_cpu.cpp, main.cu, kernels.cuh.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "bone_remodel.h"   // BoneParams, bone_idx, and the shared HD physics

// ---------------------------------------------------------------------------
// load_bone : parse a BoneParams job from the one-line-per-field sample file.
//   Format (see data/README.md):
//       nx ny remodel_steps relax_iters load load_x0 load_x1 setpoint lazy rate
//       rho_min rho_init
//   Throws std::runtime_error on a missing file or nonsensical values so demos
//   fail loudly instead of silently running on garbage.
// ---------------------------------------------------------------------------
BoneParams load_bone(const std::string& path);

// ---------------------------------------------------------------------------
// bone_cpu : the trusted serial reference. Starting from a uniform density
//   rho_init, it runs `remodel_steps` remodeling iterations. Each iteration:
//     (1) relax the mechanical-stimulus field S with `relax_iters` Jacobi
//         sweeps (ping-ponging two S buffers), then
//     (2) apply the mechanostat rule to update every voxel's density rho.
//   On return, `rho_final` (size nx*ny) holds the remodeled density field and
//   `S_final` (size nx*ny) holds the last settled stimulus field (used for the
//   state histogram in the report). This is the baseline the GPU is checked
//   against. Written as plain nested loops -- obvious over clever.
// ---------------------------------------------------------------------------
void bone_cpu(const BoneParams& p,
              std::vector<double>& rho_final,
              std::vector<double>& S_final);

// ---------------------------------------------------------------------------
// bone_summary : reduce a density field to the deterministic scalars the report
//   prints -- total bone mass (sum of rho) and per-column mass (sum down each
//   column). Shared by both paths so the comparison is apples-to-apples.
//     rho        : density field (size nx*ny)
//     total_mass : out, sum of all rho
//     col_mass   : out, size nx, sum of rho over each column x
// ---------------------------------------------------------------------------
void bone_summary(const BoneParams& p, const std::vector<double>& rho,
                  double& total_mass, std::vector<double>& col_mass);
