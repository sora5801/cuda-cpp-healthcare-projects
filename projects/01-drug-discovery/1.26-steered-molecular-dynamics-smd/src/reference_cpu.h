// ===========================================================================
// src/reference_cpu.h  --  SMD config loader + serial CPU reference
// ---------------------------------------------------------------------------
// Project 1.26 : Steered Molecular Dynamics (SMD)
//
// The "ensemble" is a set of n_traj INDEPENDENT constant-velocity SMD pulls of
// the same system, each with its own random thermal history. The config struct
// (SmdParams) and the per-trajectory physics live in smd_core.h, shared with the
// GPU. This header adds the host-only pieces: parse the data file, and run all
// trajectories serially as the trusted baseline the GPU is checked against.
// Pure C++ (no CUDA), so it compiles under cl.exe / g++; kernels.cu reuses
// SmdParams and run_trajectory() too.
//
// READ THIS AFTER: smd_core.h (the physics).  READ BEFORE: kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "smd_core.h"   // SmdParams, run_trajectory, jarzynski_dg, SMD_HD

// Load an SmdParams from the whitespace-separated text format (data/README.md):
//   xi0 xi_end n_traj steps dt k_spring v_pull gamma kT pmf_A pmf_xa pmf_xb
//   pmf_slope seed
// Throws std::runtime_error on a missing file or malformed/invalid values so a
// demo fails loudly instead of silently simulating garbage.
SmdParams load_params(const std::string& path);

// CPU reference: run every trajectory serially and fill `work` (size n_traj)
// with each trajectory's external work W_i. This is the trusted baseline; the
// GPU kernel computes the same vector and main.cu asserts they agree exactly.
// (The Jarzynski reduction over `work` is then done once, identically, by
// jarzynski_dg() in smd_core.h -- see main.cu.)
void run_cpu(const SmdParams& p, std::vector<double>& work);
