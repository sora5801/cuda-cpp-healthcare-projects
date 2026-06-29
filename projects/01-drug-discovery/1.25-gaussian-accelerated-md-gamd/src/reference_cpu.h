// ===========================================================================
// src/reference_cpu.h  --  GaMD config loader + serial CPU reference run
// ---------------------------------------------------------------------------
// Project 1.25 : Gaussian-Accelerated MD (GaMD)   (reduced-scope teaching version)
//
// The GamdConfig struct and ALL the physics (potential, boost, RNG, the per-walker
// loop run_walker(), and the reweighting) live in gamd.h so the host reference and
// the GPU kernel share one source of truth (PATTERNS.md §2). This header only
// declares the two HOST-ONLY entry points:
//   * load_config()      -- parse the text sample into a GamdConfig
//   * run_ensemble_cpu() -- run every walker serially and fill the fixed-point
//                           accumulator array (the trusted baseline the GPU is
//                           checked against, bit-for-bit -- tolerance 0).
//
// Pure C++ (no CUDA types) so it is compiled by the host compiler; kernels.cu and
// main.cu also include it for GamdConfig.
//
// READ THIS AFTER: gamd.h.  READ THIS BEFORE: reference_cpu.cpp, kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <cstdint>   // int64_t accumulators
#include <string>
#include <vector>

#include "gamd.h"    // GamdConfig + all GAMD_HD physics (shared host/device)

// ---------------------------------------------------------------------------
// load_config: read a GamdConfig from the whitespace-separated sample file.
//   Format (one value per field, see data/README.md), read in this order:
//     u_barrier kT gamma_fric dt steps equil_steps
//     e_threshold v_min v_max k0
//     n_walkers x_lo x_hi n_bins seed
//   Throws std::runtime_error on a missing/short/invalid file so demos fail loudly
//   rather than silently running on garbage.
// ---------------------------------------------------------------------------
GamdConfig load_config(const std::string& path);

// ---------------------------------------------------------------------------
// run_ensemble_cpu: the SERIAL reference. Zeroes `acc` (length acc_total(c) =
//   3*n_bins int64), then runs each of c.n_walkers walkers one after another via
//   run_walker() with a plain `acc[i] += v` adder. Because run_walker and the RNG
//   are deterministic and the accumulators are integers, this is the exact answer
//   the GPU must reproduce. Fills `final_x` with each walker's last position (a
//   cheap deterministic cross-check printed by main).
//
//   acc      : OUT, resized to acc_total(c), the fixed-point (count|S1|S2) tally
//   final_x  : OUT, resized to c.n_walkers, each walker's final position
// ---------------------------------------------------------------------------
void run_ensemble_cpu(const GamdConfig& c,
                      std::vector<int64_t>& acc,
                      std::vector<double>& final_x);
