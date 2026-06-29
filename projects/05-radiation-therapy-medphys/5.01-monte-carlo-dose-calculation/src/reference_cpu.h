// ===========================================================================
// src/reference_cpu.h  --  Problem definition + CPU Monte Carlo reference
// ---------------------------------------------------------------------------
// Project 5.01 : Monte Carlo Dose Calculation (simplified slab)
//
// The CPU reference runs the SAME histories as the GPU (same RNG, same
// transport from mc_physics.h), so the two dose tallies must be identical. This
// header is pure C++ (no CUDA); kernels.cu reuses DoseProblem.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mc_physics.h"   // SimParams, Rng, simulate_photon (host+device safe)

// A complete simulation job: the slab physics plus how many histories to run.
struct DoseProblem {
    SimParams sp{};                 // slab + interaction parameters
    unsigned long long n_photons = 0;   // number of photon histories to simulate
    uint64_t seed = 0;              // base RNG seed (history i uses stream (seed,i))
};

// Load a DoseProblem from the one-line text format (data/README.md):
//   "L n_bins mu p_abs E0 scatter_dep n_photons seed"
DoseProblem load_dose_problem(const std::string& path);

// CPU reference: simulate all n_photons histories serially and tally integer
// dose per depth bin. `dose` is sized to n_bins. Because energy is integer
// quanta, the tally is exact and order-independent -- it must equal the GPU's.
void dose_cpu(const DoseProblem& prob, std::vector<unsigned long long>& dose);
