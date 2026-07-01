// ===========================================================================
// src/reference_cpu.h  --  Problem definition + CPU Monte Carlo reference
// ---------------------------------------------------------------------------
// Project 5.10 : Secondary Cancer Risk & Stray-Dose Monte Carlo
//
// The CPU reference runs the SAME histories as the GPU (same RNG + transport from
// stray_physics.h), so the two dose tallies must be identical to the last bit.
// This header is pure C++ (no CUDA constructs), so kernels.cu can reuse
// StrayProblem directly and both build cleanly.
//
// READ THIS AFTER: stray_physics.h, risk_model.h.  READ NEXT: reference_cpu.cpp,
// kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "stray_physics.h"   // SimParams, Rng, simulate_history (host+device safe)
#include "risk_model.h"      // organ_lar, fixed_to_dose

// One organ slab's metadata: a human-readable name (for the report) and its
// BEIR-VII-style risk coefficient (cases per 10^4 persons per unit dose). The
// physics only needs counts and indices; these travel alongside for the risk
// convolution and the labelled output.
struct Organ {
    std::string name;    // e.g. "Target", "Lung", "Thyroid", ...
    double risk_coeff;   // relative cancer-risk sensitivity (illustrative units)
};

// A complete simulation job: the phantom/beam/VR physics (SimParams), the per-
// organ metadata, and the number of histories. Loaded from the sample file.
struct StrayProblem {
    SimParams sp{};             // phantom + beam + variance-reduction parameters
    std::vector<Organ> organs;  // n_organs entries (names + risk coefficients)
};

// Load a StrayProblem from the committed text format (see data/README.md):
//   line 1 : field_end mu organ_cm scatter_frac sidescatter leakage_frac
//            neutron_frac roulette_floor roulette_survive n_histories seed
//   then one line per organ: "<name> <risk_coeff>"
// n_organs is inferred from the number of organ lines.
StrayProblem load_stray_problem(const std::string& path);

// CPU reference: simulate all n_histories serially, tallying fixed-point stray
// dose per organ. `dose` is sized to n_organs. Because deposits are integer
// (fixed-point), the tally is exact and order-independent -- it must equal the
// GPU's byte-for-byte. Mirrors dose_gpu() in kernels.cu.
void stray_cpu(const StrayProblem& prob, std::vector<unsigned long long>& dose);
