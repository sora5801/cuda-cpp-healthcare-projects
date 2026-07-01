// ===========================================================================
// src/reference_cpu.h  --  Problem definition + CPU track-structure reference
// ---------------------------------------------------------------------------
// Project 5.11 : Microdosimetry & Track-Structure Simulation
//
// The CPU reference runs the SAME tracks as the GPU (same RNG, same transport
// from ts_physics.h), so their tallies must be identical to the bit. This header
// is pure C++ (no CUDA constructs) so it compiles under the host compiler AND is
// safely #included by kernels.cu / main.cu. kernels.cuh reuses TrackProblem.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ts_physics.h"   // TrackParams, Rng, simulate_track (host+device safe)

// The aggregated tallies over ALL tracks. Everything is an integer count, so the
// CPU sum and the GPU atomic sum are order-independent and must agree exactly.
struct TrackTally {
    unsigned long long total_quanta = 0;  // total energy quanta over all tracks
    unsigned long long total_ssb    = 0;  // total single-strand breaks
    unsigned long long total_dsb    = 0;  // total double-strand breaks
    std::vector<unsigned long long> y_hist;  // lineal-energy histogram [n_y_bins]
};

// A complete simulation job: the microscopic physics plus how many tracks to run.
struct TrackProblem {
    TrackParams tp{};                   // box geometry + interaction parameters
    unsigned long long n_tracks = 0;    // number of primary particle tracks
    uint64_t seed = 0;                  // base RNG seed (track i uses stream (seed,i))
};

// Load a TrackProblem from the one-line text format (see data/README.md):
//   "box_nm sigma_ion let_spread quantum_eV quanta_per_ion p_delta delta_quanta
//    dna_radius_nm n_dna_segments n_y_bins y_max_keV_um n_tracks seed"
TrackProblem load_track_problem(const std::string& path);

// CPU reference: simulate all n_tracks serially and accumulate integer tallies.
// Because every per-track quantity is an integer, the accumulation is exact and
// order-independent -- it MUST equal the GPU's atomic tally.
void track_cpu(const TrackProblem& prob, TrackTally& tally);
