// ===========================================================================
// src/reference_cpu.h  --  Problem definition + CPU Transition-Path-Sampling
//                          reference (the trusted serial baseline).
// ---------------------------------------------------------------------------
// Project 2.32 : Protein Folding Pathway Extraction (Transition Path Sampling)
//
// WHY A SEPARATE PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the plain host C++ compiler and must NOT
//   see any CUDA/__global__ syntax, so its prototypes cannot live in kernels.cuh.
//   Both main.cu and reference_cpu.cpp include THIS header so they agree on the
//   data structures and the function signature. The actual per-shooter physics
//   is in tps_physics.h, which IS host+device safe (see PATTERNS.md §2).
//
// THE VERIFICATION CONTRACT
//   The CPU reference runs the SAME shooting moves as the GPU (same RNG, same
//   Brownian dynamics from tps_physics.h), so the two integer tallies must be
//   IDENTICAL. Integer tallies (not floating sums) are what make atomics on the
//   GPU order-independent, so GPU == CPU exactly (PATTERNS.md §3).
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tps_physics.h"   // SimParams, run_shot, ShotResult (host+device safe)

// A complete TPS job. SimParams already carries n_shooters / n_bins / seed, so
// TpsProblem is a thin wrapper kept for symmetry with the other repo projects
// (whose problems hold more than just parameters).
struct TpsProblem {
    SimParams sp{};   // double-well + BD + shooting parameters (see tps_physics.h)
};

// The RESULT of a TPS run, as INTEGER tallies so CPU and GPU agree EXACTLY
// (floating-point sums reorder under atomics and would diverge -- PATTERNS.md §3):
//   * n_transitions       : how many shots were accepted as transition paths.
//   * n_fwd_to_B          : how many forward legs committed to the folded basin B.
//   * shots_per_bin[b]    : number of shooting points whose x fell in bin b.
//   * committed_per_bin[b]: of those, how many forward legs committed to B.
// The committor estimate p_B(bin) = committed_per_bin[b] / shots_per_bin[b] is
// derived from these integers AFTER the run (kept OUT of the tally so the tally
// itself stays exact-integer and order-independent).
struct TpsTally {
    long long n_transitions = 0;              // accepted transition paths
    long long n_fwd_to_B    = 0;              // forward legs reaching basin B
    std::vector<long long> shots_per_bin;     // [n_bins] shooting-point counts
    std::vector<long long> committed_per_bin; // [n_bins] forward->B counts

    // Allocate and zero the per-bin histograms for n_bins committor bins.
    void resize(int n_bins) {
        shots_per_bin.assign(static_cast<std::size_t>(n_bins), 0);
        committed_per_bin.assign(static_cast<std::size_t>(n_bins), 0);
    }
};

// Load a TpsProblem from the whitespace-separated text format (data/README.md):
//   "barrier x0 w D dt basin_tol max_steps n_shooters n_bins seed"
// Throws std::runtime_error on a missing file or malformed/invalid parameters.
TpsProblem load_tps_problem(const std::string& path);

// CPU reference: run all n_shooters shooting moves SERIALLY and accumulate the
// integer tally. Every shooter's RNG stream is fixed by (seed, shooter) and the
// tally is integer, so this result is exactly what the GPU must reproduce.
//   prob  : the loaded problem (parameters).
//   tally : output; resized to n_bins and filled with the integer counts.
void tps_cpu(const TpsProblem& prob, TpsTally& tally);
