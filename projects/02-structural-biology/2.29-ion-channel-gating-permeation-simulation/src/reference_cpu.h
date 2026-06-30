// ===========================================================================
// src/reference_cpu.h  --  Problem definition + CPU Brownian-dynamics reference
// ---------------------------------------------------------------------------
// Project 2.29 : Ion Channel Gating & Permeation Simulation
//
// The CPU reference runs the SAME ion trajectories as the GPU (same RNG, same
// per-step physics from channel_physics.h), so the two tallies -- the occupancy
// histogram and the forward/reverse crossing counts -- must be identical. This
// header is pure C++ (no CUDA); kernels.cu reuses PermeationProblem.
//
// Read this after channel_physics.h (the shared physics) and before main.cu.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "channel_physics.h"   // ChannelParams, Rng, bd_step, bin_of (host+device)

// ---------------------------------------------------------------------------
// PermeationProblem: a complete simulation job -- the channel/protocol physics
// plus how many independent ions to launch and the base RNG seed.
//   `n_ions` independent Brownian walkers are the parallel work: one GPU thread
//   per ion (PATTERNS.md §1, "stochastic / Monte-Carlo histories"). Ion i uses
//   the reproducible stream (seed, i), so CPU and GPU simulate the same ions.
// ---------------------------------------------------------------------------
struct PermeationProblem {
    ChannelParams cp{};                // pore + bath + voltage-clamp parameters
    unsigned long long n_ions = 0;     // number of independent ion trajectories
    uint64_t seed = 0;                 // base RNG seed (ion i uses stream (seed,i))
};

// ---------------------------------------------------------------------------
// PermeationResult: everything we report and verify. All integer, so GPU==CPU
// is an EXACT equality (PATTERNS.md §4) -- no floating-point tolerance needed.
// ---------------------------------------------------------------------------
struct PermeationResult {
    std::vector<unsigned long long> occupancy;  // [n_bins] ion-steps per z-bin
    unsigned long long fwd = 0;   // total forward permeations (current-carrying)
    unsigned long long rev = 0;   // total reverse permeations
};

// Load a PermeationProblem from the one-line text format (see data/README.md):
//   "L n_bins U_barrier sigma q V D dt n_steps n_ions seed"
PermeationProblem load_permeation_problem(const std::string& path);

// CPU reference: simulate all n_ions trajectories serially and tally integer
// occupancy + crossing counts. Because every observable is an integer, the tally
// is exact and order-independent -- it must equal the GPU's bit-for-bit.
void permeation_cpu(const PermeationProblem& prob, PermeationResult& out);

// Single-channel conductance proxy (a derived, reported number): the NET forward
// crossings per ion per step. This is the teaching link "crossings -> current ->
// conductance". Defined inline here because both main.cu and tests may want it.
inline double net_flux_per_ion_step(const PermeationResult& r,
                                    const PermeationProblem& p) {
    const double denom = static_cast<double>(p.n_ions) *
                         static_cast<double>(p.cp.n_steps);
    if (denom <= 0.0) return 0.0;
    return (static_cast<double>(r.fwd) - static_cast<double>(r.rev)) / denom;
}
