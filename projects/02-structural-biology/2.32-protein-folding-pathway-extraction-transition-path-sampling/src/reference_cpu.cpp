// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial Transition-Path-Sampling reference
// ---------------------------------------------------------------------------
// Project 2.32 : Protein Folding Pathway Extraction (Transition Path Sampling)
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- a single readable loop over shooters, no
//   parallelism, no cleverness -- so that when the GPU and CPU agree, we believe
//   the GPU. It runs the EXACT SAME shooting move (run_shot) the GPU runs, just
//   serially and tallying with plain '+=' instead of atomicAdd.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h
//   and tps_physics.h. Compare against kernels.cu (the GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_tps_problem: parse the one-line parameter file (see data/README.md).
// Order: barrier x0 w D dt basin_tol max_steps n_shooters n_bins seed
// We validate that the numbers describe a sane double-well + run so the demo
// fails loudly on a typo instead of silently producing garbage.
// ---------------------------------------------------------------------------
TpsProblem load_tps_problem(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open parameter file: " + path);

    TpsProblem p;
    SimParams& s = p.sp;
    if (!(in >> s.barrier >> s.x0 >> s.w >> s.D >> s.dt >> s.basin_tol
             >> s.max_steps >> s.n_shooters >> s.n_bins >> s.seed)) {
        throw std::runtime_error("bad parameters (expected 'barrier x0 w D dt "
            "basin_tol max_steps n_shooters n_bins seed') in " + path);
    }
    // Sanity checks: a positive barrier and width, a usable timestep, at least
    // one shooter and one histogram bin, and a basin tolerance small enough that
    // the two basins (separated by 2w) do not overlap.
    if (s.barrier <= 0.0 || s.w <= 0.0 || s.D <= 0.0 || s.dt <= 0.0 ||
        s.basin_tol <= 0.0 || s.basin_tol >= s.w ||
        s.max_steps <= 0 || s.n_shooters <= 0 || s.n_bins <= 0) {
        throw std::runtime_error("invalid simulation parameters in " + path);
    }
    return p;
}

// ---------------------------------------------------------------------------
// tps_cpu: run all shooting moves serially and accumulate the integer tally.
//   Complexity: O(n_shooters * max_steps) -- each shooter integrates up to
//   max_steps BD steps per leg, two legs. Fully independent across shooters,
//   which is exactly why the GPU can run one shooter per thread (kernels.cu).
// ---------------------------------------------------------------------------
void tps_cpu(const TpsProblem& prob, TpsTally& tally) {
    const SimParams& P = prob.sp;
    tally.n_transitions = 0;
    tally.n_fwd_to_B    = 0;
    tally.resize(P.n_bins);   // allocate + zero the per-bin committor histograms

    // One iteration per independent shooting move. run_shot() rebuilds shooter
    // i's reproducible RNG stream from (seed, i), so this loop's result depends
    // only on the parameters -- never on iteration order. That order-independence
    // is what lets the GPU's atomicAdd version match this byte for byte.
    for (int i = 0; i < P.n_shooters; ++i) {
        ShotResult r = run_shot(P, i);

        // Scalar tallies: did this shot connect the basins? did it commit to B?
        tally.n_transitions += r.is_transition;   // 0 or 1
        tally.n_fwd_to_B    += r.committed_B;      // 0 or 1

        // Per-bin committor histogram: this shooting point lands in bin sp_bin;
        // record the shot and (if its forward leg reached B) the commitment.
        tally.shots_per_bin[static_cast<std::size_t>(r.sp_bin)]     += 1;
        tally.committed_per_bin[static_cast<std::size_t>(r.sp_bin)] += r.committed_B;
    }
}
