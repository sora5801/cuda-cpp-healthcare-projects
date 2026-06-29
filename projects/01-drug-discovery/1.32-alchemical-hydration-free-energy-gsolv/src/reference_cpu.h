// ===========================================================================
// src/reference_cpu.h  --  Config, solvent bath, CPU reference + TI/BAR analysis
// ---------------------------------------------------------------------------
// Project 1.32 : Alchemical Hydration Free Energy (delta-G_solv)
//
// This header declares everything that is NOT device code:
//   * AlchConfig        -- the whole calculation's settings (windows, walkers, ...)
//   * the solvent bath builder (deterministic synthetic geometry)
//   * the CPU reference that runs every (window, walker) Metropolis chain serially
//   * the TI and BAR post-processors that turn per-window <dU/dlambda> and energy
//     differences into a single delta-G_solv number.
//
//   The actual per-walker physics (energy, dU/dlambda, the Metropolis loop) lives
//   in alchemy.h, shared verbatim with the GPU kernel. kernels.cu reuses
//   AlchConfig and the WalkerResult layout from there. Pure C++ -- safe to
//   #include from a .cu file.
//
// READ THIS AFTER: alchemy.h.  READ BEFORE: reference_cpu.cpp, kernels.cuh.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "alchemy.h"   // SystemParams, SolventBath, WalkerResult, run_walker

// ---------------------------------------------------------------------------
// AlchConfig: the complete description of one delta-G calculation.
//   The lambda schedule is a uniform grid of `n_windows` points from 0 to 1
//   inclusive. At each window we run `n_walkers` independent Metropolis chains
//   (the ensemble we parallelize), each doing n_equil burn-in + n_prod
//   production steps. seed makes the whole thing reproducible.
// ---------------------------------------------------------------------------
struct AlchConfig {
    SystemParams sys{};        // the physical model (LJ params, T, soft-core alpha...)
    int      n_windows = 0;    // number of lambda values (>=2; includes 0 and 1)
    int      n_walkers = 0;    // independent MC chains per window (the ensemble)
    int      n_equil   = 0;    // burn-in steps per walker (discarded)
    int      n_prod    = 0;    // production steps per walker (sampled)
    uint64_t seed      = 0;    // global RNG seed
    unsigned bath_seed = 0;    // seed for the synthetic solvent geometry
};

// Total number of independent walkers = the number of GPU threads we launch.
inline int total_walkers(const AlchConfig& c) { return c.n_windows * c.n_walkers; }

// The lambda value of window `w` on the uniform [0,1] grid.
inline double window_lambda(const AlchConfig& c, int w) {
    return (c.n_windows > 1) ? double(w) / double(c.n_windows - 1) : 1.0;
}

// ---------------------------------------------------------------------------
// Solvent geometry, stored as Structure-of-Arrays (SoA) so the GPU can hand the
// raw pointers to every thread. Built deterministically from bath_seed so the
// sample is reproducible (see scripts/make_synthetic.py for the same recipe).
// ---------------------------------------------------------------------------
struct BathStorage {
    std::vector<double> x, y, z;   // each length n_solvent
    SolventBath view() const {     // a non-owning SolventBath pointing at this storage
        return SolventBath{ x.data(), y.data(), z.data(), int(x.size()) };
    }
};

// Build the synthetic solvent bath: n_solvent sites placed on a jittered shell
// around the origin (so the solute, roaming near the centre, always has solvent
// neighbours). Deterministic given (n_solvent, box, seed).
BathStorage build_bath(const SystemParams& sys, int n_solvent, unsigned seed);

// ---------------------------------------------------------------------------
// Per-window aggregate produced from its walkers. The free-energy estimators
// consume these, not the raw walkers.
// ---------------------------------------------------------------------------
struct WindowStats {
    double lambda;        // this window's coupling
    double mean_dudl;     // <dU/dlambda> averaged over all walkers' samples (TI integrand)
    double mean_du_fwd;   // <U(next)-U(here)> (BAR forward energy difference)
    double mean_du_bwd;   // <U(prev)-U(here)> (BAR backward energy difference)
    double accept_frac;   // fraction of accepted moves (sampling-quality check)
    long   n_samples;     // total production samples behind these means
};

// Reduce a flat array of WalkerResult (one per global walker) into per-window
// statistics. The reduction is a deterministic ordered sum on the host (no
// atomics), so it is reproducible and matches whichever device produced the
// per-walker results.
std::vector<WindowStats> reduce_windows(const AlchConfig& c,
                                        const std::vector<WalkerResult>& walkers);

// TI estimate: delta-G(switch on) = integral_0^1 <dU/dlambda> d-lambda by the
// trapezoidal rule over the window grid. delta-G_solv = -that. Returns G_solv.
double estimate_ti(const AlchConfig& c, const std::vector<WindowStats>& stats);

// BAR estimate: combine adjacent windows with the Bennett Acceptance Ratio (the
// minimum-variance free-energy estimator) and sum the per-pair delta-G. Returns
// G_solv (= -sum of the switch-on free-energy increments).
double estimate_bar(const AlchConfig& c, const std::vector<WindowStats>& stats);

// ---------------------------------------------------------------------------
// I/O + the CPU reference driver.
// ---------------------------------------------------------------------------

// Load an AlchConfig from the one-line text format (see data/README.md):
//   n_solvent box T epsilon sigma q_solute alpha_sc max_step
//   n_windows n_walkers n_equil n_prod seed bath_seed
AlchConfig load_config(const std::string& path);

// CPU reference: run EVERY (window, walker) Metropolis chain serially and fill
// `walkers` (sized total_walkers(c)). This is the trusted baseline the GPU
// kernel is verified against -- identical run_walker() => identical results.
void run_cpu(const AlchConfig& c, const BathStorage& bath,
             std::vector<WalkerResult>& walkers);
