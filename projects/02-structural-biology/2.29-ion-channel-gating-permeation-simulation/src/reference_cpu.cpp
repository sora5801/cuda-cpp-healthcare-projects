// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial Brownian-dynamics reference
// ---------------------------------------------------------------------------
// Project 2.29 : Ion Channel Gating & Permeation Simulation
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. It is written to be
//   OBVIOUSLY correct -- a single readable double loop (over ions, over steps),
//   no parallelism, no cleverness -- so that when the GPU and CPU agree we
//   believe the GPU. It calls the EXACT SAME per-step physics (bd_step) and the
//   EXACT SAME RNG as the kernel, so the two integer tallies must be identical.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h
//   and channel_physics.h. Compare against kernels.cu (the GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_permeation_problem: parse the one-line whitespace-separated sample file.
//   Format (see data/README.md):
//     L n_bins U_barrier sigma q V D dt n_steps n_ions seed
//   We validate the few invariants that would otherwise produce nonsense (a
//   non-positive pore length, zero bins, zero ions) so the demo fails loudly
//   with a clear message instead of silently dividing by zero later.
// ---------------------------------------------------------------------------
PermeationProblem load_permeation_problem(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open parameter file: " + path);
    PermeationProblem p;
    if (!(in >> p.cp.L >> p.cp.n_bins >> p.cp.U_barrier >> p.cp.sigma
             >> p.cp.q >> p.cp.V >> p.cp.D >> p.cp.dt
             >> p.cp.n_steps >> p.n_ions >> p.seed))
        throw std::runtime_error("bad parameters (expected 'L n_bins U_barrier "
            "sigma q V D dt n_steps n_ions seed') in " + path);
    if (p.cp.L <= 0.0 || p.cp.n_bins <= 0 || p.cp.sigma <= 0.0 ||
        p.cp.D <= 0.0 || p.cp.dt <= 0.0 || p.cp.n_steps <= 0 || p.n_ions == 0)
        throw std::runtime_error("invalid simulation parameters in " + path);
    return p;
}

// ---------------------------------------------------------------------------
// permeation_cpu: simulate every ion serially and accumulate the tallies.
//   Complexity: O(n_ions * n_steps) time, O(n_bins) extra space. This is the
//   serial baseline whose wall time (timed in main.cu) we compare with the GPU
//   kernel. Every ion is independent -- which is precisely WHY this maps onto one
//   GPU thread per ion in kernels.cu.
//
//   The accumulation is plain integer '+=' here; the GPU uses atomicAdd. Because
//   the tallied quantities are integers, the two are order-independent and equal.
// ---------------------------------------------------------------------------
void permeation_cpu(const PermeationProblem& prob, PermeationResult& out) {
    const ChannelParams& P = prob.cp;
    out.occupancy.assign(static_cast<std::size_t>(P.n_bins), 0ULL);
    out.fwd = 0;
    out.rev = 0;

    for (unsigned long long i = 0; i < prob.n_ions; ++i) {
        // Each ion gets its own reproducible RNG stream from its index, so the
        // GPU (which seeds the same way from the same index) draws the same
        // random numbers -> the same trajectory -> the same tally.
        Rng rng = rng_seed(prob.seed, i);

        // Start every ion at the intracellular mouth (z = 0). A real simulation
        // would draw the start from the bath; starting at 0 makes the demo's
        // result reproducible and the physics easy to reason about.
        double z = 0.0;

        unsigned long long fwd = 0, rev = 0;   // this ion's crossing counters
        for (int s = 0; s < P.n_steps; ++s) {
            // Advance one Brownian step (shared physics). bd_step updates fwd/rev
            // when the ion permeates and re-injects it from the opposite bath.
            z = bd_step(P, rng, z, &fwd, &rev);
            // Record where the ion now sits: the occupancy histogram is the
            // probability density along the pore axis (the BD analogue of the
            // ion-position histogram named in the catalog's GPU pattern).
            out.occupancy[static_cast<std::size_t>(bin_of(P, z))] += 1ULL;
        }
        out.fwd += fwd;   // fold this ion's crossings into the global totals
        out.rev += rev;
    }
}
