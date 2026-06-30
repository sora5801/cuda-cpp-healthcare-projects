// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial SA-ensemble reference
// ---------------------------------------------------------------------------
// Project 2.18 : NMR Structure Refinement
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. It is written to be
//   OBVIOUSLY correct -- a single readable loop over replicas, no parallelism --
//   so that when the GPU and CPU agree, we believe the GPU. The per-replica
//   annealer it calls (anneal_one) is the EXACT same shared code the kernel runs,
//   which is why a given replica yields identical numbers on both sides.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, nmr_refine.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// load_config: parse the tiny text job file into a RefineConfig.
//   Format (whitespace-separated; see data/README.md):
//     n_beads n_restraints bond_len k_bond k_noe
//     n_replicas n_steps T_hot T_cold step_sigma base_seed
//     i j upper           (repeated n_restraints times)
//   We validate every size against the NMR_MAX_* caps so a bad file cannot make
//   the kernel index past its fixed-size arrays.
// ---------------------------------------------------------------------------
RefineConfig load_config(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open config file: " + path);

    RefineConfig c{};   // zero-initialise; restr[] entries default to {0,0,0}

    // Line 1: chain + force-field constants.
    if (!(in >> c.n_beads >> c.n_restraints >> c.bond_len >> c.k_bond >> c.k_noe))
        throw std::runtime_error("bad line 1 (expected 'n_beads n_restraints "
                                 "bond_len k_bond k_noe') in " + path);

    // Line 2: ensemble + annealing schedule.
    if (!(in >> c.n_replicas >> c.n_steps >> c.T_hot >> c.T_cold
             >> c.step_sigma >> c.base_seed))
        throw std::runtime_error("bad line 2 (expected 'n_replicas n_steps T_hot "
                                 "T_cold step_sigma base_seed') in " + path);

    // Bounds checks: keep everything inside the fixed-size device arrays.
    if (c.n_beads < 2 || c.n_beads > NMR_MAX_BEADS)
        throw std::runtime_error("n_beads out of range [2, NMR_MAX_BEADS] in " + path);
    if (c.n_restraints < 0 || c.n_restraints > NMR_MAX_RESTRAINTS)
        throw std::runtime_error("n_restraints out of range [0, NMR_MAX_RESTRAINTS] in " + path);
    if (c.n_replicas < 1 || c.n_steps < 1)
        throw std::runtime_error("n_replicas and n_steps must be >= 1 in " + path);
    if (c.T_hot <= 0.0 || c.T_cold <= 0.0 || c.T_cold > c.T_hot)
        throw std::runtime_error("need 0 < T_cold <= T_hot in " + path);

    // The restraint list.
    for (int q = 0; q < c.n_restraints; ++q) {
        Restraint& R = c.restr[q];
        if (!(in >> R.i >> R.j >> R.upper))
            throw std::runtime_error("bad restraint line (expected 'i j upper') in " + path);
        if (R.i < 0 || R.i >= c.n_beads || R.j < 0 || R.j >= c.n_beads || R.i == R.j)
            throw std::runtime_error("restraint indexes a non-existent or self bead in " + path);
        if (R.upper <= 0.0)
            throw std::runtime_error("restraint upper bound must be > 0 in " + path);
    }
    return c;
}

// ---------------------------------------------------------------------------
// anneal_ensemble_cpu: the serial reference over the whole ensemble.
//   One replica after another, each an INDEPENDENT SA trajectory (it is exactly
//   this independence that lets the GPU give every replica its own thread). The
//   two scratch arrays (current + best structure) are allocated once and reused.
// ---------------------------------------------------------------------------
void anneal_ensemble_cpu(const RefineConfig& c, std::vector<ReplicaResult>& results) {
    results.assign(static_cast<std::size_t>(c.n_replicas), ReplicaResult{});

    // Per-replica working coordinates: x = current trial, xbest = best so far.
    // 3 doubles per bead (x,y,z). Allocated here (not per replica) to mirror the
    // kernel, which keeps these in per-thread local memory.
    std::vector<double> x(static_cast<std::size_t>(3 * c.n_beads));
    std::vector<double> xbest(static_cast<std::size_t>(3 * c.n_beads));

    for (int r = 0; r < c.n_replicas; ++r) {
        // anneal_one() is the SHARED host+device annealer (nmr_refine.h). The CPU
        // calls it in a plain loop; the GPU calls it from one thread per replica.
        results[static_cast<std::size_t>(r)] =
            anneal_one(c, static_cast<uint64_t>(r), x.data(), xbest.data());
    }
}
