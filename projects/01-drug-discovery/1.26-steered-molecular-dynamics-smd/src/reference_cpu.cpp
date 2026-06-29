// ===========================================================================
// src/reference_cpu.cpp  --  Config loader + serial CPU reference (ground truth)
// ---------------------------------------------------------------------------
// Project 1.26 : Steered Molecular Dynamics (SMD)
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- a single readable loop over trajectories, no
//   parallelism, no cleverness -- so that when the GPU and CPU agree (here:
//   EXACTLY, because both call the same run_trajectory() in smd_core.h), we
//   believe the GPU.
//
//   Compiled by the host C++ compiler only (no CUDA here). The per-trajectory
//   physics lives in smd_core.h; this file just loads the config and loops.
//
// READ THIS AFTER: smd_core.h, reference_cpu.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <stdexcept>

// Parse the 14-field whitespace-separated parameter file (data/README.md gives
// the field meanings). We read into a temporary and validate before returning,
// so a truncated or nonsensical file throws rather than producing silent junk.
SmdParams load_params(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open SMD parameter file: " + path);

    SmdParams p{};
    if (!(in >> p.xi0 >> p.xi_end >> p.n_traj >> p.steps >> p.dt
             >> p.k_spring >> p.v_pull >> p.gamma >> p.kT
             >> p.pmf_A >> p.pmf_xa >> p.pmf_xb >> p.pmf_slope >> p.seed)) {
        throw std::runtime_error(
            "bad parameters (expected 'xi0 xi_end n_traj steps dt k_spring "
            "v_pull gamma kT pmf_A pmf_xa pmf_xb pmf_slope seed') in " + path);
    }
    // Guard the values the physics divides by or loops on, so a bad file cannot
    // produce a divide-by-zero, an empty ensemble, or an infinite pull.
    if (p.n_traj <= 0 || p.steps <= 0 || p.dt <= 0.0 ||
        p.gamma <= 0.0 || p.kT <= 0.0 || p.k_spring <= 0.0 ||
        p.pmf_xb == p.pmf_xa) {
        throw std::runtime_error("invalid SMD parameters (non-positive count/dt/"
                                 "gamma/kT/k, or degenerate PMF) in " + path);
    }
    return p;
}

// CPU reference: run every trajectory serially.
//   Each trajectory is an INDEPENDENT stochastic SMD pull -> a plain for-loop
//   here, but one GPU thread per trajectory in kernels.cu. We seed trajectory i
//   from (seed, i) inside run_trajectory(), so the i-th CPU work equals the i-th
//   GPU work bit-for-bit -- that is what makes the verification exact.
//   Complexity: O(n_traj * steps) time, O(n_traj) output space.
void run_cpu(const SmdParams& p, std::vector<double>& work) {
    work.assign(static_cast<std::size_t>(p.n_traj), 0.0);
    for (int i = 0; i < p.n_traj; ++i) {
        work[static_cast<std::size_t>(i)] =
            run_trajectory(p, static_cast<uint64_t>(i));
    }
}
