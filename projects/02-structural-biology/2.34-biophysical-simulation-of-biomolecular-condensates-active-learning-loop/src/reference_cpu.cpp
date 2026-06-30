// ===========================================================================
// src/reference_cpu.cpp  --  Config loader, serial ensemble, AL acquisition
// ---------------------------------------------------------------------------
// Project 2.34 : Biophysical Simulation of Biomolecular Condensates
//                (Active Learning Loop)  --  reduced-scope teaching version
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It runs every
//   replica through the SAME shared integrator (condensate.h) the GPU kernel
//   uses, just serially -- so when the GPU and CPU agree (within a small float
//   tolerance) we believe the GPU. It also implements the deterministic active-
//   learning step (acquisition + proposal), which runs once on the host after
//   the ensemble finishes regardless of where the ensemble ran.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, condensate.h. Compare with kernels.cu (twin).
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>       // std::fabs
#include <fstream>     // std::ifstream
#include <limits>      // std::numeric_limits
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// load_ensemble: parse the one-line whitespace-separated config. The field
// order matches data/README.md exactly; mismatched/short files throw so a demo
// fails loudly instead of running on garbage.
// ---------------------------------------------------------------------------
EnsembleConfig load_ensemble(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open ensemble file: " + path);

    EnsembleConfig c;
    // First the CG-MD model constants, then the active-learning sweep fields.
    // 'seed' is read as an unsigned via a temporary long long to stay portable.
    long long seed_tmp = 0;
    if (!(in >> c.model.n_beads >> c.model.steps >> c.model.dt >> c.model.kT
             >> c.model.gamma >> c.model.k_bond >> c.model.r0
             >> c.model.eq_steps >> c.model.lag >> seed_tmp
             >> c.n_members >> c.lambda_lo >> c.lambda_hi
             >> c.k_cohese >> c.target_D)) {
        throw std::runtime_error(
            "bad parameters (expected 'n_beads steps dt kT gamma k_bond r0 "
            "eq_steps lag seed n_members lambda_lo lambda_hi k_cohese target_D') in "
            + path);
    }
    c.model.seed = static_cast<std::uint32_t>(seed_tmp);

    // Sanity checks: every count must be positive and the chain must fit the
    // fixed-size local arrays the kernel uses (CND_MAX_BEADS in condensate.h).
    if (c.model.n_beads <= 0 || c.model.n_beads > CND_MAX_BEADS)
        throw std::runtime_error("n_beads must be in [1, 32] in " + path);
    if (c.model.steps <= 0 || c.model.eq_steps < 0 || c.model.eq_steps >= c.model.steps)
        throw std::runtime_error("need 0 <= eq_steps < steps in " + path);
    if (c.model.lag <= 0 || c.model.lag > CND_MAX_LAG
        || c.model.lag >= c.model.steps - c.model.eq_steps)
        throw std::runtime_error("need 0 < lag <= 24 and lag < (steps-eq_steps) in " + path);
    if (c.model.dt <= 0 || c.model.gamma <= 0 || c.model.kT < 0)
        throw std::runtime_error("dt, gamma must be > 0 and kT >= 0 in " + path);
    if (c.n_members <= 0)
        throw std::runtime_error("n_members must be > 0 in " + path);
    return c;
}

// ---------------------------------------------------------------------------
// integrate_cpu: the serial reference. One readable loop over members, each an
// INDEPENDENT trajectory -> the structure that becomes "one GPU thread per
// member" in kernels.cu. Uses integrate_replica from condensate.h so the math
// is byte-for-byte the same code the GPU runs.
// ---------------------------------------------------------------------------
void integrate_cpu(const EnsembleConfig& c, std::vector<ReplicaResult>& results) {
    const int M = ensemble_size(c);
    results.assign(static_cast<std::size_t>(M), ReplicaResult{});
    for (int m = 0; m < M; ++m) {
        const double lam = member_lambda(c, m);     // this candidate's stickiness
        results[static_cast<std::size_t>(m)] =
            integrate_replica(c.model, m, lam, c.k_cohese);
    }
}

// ---------------------------------------------------------------------------
// acquisition_score: residual to the experimental target. SMALLER is better.
//   A full Bayesian-optimization acquisition would add an uncertainty term from
//   the GNN surrogate; with the ensemble itself as the surrogate the predictive
//   variance is ~0 at sampled points, so the score collapses to exploitation
//   only: how close this candidate's measured D is to the target. (THEORY shows
//   where the uncertainty term would re-enter in the full loop.)
// ---------------------------------------------------------------------------
double acquisition_score(double measured_D, double target_D) {
    return std::fabs(measured_D - target_D);
}

// ---------------------------------------------------------------------------
// propose_next_lambda: deterministic argmin of acquisition_score over the
// ensemble. Returns the winning lambda and (via *best_member) its member index.
// Ties break toward the lowest member index because we use a strict '<' update,
// keeping the proposal reproducible.
// ---------------------------------------------------------------------------
double propose_next_lambda(const EnsembleConfig& c,
                           const std::vector<ReplicaResult>& results,
                           int* best_member) {
    const int M = ensemble_size(c);
    int    best_m = 0;
    double best_s = std::numeric_limits<double>::infinity();
    for (int m = 0; m < M; ++m) {
        const double s = acquisition_score(results[static_cast<std::size_t>(m)].diffusion,
                                           c.target_D);
        if (s < best_s) { best_s = s; best_m = m; }   // strict '<' -> first wins ties
    }
    if (best_member) *best_member = best_m;
    return member_lambda(c, best_m);
}
