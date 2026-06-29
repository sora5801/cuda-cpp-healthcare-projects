// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial Monte Carlo dose reference
// ---------------------------------------------------------------------------
// Project 5.01 : Monte Carlo Dose Calculation (simplified slab)
// Compiled by the host C++ compiler only. See reference_cpu.h / mc_physics.h.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <stdexcept>

DoseProblem load_dose_problem(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open parameter file: " + path);
    DoseProblem p;
    if (!(in >> p.sp.L >> p.sp.n_bins >> p.sp.mu >> p.sp.p_abs
             >> p.sp.E0 >> p.sp.scatter_dep >> p.n_photons >> p.seed))
        throw std::runtime_error("bad parameters (expected "
            "'L n_bins mu p_abs E0 scatter_dep n_photons seed') in " + path);
    if (p.sp.n_bins <= 0 || p.sp.L <= 0 || p.sp.mu <= 0 || p.n_photons == 0)
        throw std::runtime_error("invalid simulation parameters in " + path);
    return p;
}

void dose_cpu(const DoseProblem& prob, std::vector<unsigned long long>& dose) {
    dose.assign(prob.sp.n_bins, 0ULL);
    int bins[MC_MAX_DEPOSITS];
    unsigned long long amts[MC_MAX_DEPOSITS];
    // Loop over every history. Each gets its own reproducible RNG stream, runs
    // the shared transport, and adds its deposits to the tally. This is exactly
    // what the GPU does -- just serially and with plain '+=' instead of atomics.
    for (unsigned long long i = 0; i < prob.n_photons; ++i) {
        Rng rng = rng_seed(prob.seed, i);
        const int nd = simulate_photon(prob.sp, rng, bins, amts);
        for (int d = 0; d < nd; ++d)
            dose[bins[d]] += amts[d];
    }
}
