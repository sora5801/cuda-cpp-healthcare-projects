// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial Monte Carlo BNCT dose reference
// ---------------------------------------------------------------------------
// Project 5.13 : BNCT Dose Calculation & Optimization (reduced-scope teaching MC)
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- a single serial loop over histories, no
//   parallelism, no cleverness -- so that when the GPU and CPU agree bin-for-bin
//   we trust the GPU. It runs the IDENTICAL histories the kernel runs, because
//   both call the shared simulate_neutron() from bnct_physics.h.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, bnct_physics.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// slurp_noncomment: read the file, drop full-line comments (lines whose first
// non-space character is '#'), and return the rest as one whitespace-separated
// stream. This lets the sample file carry a self-documenting '#' header while
// the parser below just reads 15 numbers in order.
// ---------------------------------------------------------------------------
static std::string slurp_noncomment(std::ifstream& in) {
    std::ostringstream body;
    std::string line;
    while (std::getline(in, line)) {
        std::size_t i = line.find_first_not_of(" \t\r\n");
        if (i == std::string::npos || line[i] == '#') continue;  // blank/comment
        body << line << ' ';
    }
    return body.str();
}

// ---------------------------------------------------------------------------
// load_bnct_problem: parse the whitespace-separated sample file into a
// BnctProblem. The field order is fixed and documented in data/README.md and in
// reference_cpu.h. We validate the physically meaningful invariants so a bad
// file fails loudly instead of producing silent garbage.
// ---------------------------------------------------------------------------
BnctProblem load_bnct_problem(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open parameter file: " + path);
    std::istringstream in(slurp_noncomment(file));   // comment-stripped numbers
    BnctProblem p;
    SimParams& s = p.sp;
    if (!(in >> s.L >> s.n_bins
             >> s.Sig_s_fast >> s.p_thermalize
             >> s.Sig_a_B >> s.Sig_a_N >> s.Sig_a_H >> s.Sig_s_th
             >> s.Q_boron_keV >> s.Q_nitro_keV >> s.Q_gamma_keV >> s.Q_fast_keV
             >> p.n_histories >> p.seed >> p.gray_per_keV))
        throw std::runtime_error("bad parameters (expected 15 fields: "
            "'L n_bins Sig_s_fast p_thermalize Sig_a_B Sig_a_N Sig_a_H Sig_s_th "
            "Q_boron_keV Q_nitro_keV Q_gamma_keV Q_fast_keV n_histories seed "
            "gray_per_keV') in " + path);
    // Physical sanity: positive geometry, valid probability, some capture path.
    if (s.n_bins <= 0 || s.L <= 0.0 || s.Sig_s_fast <= 0.0)
        throw std::runtime_error("invalid slab/fast parameters in " + path);
    if (s.p_thermalize < 0.0 || s.p_thermalize > 1.0)
        throw std::runtime_error("p_thermalize must be in [0,1] in " + path);
    if (s.Sig_a_B < 0.0 || s.Sig_a_N < 0.0 || s.Sig_a_H < 0.0
        || (s.Sig_a_B + s.Sig_a_N + s.Sig_a_H) <= 0.0)
        throw std::runtime_error("thermal capture cross sections invalid in " + path);
    if (p.n_histories == 0)
        throw std::runtime_error("n_histories must be > 0 in " + path);
    return p;
}

// ---------------------------------------------------------------------------
// dose_cpu: serial Monte Carlo. Loop over every history; each gets its own
// reproducible RNG stream (seed, i), runs the shared transport, and its
// deposits are added into the per-component integer tally. This is EXACTLY what
// the GPU does -- just serially and with plain '+=' instead of atomicAdd.
//   Complexity: O(n_histories * avg_steps) time, O(DC_COUNT * n_bins) memory.
// ---------------------------------------------------------------------------
void dose_cpu(const BnctProblem& prob, DoseTally& t) {
    t.reset(prob.sp.n_bins);
    Deposit dep[BNCT_MAX_DEPOSITS];   // caller-owned scratch for one history

    for (unsigned long long i = 0; i < prob.n_histories; ++i) {
        Rng rng = rng_seed(prob.seed, i);                 // history i's stream
        const int nd = simulate_neutron(prob.sp, rng, dep);
        for (int d = 0; d < nd; ++d)
            t.dose[dep[d].component][dep[d].bin] += dep[d].keV;   // integer +=
    }
}
