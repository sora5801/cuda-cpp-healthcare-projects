// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial Monte Carlo stray-dose reference
// ---------------------------------------------------------------------------
// Project 5.10 : Secondary Cancer Risk & Stray-Dose Monte Carlo
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- a single readable loop over histories, no
//   parallelism, no cleverness -- so that when the GPU and CPU dose tallies agree
//   exactly, we believe the GPU. All the actual physics is in stray_physics.h,
//   shared with the kernel, so "agree exactly" is achievable (not just "close").
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h. Compare against kernels.cu (the GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_stray_problem: parse the committed text format into a StrayProblem.
//   Line 1  : the 11 scalar parameters (see reference_cpu.h / data/README.md).
//   Line 2+ : one "<name> <risk_coeff>" per organ; the count sets n_organs.
// Throws std::runtime_error on any malformed input so main.cu can report it.
// ---------------------------------------------------------------------------
StrayProblem load_stray_problem(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open parameter file: " + path);

    StrayProblem p;
    SimParams& sp = p.sp;

    // -- First non-comment line: the scalar parameters. We skip blank lines and
    //    lines beginning with '#' so the sample file can carry a header comment.
    std::string line;
    auto next_data_line = [&](std::string& out) -> bool {
        while (std::getline(in, out)) {
            std::string t = out;
            std::size_t a = t.find_first_not_of(" \t\r\n");
            if (a == std::string::npos) continue;          // blank
            if (t[a] == '#') continue;                     // comment
            return true;
        }
        return false;
    };

    if (!next_data_line(line))
        throw std::runtime_error("missing parameter line in " + path);
    {
        std::istringstream ss(line);
        if (!(ss >> sp.field_end >> sp.mu >> sp.organ_cm >> sp.scatter_frac
                 >> sp.sidescatter >> sp.leakage_frac >> sp.neutron_frac
                 >> sp.roulette_floor >> sp.roulette_survive
                 >> sp.n_histories >> sp.seed))
            throw std::runtime_error(
                "bad parameter line (expected 11 fields: field_end mu organ_cm "
                "scatter_frac sidescatter leakage_frac neutron_frac roulette_floor "
                "roulette_survive n_histories seed) in " + path);
    }

    // -- Remaining lines: organ metadata (name + risk coefficient). --
    while (next_data_line(line)) {
        std::istringstream ss(line);
        Organ o;
        if (!(ss >> o.name >> o.risk_coeff))
            throw std::runtime_error("bad organ line (expected '<name> <risk_coeff>') in " + path);
        p.organs.push_back(o);
    }

    sp.n_organs = static_cast<int>(p.organs.size());

    // -- Sanity checks so a bad file fails loudly instead of silently mis-running. --
    if (sp.n_organs <= 0)
        throw std::runtime_error("no organs defined in " + path);
    if (sp.field_end < 0 || sp.field_end > sp.n_organs)
        throw std::runtime_error("field_end out of range in " + path);
    if (sp.mu <= 0 || sp.organ_cm <= 0 || sp.n_histories == 0)
        throw std::runtime_error("invalid physics parameters in " + path);
    if (sp.roulette_survive <= 0.0 || sp.roulette_survive > 1.0)
        throw std::runtime_error("roulette_survive must be in (0,1] in " + path);

    return p;
}

// ---------------------------------------------------------------------------
// stray_cpu: the serial reference Monte Carlo. Loop over every history; each gets
// its own reproducible RNG stream, runs the shared transport (simulate_history),
// and adds its deposits to the per-organ tally. This is exactly what the GPU does
// -- just serially and with plain '+=' instead of atomicAdd. Because deposits are
// fixed-point integers, the two tallies are bit-identical.
//   Complexity: O(n_histories * n_organs^2) worst case; fully sequential here.
// ---------------------------------------------------------------------------
void stray_cpu(const StrayProblem& prob, std::vector<unsigned long long>& dose) {
    const SimParams& sp = prob.sp;
    dose.assign(static_cast<std::size_t>(sp.n_organs), 0ULL);

    DepositList dl;   // reused across histories to avoid per-history allocation
    for (unsigned long long i = 0; i < sp.n_histories; ++i) {
        Rng rng = rng_seed(sp.seed, i);      // stream identical to the GPU's for history i
        simulate_history(sp, rng, dl);       // fills dl with (organ, fixed-dose) deposits
        for (int d = 0; d < dl.count; ++d)
            dose[static_cast<std::size_t>(dl.organ[d])] += dl.dose[d];   // plain integer add
    }
}
