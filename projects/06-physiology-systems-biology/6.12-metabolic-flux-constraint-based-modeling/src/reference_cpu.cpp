// ===========================================================================
// src/reference_cpu.cpp  --  Model loader + serial CPU knockout screen
// ---------------------------------------------------------------------------
// Project 6.12 : Metabolic Flux / Constraint-Based Modeling
//
// ROLE IN THE PROJECT
//   Two host-only jobs:
//     1. load_model(): turn the text model file into an FbaModel.
//     2. screen_cpu(): the trusted, obviously-correct baseline -- solve the
//        wild-type FBA LP and every single-reaction knockout in a plain serial
//        loop. main.cu compares this array against the GPU screen (kernels.cu).
//
//   The LP solver itself is NOT here -- it lives in fba.h as shared host+device
//   code, so this reference and the GPU kernel run identical arithmetic.
//
//   Compiled by the host C++ compiler only (no CUDA). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, fba.h. Compare screen_cpu() with the GPU
// twin screen_kernel() in kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <cstring>     // std::strcmp
#include <fstream>     // std::ifstream
#include <sstream>     // std::istringstream
#include <stdexcept>   // std::runtime_error
#include <string>

// ---------------------------------------------------------------------------
// load_model: parse the text model format documented in reference_cpu.h.
//   We read the whole file line by line. A line beginning "#names:" carries the
//   reaction labels (everything after the token). Any other '#'/blank line is a
//   comment. All remaining whitespace-separated tokens form one positional
//   stream: nmet, nrxn, then the S matrix, lb, ub, c.
//
//   We validate sizes against the fixed FBA_MAX_* capacities (the solver's local
//   arrays) and throw on any shortfall, so a malformed file fails immediately and
//   legibly rather than producing a silently-wrong answer.
// ---------------------------------------------------------------------------
FbaModel load_model(const std::string& path, std::vector<std::string>& names) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open model file: " + path);

    std::ostringstream numbers;   // accumulates the positional numeric stream
    names.clear();
    std::string line;
    while (std::getline(in, line)) {
        // Trim a leading carriage return (Windows line endings on a *nix read).
        if (!line.empty() && line.back() == '\r') line.pop_back();

        // Find the first non-space character to classify the line.
        std::size_t p = line.find_first_not_of(" \t");
        if (p == std::string::npos) continue;                 // blank -> skip

        if (line.compare(p, 7, "#names:") == 0) {
            // Reaction labels: split the remainder on whitespace.
            std::istringstream ls(line.substr(p + 7));
            std::string tok;
            while (ls >> tok) names.push_back(tok);
            continue;
        }
        if (line[p] == '#') continue;                          // ordinary comment

        numbers << ' ' << line;                                // a data line
    }

    std::istringstream ns(numbers.str());
    FbaModel model;
    if (!(ns >> model.nmet >> model.nrxn))
        throw std::runtime_error("model file missing 'nmet nrxn' header: " + path);

    const int m = model.nmet, n = model.nrxn;
    if (m <= 0 || n <= 0)
        throw std::runtime_error("model has non-positive nmet/nrxn: " + path);
    if (m > FBA_MAX_MET || n > FBA_MAX_RXN)
        throw std::runtime_error("model exceeds compiled FBA_MAX_MET/RXN capacity "
                                 "(increase them in fba.h): " + path);

    // Stoichiometry matrix S (m*n), row-major.
    for (int i = 0; i < m * n; ++i)
        if (!(ns >> model.S[i]))
            throw std::runtime_error("model file: not enough S entries in " + path);
    // Bounds and objective (n each).
    for (int j = 0; j < n; ++j)
        if (!(ns >> model.lb[j])) throw std::runtime_error("model file: not enough lb in " + path);
    for (int j = 0; j < n; ++j)
        if (!(ns >> model.ub[j])) throw std::runtime_error("model file: not enough ub in " + path);
    for (int j = 0; j < n; ++j)
        if (!(ns >> model.c[j]))  throw std::runtime_error("model file: not enough c in "  + path);

    // Default labels if the file supplied none (or the wrong count).
    if (static_cast<int>(names.size()) != n) {
        names.assign(static_cast<std::size_t>(n), std::string());
        for (int j = 0; j < n; ++j) names[static_cast<std::size_t>(j)] = "R" + std::to_string(j);
    }
    return model;
}

// ---------------------------------------------------------------------------
// screen_cpu: solve the wild type + every single-reaction knockout, serially.
//   results has (nrxn + 1) entries: [0..nrxn) = knockout of each reaction, and
//   the final entry = wild type. Each solve is INDEPENDENT (a fresh LP on a
//   local copy of the model with one reaction clamped to zero) -- which is
//   exactly why the GPU version gives each solve its own thread. Here we just
//   loop, so the correctness is self-evident.
// ---------------------------------------------------------------------------
void screen_cpu(const FbaModel& model, std::vector<FbaResult>& results) {
    const int n = model.nrxn;
    results.assign(static_cast<std::size_t>(n + 1), FbaResult{});
    for (int k = 0; k < n; ++k)
        results[static_cast<std::size_t>(k)] = solve_knockout(model, k);   // delete reaction k
    results[static_cast<std::size_t>(n)] = solve_knockout(model, -1);       // wild type
}
