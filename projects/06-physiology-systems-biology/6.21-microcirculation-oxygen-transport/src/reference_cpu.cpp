// ===========================================================================
// src/reference_cpu.cpp  --  Problem loader + serial CPU reference solver
// ---------------------------------------------------------------------------
// Project 6.21 : Microcirculation & Oxygen Transport
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- a single readable loop over grid points, no
//   parallelism, no cleverness -- so that when the GPU and CPU agree, we believe
//   the GPU. Both call the SAME solve_point() from reference_cpu.h, so agreement
//   is exact to round-off.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h
//   for the containers and the shared per-point math; oxygen.h for the physics.
//
// READ THIS AFTER: oxygen.h, reference_cpu.h. Compare against kernels.cu (the
//   GPU twin, which runs solve_point() one-thread-per-grid-point).
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_problem: parse the sample text file into an OxyProblem.
//
//   FILE FORMAT (whitespace-separated; see data/README.md for the annotated
//   version). A header line of grid + physiology, then one line per capillary
//   segment. The '#'-comment lines in the sample are stripped by reading token
//   by token and skipping anything that starts with '#'... but to keep the loader
//   dead simple and robust we instead read a strict fixed sequence of numbers and
//   let the sample keep comments only at the very top (the file we ship has the
//   numbers first). Concretely we read:
//
//     nx ny nz spacing po2_inflow m0 km p50 hill_n n_src
//     then n_src lines of:  x  y  z  blood_po2
//
//   Each segment's SOURCE STRENGTH q is derived here (host-side, once) from its
//   blood PO2 via the Hill saturation: q = q_scale * S(blood_po2). A segment with
//   more-saturated blood delivers more O2, so it is a stronger tissue source.
//   Doing this conversion ONCE at load time (not per grid point) keeps the hot
//   solve loop a pure linear superposition and keeps CPU/GPU identical (they both
//   just read the precomputed q).
//
//   Throws std::runtime_error on any I/O or range problem so demos fail loudly.
// ---------------------------------------------------------------------------
OxyProblem load_problem(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open problem file: " + path);

    // Helper: read the next whitespace-separated token, skipping full-line
    // comments that begin with '#'. This lets the shipped sample carry a
    // human-readable comment header without confusing the parser.
    auto next_token = [&](std::string& tok) -> bool {
        while (in >> tok) {
            if (!tok.empty() && tok[0] == '#') {
                // Discard the rest of this comment line.
                std::string rest;
                std::getline(in, rest);
                continue;
            }
            return true;
        }
        return false;
    };
    auto next_double = [&](const char* what) -> double {
        std::string tok;
        if (!next_token(tok)) throw std::runtime_error(std::string("unexpected EOF reading ") + what + " in " + path);
        try { return std::stod(tok); }
        catch (...) { throw std::runtime_error(std::string("bad number '") + tok + "' for " + what + " in " + path); }
    };
    auto next_int = [&](const char* what) -> int {
        return static_cast<int>(next_double(what));
    };

    OxyProblem prob;
    TissueGrid& g = prob.grid;

    g.nx = next_int("nx");
    g.ny = next_int("ny");
    g.nz = next_int("nz");
    g.spacing    = next_double("spacing");
    g.po2_inflow = next_double("po2_inflow");
    g.m0         = next_double("m0");
    g.km         = next_double("km");
    const double p50    = next_double("p50");      // consumed here to set q
    const double hill_n = next_double("hill_n");   // consumed here to set q
    const int    n_src  = next_int("n_src");

    if (g.nx <= 0 || g.ny <= 0 || g.nz <= 0 || g.spacing <= 0.0)
        throw std::runtime_error("invalid grid dimensions/spacing in " + path);
    if (n_src <= 0)
        throw std::runtime_error("need at least one capillary segment in " + path);

    // A fixed scale converting Hill saturation (0..1) into a source strength.
    // Chosen so the sample produces a physiologically legible PO2 field
    // (see data/README.md). Kept explicit so the model has no hidden magic.
    const double q_scale = 400.0;   // (mmHg * um) per unit saturation

    prob.sources.reserve(static_cast<std::size_t>(n_src));
    for (int j = 0; j < n_src; ++j) {
        OxySource s;
        s.x = next_double("segment.x");
        s.y = next_double("segment.y");
        s.z = next_double("segment.z");
        const double blood_po2 = next_double("segment.blood_po2");
        // Hill saturation of THIS segment's blood -> its source strength.
        s.q = q_scale * hill_saturation(blood_po2, p50, hill_n);
        prob.sources.push_back(s);
    }
    return prob;
}

// ---------------------------------------------------------------------------
// solve_cpu: evaluate solve_point() at every grid point, serially.
//   Complexity: O(N_grid * N_src). This is the readable baseline whose wall time
//   (timed in main.cu) we compare with the GPU kernel, and whose numbers the GPU
//   must reproduce. One flat loop -> obviously correct.
// ---------------------------------------------------------------------------
void solve_cpu(const OxyProblem& problem, std::vector<double>& po2) {
    const int n = grid_size(problem.grid);
    po2.assign(static_cast<std::size_t>(n), 0.0);
    const OxySource* src = problem.sources.data();
    const int n_src = static_cast<int>(problem.sources.size());
    for (int idx = 0; idx < n; ++idx) {
        po2[static_cast<std::size_t>(idx)] = solve_point(problem.grid, src, n_src, idx);
    }
}
