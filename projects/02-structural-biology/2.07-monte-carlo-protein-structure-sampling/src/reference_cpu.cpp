// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial Monte Carlo reference (the baseline)
// ---------------------------------------------------------------------------
// Project 2.7 : Monte Carlo Protein Structure Sampling (HP lattice model)
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- a single serial loop over replicas calling the
//   SAME run_replica() the GPU thread calls -- so when the GPU and CPU agree, we
//   believe the GPU. The only difference between this and the kernel is HOW the
//   replicas are scheduled (serial for-loop here, one-thread-each on the GPU).
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h
//   and the shared engine in mc_moves.h.
//
// READ THIS AFTER: reference_cpu.h, mc_moves.h. Compare with kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <stdexcept>
#include <cctype>     // std::toupper

// ---------------------------------------------------------------------------
// load_mc_problem: parse the two-line sample format (see data/README.md).
//   Line 1:  n sweeps n_replicas t_min t_max seed
//   Line 2:  HP sequence, n characters from {H,P} (case-insensitive)
// We validate aggressively because a silently-truncated problem would make the
// "GPU matches CPU" check meaningless.
// ---------------------------------------------------------------------------
McProblem load_mc_problem(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open problem file: " + path);

    McProblem P{};
    // Read the scalar header. operator>> skips whitespace/newlines, so the two
    // logical "lines" just need to be in this order in the file.
    if (!(in >> P.n >> P.sweeps >> P.n_replicas >> P.t_min >> P.t_max >> P.seed))
        throw std::runtime_error("bad header (expected "
            "'n sweeps n_replicas t_min t_max seed') in " + path);

    if (P.n < 2 || P.n > MC_MAX_N)
        throw std::runtime_error("chain length n out of range [2, "
            + std::to_string(MC_MAX_N) + "] in " + path);
    if (P.sweeps <= 0 || P.n_replicas <= 0)
        throw std::runtime_error("sweeps and n_replicas must be positive in " + path);
    if (!(P.t_min > 0.0) || !(P.t_max >= P.t_min))
        throw std::runtime_error("require 0 < t_min <= t_max in " + path);

    // Read exactly n residue-type characters into the hp[] array.
    std::string seq;
    if (!(in >> seq) || (int)seq.size() != P.n)
        throw std::runtime_error("HP sequence must have exactly n characters in " + path);
    for (int i = 0; i < P.n; ++i) {
        char c = (char)std::toupper((unsigned char)seq[i]);
        if      (c == 'H') P.hp[i] = 1;   // hydrophobic: forms H-H contacts
        else if (c == 'P') P.hp[i] = 0;   // polar: inert in the energy function
        else throw std::runtime_error(std::string("sequence char must be H or P, got '")
                                      + seq[i] + "' in " + path);
    }
    return P;
}

// ---------------------------------------------------------------------------
// sample_cpu: run every replica serially. Each replica r uses its own slice of
// the prebuilt Boltzmann `tables` buffer (stride = boltzmann_table_size()) and
// its own RNG stream (seed, r) inside run_replica(). This is EXACTLY what the
// GPU does -- just one replica at a time instead of one thread each.
//   Complexity: O(n_replicas * sweeps * n * n) -- the inner n*n is the O(n^2)
//   contact recount per accepted-or-rejected attempt (kept naive for clarity).
// ---------------------------------------------------------------------------
void sample_cpu(const McProblem& prob, const std::vector<double>& tables,
                std::vector<McResult>& out) {
    const int stride = boltzmann_table_size();   // doubles per replica table
    out.assign((std::size_t)prob.n_replicas, McResult{});
    for (int r = 0; r < prob.n_replicas; ++r) {
        // Hand run_replica this replica's table slice; it reads (never writes) it.
        const double* tbl = tables.data() + (std::size_t)r * stride;
        out[(std::size_t)r] = run_replica(prob, r, tbl);
    }
}
