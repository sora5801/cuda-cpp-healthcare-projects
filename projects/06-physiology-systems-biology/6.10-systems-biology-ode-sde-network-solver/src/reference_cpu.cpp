// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial ensemble integration (the baseline)
// ---------------------------------------------------------------------------
// Project 6.10 : Systems-Biology ODE/SDE Network Solver
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- a single readable loop, no parallelism -- so that
//   when the GPU and CPU agree, we believe the GPU. Compiled by the host C++
//   compiler only (no CUDA here); the ODE/RK4 arithmetic lives in grn.h and is
//   shared verbatim with the GPU kernel.
//
// READ THIS AFTER: reference_cpu.h. Compare against kernels.cu (the GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// load_ensemble: parse the whitespace-separated config file. The 16 fields are
//   alpha0 beta dt steps na nn alpha_lo alpha_hi n_lo n_hi  m0 m1 m2 p0 p1 p2
// (the last 6 are the shared initial state). We validate the essentials so a
// truncated or nonsensical file aborts with a clear message rather than
// producing meaningless "results".
// ---------------------------------------------------------------------------
EnsembleConfig load_ensemble(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open ensemble file: " + path);

    EnsembleConfig c;
    if (!(in >> c.alpha0 >> c.beta >> c.dt >> c.steps >> c.na >> c.nn
             >> c.alpha_lo >> c.alpha_hi >> c.n_lo >> c.n_hi))
        throw std::runtime_error("bad header (expected 'alpha0 beta dt steps na nn "
            "alpha_lo alpha_hi n_lo n_hi ...') in " + path);
    for (int j = 0; j < STATE_DIM; ++j) {
        if (!(in >> c.s0[j]))
            throw std::runtime_error("bad initial state (expected 6 values "
                "m0 m1 m2 p0 p1 p2) in " + path);
    }
    if (c.na <= 0 || c.nn <= 0 || c.steps <= 0 || c.dt <= 0.0 || c.beta <= 0.0)
        throw std::runtime_error("invalid ensemble parameters in " + path);
    return c;
}

// ---------------------------------------------------------------------------
// integrate_cpu: solve every member serially. Each member is an INDEPENDENT
//   initial-value problem -> a plain for-loop here, exactly one GPU thread per
//   member in kernels.cu. The per-member work (RK4 loop + oscillation summary)
//   is integrate_member() from grn.h, so this loop and the kernel run identical
//   arithmetic and their MemberResults match to round-off.
//
//   Complexity: O(M * steps) with M = na*nn members; O(1) extra space per
//   member (the whole 6-D state lives on the stack).
// ---------------------------------------------------------------------------
void integrate_cpu(const EnsembleConfig& c, std::vector<MemberResult>& results) {
    const int M = ensemble_size(c);
    results.assign(static_cast<std::size_t>(M), MemberResult{});
    for (int idx = 0; idx < M; ++idx) {
        GrnParams pr;
        member_params(c, idx, pr);                 // this member's (alpha, n) + fixed knobs
        results[static_cast<std::size_t>(idx)] =
            integrate_member(c.s0, pr, c.dt, c.steps);
    }
}
