// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial ensemble integration
// ---------------------------------------------------------------------------
// Project 9.02 : Large-Scale Compartmental & Metapopulation Models
// Compiled by the host compiler only. ODE/RK4 lives in seir.h.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <stdexcept>

EnsembleConfig load_ensemble(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open ensemble file: " + path);
    EnsembleConfig c;
    if (!(in >> c.N >> c.I0 >> c.dt >> c.steps >> c.sigma
             >> c.nb >> c.ng >> c.beta_lo >> c.beta_hi >> c.gamma_lo >> c.gamma_hi))
        throw std::runtime_error("bad parameters (expected "
            "'N I0 dt steps sigma nb ng beta_lo beta_hi gamma_lo gamma_hi') in " + path);
    if (c.N <= 0 || c.nb <= 0 || c.ng <= 0 || c.steps <= 0 || c.dt <= 0)
        throw std::runtime_error("invalid ensemble parameters in " + path);
    return c;
}

void integrate_cpu(const EnsembleConfig& c, std::vector<MemberResult>& results) {
    const int M = ensemble_size(c);
    results.assign(M, MemberResult{});
    // Each member is an independent SEIR solve -> a plain loop here, one GPU
    // thread per member in kernels.cu. Initial condition: one seed of I0
    // infectious, the rest susceptible.
    for (int idx = 0; idx < M; ++idx) {
        double beta, gamma;
        member_params(c, idx, beta, gamma);
        results[idx] = integrate_member(c.N, c.N - c.I0, 0.0, c.I0, 0.0,
                                        beta, c.sigma, gamma, c.dt, c.steps);
    }
}
