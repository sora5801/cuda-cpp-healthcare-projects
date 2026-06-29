// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial LBM reference
// ---------------------------------------------------------------------------
// Project 6.04 : Lattice-Boltzmann Blood/Airflow Solver
// Compiled by the host compiler only. Physics lives in lbm_d2q9.h.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <stdexcept>

LbmParams load_lbm(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open LBM parameter file: " + path);
    LbmParams p;
    if (!(in >> p.nx >> p.ny >> p.steps >> p.tau >> p.gx))
        throw std::runtime_error("bad parameters (expected 'nx ny steps tau gx') in " + path);
    if (p.nx <= 0 || p.ny <= 0 || p.steps < 0 || p.tau <= 0.5)
        throw std::runtime_error("invalid LBM parameters (need tau>0.5) in " + path);
    return p;
}

void lbm_cpu(const LbmParams& p, std::vector<double>& f_final) {
    const std::size_t cells = static_cast<std::size_t>(9) * p.nx * p.ny;
    std::vector<double> fa(cells), fb(cells);

    // Initialize at REST equilibrium: f_i = w_i (so rho=1, u=0 everywhere).
    for (int y = 0; y < p.ny; ++y)
        for (int x = 0; x < p.nx; ++x)
            for (int i = 0; i < 9; ++i)
                fa[lbm_idx(i, x, y, p.nx, p.ny)] = w_i(i);

    // Time loop: read from `src`, write to `dst`, then swap (ping-pong buffers).
    double* src = fa.data();
    double* dst = fb.data();
    for (int s = 0; s < p.steps; ++s) {
        for (int y = 0; y < p.ny; ++y)
            for (int x = 0; x < p.nx; ++x)
                lbm_collide_stream(x, y, p.nx, p.ny, p.tau, p.gx, src, dst);
        double* tmp = src; src = dst; dst = tmp;
    }

    // `src` now holds the latest state (after the final swap).
    f_final.assign(src, src + cells);
}

void velocity_field(const LbmParams& p, const std::vector<double>& f, std::vector<double>& ux) {
    ux.assign(static_cast<std::size_t>(p.nx) * p.ny, 0.0);
    for (int y = 0; y < p.ny; ++y)
        for (int x = 0; x < p.nx; ++x)
            ux[static_cast<std::size_t>(y) * p.nx + x] = lbm_ux(x, y, p.nx, p.ny, f.data());
}
