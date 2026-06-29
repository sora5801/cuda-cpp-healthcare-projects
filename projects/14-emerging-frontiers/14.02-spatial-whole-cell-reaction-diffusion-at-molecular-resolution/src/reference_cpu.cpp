// ===========================================================================
// src/reference_cpu.cpp  --  Loader, field init, serial RD reference
// ---------------------------------------------------------------------------
// Project 14.02 : Spatial / Whole-Cell Reaction-Diffusion (teaching stencil)
// Compiled by the host compiler only. Physics lives in rd.h.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <stdexcept>
#include <utility>

RdParams load_rd(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open RD parameter file: " + path);
    RdParams P{};
    if (!(in >> P.nx >> P.ny >> P.Du >> P.Dv >> P.F >> P.k >> P.dt >> P.steps >> P.seed_half))
        throw std::runtime_error("bad parameters (expected "
            "'nx ny Du Dv F k dt steps seed_half') in " + path);
    if (P.nx <= 2 || P.ny <= 2 || P.steps < 0 || P.dt <= 0)
        throw std::runtime_error("invalid RD parameters in " + path);
    return P;
}

void init_fields(const RdParams& P, std::vector<double>& U, std::vector<double>& V) {
    const int N = P.nx * P.ny;
    U.assign(N, 1.0);
    V.assign(N, 0.0);
    const int cx = P.nx / 2, cy = P.ny / 2;
    for (int y = cy - P.seed_half; y <= cy + P.seed_half; ++y)
        for (int x = cx - P.seed_half; x <= cx + P.seed_half; ++x) {
            if (x < 0 || x >= P.nx || y < 0 || y >= P.ny) continue;
            U[y * P.nx + x] = 0.5;
            V[y * P.nx + x] = 0.25;
        }
}

void simulate_cpu(const RdParams& P, std::vector<double>& U, std::vector<double>& V) {
    const int N = P.nx * P.ny;
    std::vector<double> Ub(N), Vb(N);
    double* Us = U.data(); double* Vs = V.data();   // source (current state)
    double* Ud = Ub.data(); double* Vd = Vb.data(); // destination (next state)

    for (int s = 0; s < P.steps; ++s) {
        for (int y = 0; y < P.ny; ++y)
            for (int x = 0; x < P.nx; ++x)
                rd_update(x, y, P, Us, Vs, Ud, Vd);
        std::swap(Us, Ud);
        std::swap(Vs, Vd);
    }
    // After the final swap, Us/Vs hold the latest state. If they point at the
    // local buffers (odd step count), copy back into U/V so the caller sees it.
    if (Us != U.data()) { U.assign(Us, Us + N); V.assign(Vs, Vs + N); }
}
