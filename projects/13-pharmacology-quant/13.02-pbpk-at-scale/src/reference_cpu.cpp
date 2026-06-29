// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial PBPK population integration
// ---------------------------------------------------------------------------
// Project 13.02 : PBPK at Scale
// Compiled by the host compiler only. Model/RK4 live in pbpk.h.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <stdexcept>

PbpkParams load_pbpk(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open PBPK parameter file: " + path);
    PbpkParams P{};
    if (!(in >> P.dose >> P.ka >> P.CL >> P.Vc >> P.Vp >> P.Q
             >> P.cv >> P.dt >> P.steps >> P.n_patients >> P.seed))
        throw std::runtime_error("bad parameters (expected "
            "'dose ka CL Vc Vp Q cv dt steps n_patients seed') in " + path);
    if (P.dose <= 0 || P.Vc <= 0 || P.Vp <= 0 || P.steps <= 0 || P.n_patients <= 0 || P.dt <= 0)
        throw std::runtime_error("invalid PBPK parameters in " + path);
    return P;
}

void integrate_cpu(const PbpkParams& P, std::vector<PatientResult>& results) {
    results.assign(P.n_patients, PatientResult{});
    // Each patient is an independent ODE solve -> a plain loop here, one GPU
    // thread per patient in kernels.cu.
    for (int i = 0; i < P.n_patients; ++i)
        results[i] = pbpk_integrate(P, i);
}
