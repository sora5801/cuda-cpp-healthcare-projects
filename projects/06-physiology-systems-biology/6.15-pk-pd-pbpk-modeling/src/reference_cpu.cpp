// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial PK/PD population integration
// ---------------------------------------------------------------------------
// Project 6.15 : PK/PD & PBPK Modeling
//
// Compiled by the HOST compiler only (cl.exe / g++). It contains no CUDA: the
// model, RK4, and RNG all live in pkpd.h as __host__ __device__ inline functions,
// which this file includes and drives from a plain serial loop. That is exactly
// why the CPU reference and the GPU kernel agree to round-off (PATTERNS.md §2).
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// load_pkpd: parse the one-line population config.
//   Format (see data/README.md):
//     dose ka CL Vc kin kout Imax IC50 cv dt steps n_patients seed
//   We read strictly in that order and validate the physically-meaningful fields
//   (positive volumes/rates/counts, an inhibition fraction in [0,1]). A bad file
//   throws so the demo stops with a clear message rather than producing nonsense.
// ---------------------------------------------------------------------------
PkPdParams load_pkpd(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open PK/PD parameter file: " + path);

    PkPdParams P{};
    if (!(in >> P.dose >> P.ka >> P.CL >> P.Vc
             >> P.kin >> P.kout >> P.Imax >> P.IC50
             >> P.cv >> P.dt >> P.steps >> P.n_patients >> P.seed))
        throw std::runtime_error(
            "bad parameters (expected 'dose ka CL Vc kin kout Imax IC50 cv dt "
            "steps n_patients seed') in " + path);

    // Physical sanity checks: everything that would break the model or the RK4.
    if (P.dose <= 0 || P.ka <= 0 || P.CL <= 0 || P.Vc <= 0 ||
        P.kin <= 0 || P.kout <= 0 || P.IC50 <= 0 ||
        P.dt <= 0 || P.steps <= 0 || P.n_patients <= 0)
        throw std::runtime_error("non-physical (non-positive) PK/PD parameter in " + path);
    if (P.Imax < 0.0 || P.Imax > 1.0)
        throw std::runtime_error("Imax must be a fraction in [0,1] in " + path);
    return P;
}

// ---------------------------------------------------------------------------
// integrate_cpu: the serial reference population.
//   Each patient is an INDEPENDENT ODE solve, so the reference is a plain loop
//   here; the GPU replaces this loop with one thread per patient (kernels.cu).
//   Both call the identical pkpd_integrate() from pkpd.h.
// ---------------------------------------------------------------------------
void integrate_cpu(const PkPdParams& P, std::vector<PatientResult>& results) {
    results.assign(P.n_patients, PatientResult{});
    for (int i = 0; i < P.n_patients; ++i)
        results[i] = pkpd_integrate(P, i);
}
