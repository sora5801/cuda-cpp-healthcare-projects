// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial ensemble integration (the baseline)
// ---------------------------------------------------------------------------
// Project 1.23 : QM/MM Molecular Dynamics   (reduced-scope teaching version)
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. It is written to be
//   OBVIOUSLY correct: a single readable loop over ensemble members, each one a
//   call to the shared qmmm::integrate_trajectory (qmmm.h). Because the GPU
//   kernel calls the SAME function, agreement is a real correctness signal.
//
//   Compiled by the host C++ compiler only (no CUDA syntax here). The physics
//   and the Verlet integrator live in qmmm.h; this file only parses the config
//   and drives the serial loop. See reference_cpu.h for the declarations.
//
// READ THIS AFTER: reference_cpu.h, qmmm.h. Compare against kernels.cu (GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// load_ensemble: read the single-line config file (data/README.md format).
//   Layout:  dt steps v0 nf nx field_lo field_hi x0_lo x0_hi
//   We validate aggressively so a malformed file fails LOUDLY at startup rather
//   than silently simulating garbage.
// ---------------------------------------------------------------------------
EnsembleConfig load_ensemble(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open ensemble file: " + path);
    EnsembleConfig c;
    if (!(in >> c.dt >> c.steps >> c.v0 >> c.nf >> c.nx
             >> c.field_lo >> c.field_hi >> c.x0_lo >> c.x0_hi))
        throw std::runtime_error("bad parameters (expected "
            "'dt steps v0 nf nx field_lo field_hi x0_lo x0_hi') in " + path);
    if (c.dt <= 0.0 || c.steps <= 0 || c.nf <= 0 || c.nx <= 0)
        throw std::runtime_error("invalid ensemble parameters in " + path);
    return c;
}

// ---------------------------------------------------------------------------
// integrate_cpu: integrate every ensemble member serially.
//   Each member is an INDEPENDENT QM/MM trajectory -> a plain for-loop here, and
//   one GPU thread per member in kernels.cu. The (field, x0) for member idx come
//   from member_params(); the run itself is qmmm::integrate_trajectory().
//   Complexity: O(M * steps) where M = nf*nx; each step does one 2x2 QM solve.
// ---------------------------------------------------------------------------
void integrate_cpu(const EnsembleConfig& c, std::vector<qmmm::TrajResult>& results) {
    const int M = ensemble_size(c);
    results.assign(static_cast<std::size_t>(M), qmmm::TrajResult{});
    for (int idx = 0; idx < M; ++idx) {
        double field, x0;
        member_params(c, idx, field, x0);
        results[static_cast<std::size_t>(idx)] =
            qmmm::integrate_trajectory(x0, c.v0, field, c.dt, c.steps);
    }
}
