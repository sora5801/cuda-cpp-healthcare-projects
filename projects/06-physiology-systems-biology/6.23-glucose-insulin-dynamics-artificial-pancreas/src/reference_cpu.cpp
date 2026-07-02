// ===========================================================================
// src/reference_cpu.cpp  --  Cohort loader + serial reference simulation
// ---------------------------------------------------------------------------
// Project 6.23 : Glucose-Insulin Dynamics & Artificial Pancreas
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU cohort is checked against. It is written
//   to be OBVIOUSLY correct -- a single readable loop over patients, no
//   parallelism -- so that when the GPU and CPU agree, we believe the GPU. The
//   actual physiology/RK4/PID lives in bergman.h and is shared with the kernel,
//   so the two sides run identical math.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, bergman.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_cohort: parse the whitespace text sample into a CohortConfig.
//   The field order is fixed and documented in data/README.md. We read every
//   field explicitly (rather than a loop) so the parser doubles as documentation
//   of the format, and so a truncated/garbled file fails loudly.
// ---------------------------------------------------------------------------
CohortConfig load_cohort(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open cohort file: " + path);

    CohortConfig c;
    if (!(in >> c.p2 >> c.n >> c.Gb >> c.Ib >> c.VG >> c.VI
             >> c.meal_D >> c.meal_Ag >> c.meal_k >> c.meal_t
             >> c.G_target >> c.Kp >> c.Ki >> c.Kd
             >> c.u_basal >> c.u_max >> c.control_dt
             >> c.G0 >> c.dt >> c.steps
             >> c.nSI >> c.nSG >> c.p3_lo >> c.p3_hi >> c.p1_lo >> c.p1_hi)) {
        throw std::runtime_error(
            "bad parameters (expected 26 values: p2 n Gb Ib VG VI "
            "meal_D meal_Ag meal_k meal_t G_target Kp Ki Kd u_basal u_max "
            "control_dt G0 dt steps nSI nSG p3_lo p3_hi p1_lo p1_hi) in " + path);
    }
    // Sanity guards: reject nonsense so demos fail with a clear message rather
    // than dividing by zero or looping forever.
    if (c.nSI <= 0 || c.nSG <= 0 || c.steps <= 0 || c.dt <= 0.0 ||
        c.control_dt <= 0.0 || c.VG == 0.0 || c.VI == 0.0) {
        throw std::runtime_error("invalid cohort parameters in " + path);
    }
    return c;
}

// ---------------------------------------------------------------------------
// simulate_cohort_cpu: run every virtual patient serially.
//   Each patient is an INDEPENDENT closed-loop simulation -> a plain loop here,
//   one GPU thread per patient in kernels.cu. simulate_patient() (bergman.h) does
//   the actual work and is shared verbatim with the kernel.
//   Complexity: O(cohort * steps).
// ---------------------------------------------------------------------------
void simulate_cohort_cpu(const CohortConfig& c, std::vector<PatientResult>& results) {
    const int M = cohort_size(c);
    results.assign(static_cast<std::size_t>(M), PatientResult{});
    for (int idx = 0; idx < M; ++idx) {
        const PatientParams p = patient_params(c, idx);
        results[static_cast<std::size_t>(idx)] = simulate_patient(p);
    }
}
