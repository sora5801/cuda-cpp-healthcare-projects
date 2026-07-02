// ===========================================================================
// src/reference_cpu.h  --  Cohort config + CPU reference simulation
// ---------------------------------------------------------------------------
// Project 6.23 : Glucose-Insulin Dynamics & Artificial Pancreas
//
// The ensemble is a VIRTUAL-PATIENT COHORT: a 2-D sweep of the two most
// clinically-variable Bergman parameters --
//     * insulin sensitivity  SI  (we vary the insulin-action gain p3), and
//     * glucose effectiveness SG (= p1),
// giving nSI * nSG independent closed-loop simulations. The config + the
// (idx -> patient parameters) mapping live here (shared host+device so the
// kernel reuses them); the actual ODE/RK4/PID is in bergman.h. This file is pure
// C++ (no __global__), so both the host compiler (reference_cpu.cpp) and nvcc
// (kernels.cu) can include it.
//
// READ THIS AFTER: bergman.h (the model). READ BEFORE: reference_cpu.cpp, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "bergman.h"   // BERG_HD, PatientParams, PatientResult, simulate_patient

// ---------------------------------------------------------------------------
// CohortConfig: fixed settings shared by every patient + the two sweep ranges.
//   Loaded from the whitespace text sample (see data/README.md). Fields with a
//   single value are common to all patients; nSI/nSG + the *_lo/*_hi ranges
//   define the cohort grid.
// ---------------------------------------------------------------------------
struct CohortConfig {
    // Fixed Bergman constants (see bergman.h for units/meaning).
    double p2 = 0.0, n = 0.0;
    double Gb = 0.0, Ib = 0.0, VG = 0.0, VI = 0.0;

    // Meal disturbance.
    double meal_D = 0.0, meal_Ag = 0.0, meal_k = 0.0, meal_t = 0.0;

    // PID controller + pump.
    double G_target = 0.0, Kp = 0.0, Ki = 0.0, Kd = 0.0;
    double u_basal = 0.0, u_max = 0.0, control_dt = 0.0;

    // Integration.
    double G0 = 0.0, dt = 0.0;
    int    steps = 0;

    // Cohort sweep: nSI values of insulin-action gain (p3) x nSG values of
    // glucose effectiveness (p1).
    int    nSI = 0, nSG = 0;
    double p3_lo = 0.0, p3_hi = 0.0;   // insulin-action gain range (SI ~ p3/p2)
    double p1_lo = 0.0, p1_hi = 0.0;   // glucose effectiveness (SG) range
};

// Number of virtual patients in the cohort.
BERG_HD inline int cohort_size(const CohortConfig& c) { return c.nSI * c.nSG; }

// ---------------------------------------------------------------------------
// patient_params: build patient `idx`'s full PatientParams from the cohort grid.
//   idx = a*nSG + b  ->  p3 (insulin sensitivity) from a, p1 (glucose
//   effectiveness) from b. Everything else is copied from the shared config.
//   Shared host+device so the kernel and the CPU reference construct byte-for-
//   byte identical parameter sets.
// ---------------------------------------------------------------------------
BERG_HD inline PatientParams patient_params(const CohortConfig& c, int idx) {
    const int a = idx / c.nSG;    // insulin-sensitivity index
    const int b = idx % c.nSG;    // glucose-effectiveness index

    PatientParams p;
    // The two swept parameters (linear grid; single-point when n==1).
    p.p3 = (c.nSI > 1) ? c.p3_lo + (c.p3_hi - c.p3_lo) * a / (c.nSI - 1) : c.p3_lo;
    p.p1 = (c.nSG > 1) ? c.p1_lo + (c.p1_hi - c.p1_lo) * b / (c.nSG - 1) : c.p1_lo;

    // Fixed physiology / settings shared across the cohort.
    p.p2 = c.p2;  p.n = c.n;
    p.Gb = c.Gb;  p.Ib = c.Ib;  p.VG = c.VG;  p.VI = c.VI;
    p.meal_D = c.meal_D;  p.meal_Ag = c.meal_Ag;  p.meal_k = c.meal_k;  p.meal_t = c.meal_t;
    p.G_target = c.G_target;  p.Kp = c.Kp;  p.Ki = c.Ki;  p.Kd = c.Kd;
    p.u_basal = c.u_basal;  p.u_max = c.u_max;  p.control_dt = c.control_dt;
    p.G0 = c.G0;  p.dt = c.dt;  p.steps = c.steps;
    return p;
}

// Load a CohortConfig from the text format (documented in data/README.md).
CohortConfig load_cohort(const std::string& path);

// CPU reference: simulate every patient serially. `results` sized to nSI*nSG.
// The trusted baseline the GPU cohort is checked against (same RK4+PID -> same
// numbers).
void simulate_cohort_cpu(const CohortConfig& c, std::vector<PatientResult>& results);
