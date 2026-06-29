// ===========================================================================
// src/pbpk.h  --  Shared (host + device) PBPK model + RK4 + patient sampling
// ---------------------------------------------------------------------------
// Project 13.02 : PBPK at Scale
//
// WHAT THIS PROJECT COMPUTES
//   A VIRTUAL POPULATION pharmacokinetic study: solve a physiologically-based
//   pharmacokinetic (PBPK) ODE for thousands of virtual patients, each with
//   different physiology (sampled clearance, volumes, absorption), and summarize
//   each patient's exposure (Cmax, Tmax, AUC). Every patient is an INDEPENDENT
//   ODE solve, so each GPU thread integrates one patient.
//
//   We use a SIMPLIFIED 3-compartment model (a teaching reduction of the full
//   ~15-compartment PBPK; see THEORY): an oral DEPOT (gut), a CENTRAL compartment
//   (plasma), and a PERIPHERAL tissue compartment, with first-order absorption ka,
//   inter-compartment flow Q, and hepatic clearance CL. Amounts (mg):
//     dA_gut/dt = -ka*A_gut
//     dA_cen/dt =  ka*A_gut - CL*Cc - Q*(Cc - Cp)
//     dA_per/dt =  Q*(Cc - Cp)
//   with plasma concentration Cc = A_cen/Vc and Cp = A_per/Vp.
//
//   The RNG, the ODE, and RK4 all live here as __host__ __device__ functions so
//   the CPU reference and the GPU kernel produce identical patients and results.
//   (A shared splitmix64 RNG -- not cuRAND -- so the populations match exactly;
//    cf. project 5.01.)
// ===========================================================================
#pragma once

#include <cstdint>
#include <cmath>

#ifdef __CUDACC__
#define PBPK_HD __host__ __device__
#else
#define PBPK_HD
#endif

// --- Reproducible per-patient RNG (splitmix64), shared host/device ---
struct Rng { uint64_t state; };
PBPK_HD inline uint64_t pbpk_splitmix(uint64_t& x) {
    x += 0x9E3779B97F4A7C15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
PBPK_HD inline Rng pbpk_seed(uint64_t base, uint64_t patient) {
    Rng r; r.state = base ^ (patient * 0x9E3779B97F4A7C15ULL + 0xD1B54A32D192ED03ULL);
    pbpk_splitmix(r.state);
    return r;
}
PBPK_HD inline double pbpk_uniform(Rng& r) {
    return (pbpk_splitmix(r.state) >> 11) * (1.0 / 9007199254740992.0);   // [0,1)
}
// Standard normal via Box-Muller (deterministic, host+device).
PBPK_HD inline double pbpk_normal(Rng& r) {
    const double u1 = 1.0 - pbpk_uniform(r);   // (0,1]
    const double u2 = pbpk_uniform(r);
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}
// Log-normal sample with given median and coefficient-of-variation-like spread.
PBPK_HD inline double pbpk_lognormal(Rng& r, double median, double cv) {
    return median * exp(cv * pbpk_normal(r));
}

// Population parameters (medians + variability) read from the data file.
struct PbpkParams {
    double dose;        // oral dose (mg)
    double ka, CL, Vc, Vp, Q;   // median absorption / clearance / volumes / flow
    double cv;          // log-normal spread for the sampled parameters
    double dt;          // integration step (h)
    int    steps;       // number of steps (run length = steps*dt hours)
    int    n_patients;  // virtual population size
    uint64_t seed;      // base RNG seed
};

// Per-patient exposure summary.
struct PatientResult { double Cmax; double Tmax; double AUC; };

// PBPK right-hand side (3 compartments).
PBPK_HD inline void pbpk_deriv(double Ag, double Ac, double Ap,
                              double ka, double CL, double Vc, double Vp, double Q,
                              double& dAg, double& dAc, double& dAp) {
    const double Cc = Ac / Vc, Cp = Ap / Vp;
    dAg = -ka * Ag;
    dAc = ka * Ag - CL * Cc - Q * (Cc - Cp);
    dAp = Q * (Cc - Cp);
}

// One RK4 step on the 3 compartment amounts.
PBPK_HD inline void pbpk_rk4(double& Ag, double& Ac, double& Ap,
                            double ka, double CL, double Vc, double Vp, double Q, double dt) {
    double k1g,k1c,k1p, k2g,k2c,k2p, k3g,k3c,k3p, k4g,k4c,k4p;
    pbpk_deriv(Ag, Ac, Ap, ka,CL,Vc,Vp,Q, k1g,k1c,k1p);
    pbpk_deriv(Ag+0.5*dt*k1g, Ac+0.5*dt*k1c, Ap+0.5*dt*k1p, ka,CL,Vc,Vp,Q, k2g,k2c,k2p);
    pbpk_deriv(Ag+0.5*dt*k2g, Ac+0.5*dt*k2c, Ap+0.5*dt*k2p, ka,CL,Vc,Vp,Q, k3g,k3c,k3p);
    pbpk_deriv(Ag+dt*k3g, Ac+dt*k3c, Ap+dt*k3p, ka,CL,Vc,Vp,Q, k4g,k4c,k4p);
    Ag += (dt/6.0)*(k1g+2*k2g+2*k3g+k4g);
    Ac += (dt/6.0)*(k1c+2*k2c+2*k3c+k4c);
    Ap += (dt/6.0)*(k1p+2*k2p+2*k3p+k4p);
}

// Integrate ONE virtual patient: sample physiology, run RK4, summarize exposure.
PBPK_HD inline PatientResult pbpk_integrate(const PbpkParams& P, int patient) {
    Rng r = pbpk_seed(P.seed, patient);
    // Sample this patient's parameters log-normally around the population medians.
    const double ka = pbpk_lognormal(r, P.ka, P.cv);
    const double CL = pbpk_lognormal(r, P.CL, P.cv);
    const double Vc = pbpk_lognormal(r, P.Vc, P.cv);
    const double Vp = pbpk_lognormal(r, P.Vp, P.cv);
    const double Q  = pbpk_lognormal(r, P.Q,  P.cv);

    double Ag = P.dose, Ac = 0.0, Ap = 0.0;     // oral dose enters the gut depot
    PatientResult out{0.0, 0.0, 0.0};
    double Cc_prev = 0.0;                        // plasma conc at t=0 is 0
    for (int s = 1; s <= P.steps; ++s) {
        pbpk_rk4(Ag, Ac, Ap, ka, CL, Vc, Vp, Q, P.dt);
        const double Cc = Ac / Vc;
        out.AUC += 0.5 * (Cc_prev + Cc) * P.dt;  // trapezoidal area under the curve
        if (Cc > out.Cmax) { out.Cmax = Cc; out.Tmax = s * P.dt; }
        Cc_prev = Cc;
    }
    return out;
}
