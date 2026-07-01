// ===========================================================================
// src/pkpd.h  --  Shared (host + device) PK/PD model + RK4 + patient sampling
// ---------------------------------------------------------------------------
// Project 6.15 : PK/PD & PBPK Modeling
//
// WHAT THIS PROJECT COMPUTES
//   A VIRTUAL POPULATION PK/PD study. For each of thousands of virtual patients
//   we solve a coupled ordinary-differential-equation (ODE) system with two
//   linked halves:
//
//     * PK (pharmacoKINETICS) -- "what the body does to the drug": how the drug
//       concentration in plasma rises and falls after an oral dose. We use the
//       classic ONE-COMPARTMENT model with first-order absorption: an oral
//       DEPOT (gut) feeds a CENTRAL (plasma) compartment; the drug is cleared
//       from plasma at rate CL. This is the textbook starting point for PBPK
//       (project 13.02 extends it to a multi-tissue PBPK model).
//
//     * PD (pharmacoDYNAMICS) -- "what the drug does to the body": how the drug
//       moves a biological response. We use an INDIRECT-RESPONSE (turnover)
//       model (Dayneka/Jusko 1993): a biomarker R (e.g. a clotting factor, a
//       cholesterol level, an enzyme) is produced at zero-order rate kin and
//       removed at first-order rate kout, so at baseline R0 = kin/kout. The drug
//       INHIBITS the removal of R (an "inhibition of loss" model), so higher
//       plasma concentration -> slower loss -> R climbs above baseline.
//
//   The two halves are COUPLED: the PK concentration Cc(t) drives the PD effect
//   through a saturating Emax (Hill) inhibition term. That coupling -- solving PK
//   and PD together in one ODE system -- is the single idea this project teaches
//   that a pure-PK project (13.02) does not.
//
//   Amounts (mg) and the response state:
//     dA_gut/dt = -ka * A_gut                                (absorption)
//     dA_cen/dt =  ka * A_gut - CL * Cc                      (absorption - clearance)
//     dR/dt     =  kin - kout * (1 - I(Cc)) * R              (indirect response)
//   with plasma concentration  Cc = A_cen / Vc   and inhibition
//     I(Cc)     =  Imax * Cc / (IC50 + Cc)                   (Emax/Hill, hill=1)
//
//   Each virtual patient has its OWN physiology (ka, CL, Vc, IC50 sampled around
//   population medians), so each patient is an INDEPENDENT ODE solve -> one GPU
//   thread integrates one patient. This is the ensemble-ODE pattern (PATTERNS.md
//   §1, cf. flagships 9.02 SEIR and 13.02 PBPK).
//
//   The RNG, the ODE right-hand side, and the RK4 step all live HERE as
//   `__host__ __device__` inline functions so the CPU reference and the GPU
//   kernel run BYTE-FOR-BYTE-IDENTICAL math (PATTERNS.md §2). We use a shared
//   deterministic splitmix64 RNG (NOT cuRAND) so the CPU reproduces exactly the
//   same virtual population as the GPU -> verification is EXACT to round-off,
//   the same strategy as flagships 5.01 and 13.02.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh.
// READ THIS BEFORE: kernels.cu (GPU twin) and reference_cpu.cpp (serial twin).
// ===========================================================================
#pragma once

#include <cstdint>   // uint64_t -- the RNG state is a 64-bit integer
#include <cmath>     // sqrt, log, cos, exp -- used by the sampler and (host) math

// ---------------------------------------------------------------------------
// HD (host+device) decorator idiom (PATTERNS.md §2).
//   When this header is compiled by nvcc (__CUDACC__ is defined), every function
//   below is marked `__host__ __device__` so it can run on BOTH the CPU and the
//   GPU. When it is compiled by the plain host C++ compiler (cl.exe / g++) for
//   reference_cpu.cpp, those decorators do not exist, so PKPD_HD expands to
//   nothing. Same source, two compilers, identical numerics.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define PKPD_HD __host__ __device__
#else
#define PKPD_HD
#endif

// ---------------------------------------------------------------------------
// Reproducible per-patient RNG: splitmix64.
//   splitmix64 is a tiny, well-distributed 64-bit generator. We use it (instead
//   of cuRAND) precisely because it is a pure integer recurrence with no library
//   state: the CPU and the GPU, given the same seed, emit the SAME stream, so the
//   two virtual populations are identical and the results match exactly.
// ---------------------------------------------------------------------------
struct Rng {
    uint64_t state;   // the generator's 64-bit internal state
};

// Advance the state once and return a well-mixed 64-bit value.
//   The three magic constants are the published splitmix64 mixing constants.
PKPD_HD inline uint64_t pkpd_splitmix(uint64_t& x) {
    x += 0x9E3779B97F4A7C15ULL;                       // add the golden-ratio odd constant
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;      // avalanche step 1
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;      // avalanche step 2
    return z ^ (z >> 31);                             // final xor-shift
}

// Seed a per-patient RNG deterministically from a base seed + the patient index.
//   Mixing the patient index into the seed gives every patient an independent,
//   reproducible stream. We advance once so consecutive patient indices do not
//   produce correlated first draws.
PKPD_HD inline Rng pkpd_seed(uint64_t base, uint64_t patient) {
    Rng r;
    r.state = base ^ (patient * 0x9E3779B97F4A7C15ULL + 0xD1B54A32D192ED03ULL);
    pkpd_splitmix(r.state);   // warm up so patient 0 and 1 are well separated
    return r;
}

// Uniform double in [0, 1).
//   Take the top 53 bits of a fresh 64-bit draw (the mantissa width of a double)
//   and scale by 2^-53 so every representable value is equally likely.
PKPD_HD inline double pkpd_uniform(Rng& r) {
    return (pkpd_splitmix(r.state) >> 11) * (1.0 / 9007199254740992.0);   // [0,1)
}

// Standard normal N(0,1) via the Box-Muller transform.
//   Deterministic and identical on host and device (it is just sqrt/log/cos of
//   two uniforms), so the sampled populations match exactly.
PKPD_HD inline double pkpd_normal(Rng& r) {
    const double u1 = 1.0 - pkpd_uniform(r);   // in (0,1] so log(u1) is finite
    const double u2 = pkpd_uniform(r);
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

// Log-normal sample: median * exp(cv * z), z ~ N(0,1).
//   Physiological PK parameters (clearance, volumes, rate constants) are strictly
//   positive and right-skewed, so pharmacometricians model between-subject
//   variability as log-normal. `cv` here is the standard deviation on the LOG
//   scale (a common shorthand for coefficient-of-variation-like spread).
PKPD_HD inline double pkpd_lognormal(Rng& r, double median, double cv) {
    return median * exp(cv * pkpd_normal(r));
}

// ---------------------------------------------------------------------------
// Population parameters: medians + variability, read from the one-line data file.
//   Units are documented per field; they matter for interpreting Cmax/AUC/effect.
// ---------------------------------------------------------------------------
struct PkPdParams {
    // ---- dosing ----
    double dose;    // oral dose (mg) placed into the gut depot at t=0
    // ---- PK medians (one-compartment oral) ----
    double ka;      // first-order absorption rate constant (1/h)
    double CL;      // plasma clearance (L/h)
    double Vc;      // central (plasma) volume of distribution (L)
    // ---- PD medians (indirect-response turnover) ----
    double kin;     // zero-order production rate of the biomarker R (response-units/h)
    double kout;    // first-order loss rate of R (1/h); baseline R0 = kin/kout
    double Imax;    // maximum fractional inhibition of loss (0..1); Imax=1 -> loss can fully stop
    double IC50;    // plasma conc giving half-maximal inhibition (mg/L)
    // ---- between-subject variability + integration ----
    double cv;      // log-normal spread applied to the sampled PK/PD parameters
    double dt;      // fixed RK4 step (h)
    int    steps;   // number of steps; simulated horizon = steps*dt hours
    int    n_patients;  // virtual population size (one GPU thread each)
    uint64_t seed;  // base RNG seed for reproducibility
};

// Per-patient summary of the coupled PK/PD trajectory.
//   PK exposure metrics (Cmax/Tmax/AUC) + PD effect metrics (Rmax, its time, and
//   the peak fractional response above baseline). These are what a pharmacometric
//   report tabulates for a virtual population.
struct PatientResult {
    double Cmax;    // peak plasma concentration (mg/L)
    double Tmax;    // time of Cmax (h)
    double AUC;     // area under the plasma concentration-time curve (mg.h/L)
    double Rmax;    // peak biomarker response R (response-units)
    double Tresp;   // time of Rmax (h)
    double effect;  // peak fractional response above baseline = (Rmax - R0)/R0
};

// ---------------------------------------------------------------------------
// The coupled PK/PD right-hand side (3 states: A_gut, A_cen, R).
//   This ONE function is the model. Both the CPU loop and the GPU thread call it,
//   so there is a single source of truth for the physics (PATTERNS.md §2).
//
//   Inputs are the current states and the (already-sampled) patient parameters.
//   Outputs (by reference) are the time derivatives. Concentration Cc = A_cen/Vc
//   is computed once and fed to the Emax inhibition term I(Cc).
// ---------------------------------------------------------------------------
PKPD_HD inline void pkpd_deriv(double Agut, double Acen, double R,
                               double ka, double CL, double Vc,
                               double kin, double kout, double Imax, double IC50,
                               double& dAgut, double& dAcen, double& dR) {
    const double Cc = Acen / Vc;                 // plasma concentration (mg/L)
    // Emax / Hill inhibition of the biomarker's LOSS. Saturates at Imax as Cc
    // grows; equals Imax/2 when Cc == IC50. This is the PK->PD coupling.
    const double inhib = Imax * Cc / (IC50 + Cc);

    dAgut = -ka * Agut;                           // gut empties into plasma
    dAcen =  ka * Agut - CL * Cc;                 // absorption in, clearance out
    // Indirect response: production kin minus loss kout*R, but the loss is slowed
    // by the drug (factor (1 - inhib)). With inhib=0 this relaxes to R0=kin/kout.
    dR    =  kin - kout * (1.0 - inhib) * R;
}

// ---------------------------------------------------------------------------
// One classical 4th-order Runge-Kutta (RK4) step on the 3-state system.
//   RK4 evaluates the derivative at four points (start, two midpoints, end) and
//   takes a weighted average. It is 4th-order accurate (local error ~dt^5) yet
//   uses no history -- ideal for a per-thread register-only integrator. We hand
//   -roll it (rather than call a library ODE solver) so the exact same arithmetic
//   runs on CPU and GPU; see THEORY §5 for the stiffness caveat.
// ---------------------------------------------------------------------------
PKPD_HD inline void pkpd_rk4(double& Agut, double& Acen, double& R,
                             double ka, double CL, double Vc,
                             double kin, double kout, double Imax, double IC50,
                             double dt) {
    double k1g,k1c,k1r, k2g,k2c,k2r, k3g,k3c,k3r, k4g,k4c,k4r;
    // k1: slope at the start of the interval.
    pkpd_deriv(Agut, Acen, R, ka,CL,Vc, kin,kout,Imax,IC50, k1g,k1c,k1r);
    // k2: slope at the midpoint using a half-step of k1.
    pkpd_deriv(Agut+0.5*dt*k1g, Acen+0.5*dt*k1c, R+0.5*dt*k1r,
               ka,CL,Vc, kin,kout,Imax,IC50, k2g,k2c,k2r);
    // k3: slope at the midpoint using a half-step of k2.
    pkpd_deriv(Agut+0.5*dt*k2g, Acen+0.5*dt*k2c, R+0.5*dt*k2r,
               ka,CL,Vc, kin,kout,Imax,IC50, k3g,k3c,k3r);
    // k4: slope at the end using a full step of k3.
    pkpd_deriv(Agut+dt*k3g, Acen+dt*k3c, R+dt*k3r,
               ka,CL,Vc, kin,kout,Imax,IC50, k4g,k4c,k4r);
    // Weighted average: (k1 + 2k2 + 2k3 + k4)/6.
    Agut += (dt/6.0)*(k1g + 2*k2g + 2*k3g + k4g);
    Acen += (dt/6.0)*(k1c + 2*k2c + 2*k3c + k4c);
    R    += (dt/6.0)*(k1r + 2*k2r + 2*k3r + k4r);
}

// ---------------------------------------------------------------------------
// Integrate ONE virtual patient end to end and return its PK/PD summary.
//   Steps:
//     1. Seed a reproducible RNG from the patient index.
//     2. Sample this patient's physiology (ka, CL, Vc, IC50) log-normally around
//        the population medians. kin/kout/Imax are kept at the population values
//        so the baseline R0 = kin/kout is a clean, shared reference.
//     3. Set initial conditions: the oral dose sits in the gut depot; plasma is
//        empty; the biomarker starts at its steady-state baseline R0 = kin/kout.
//     4. March RK4 forward, tracking Cmax/Tmax, AUC (trapezoid), and Rmax/Tresp.
//     5. Report the peak fractional effect above baseline.
//
//   Runs ENTIRELY IN REGISTERS on the GPU (no global memory during integration),
//   which is why the kernel is compute-bound and fast. `patient` is the thread's
//   global index; `P` is passed by value (a small struct) so each thread has its
//   own copy in registers/constant.
// ---------------------------------------------------------------------------
PKPD_HD inline PatientResult pkpd_integrate(const PkPdParams& P, int patient) {
    Rng r = pkpd_seed(P.seed, patient);

    // (2) Sample per-patient PK/PD physiology. Positive, right-skewed -> lognormal.
    const double ka   = pkpd_lognormal(r, P.ka,   P.cv);
    const double CL   = pkpd_lognormal(r, P.CL,   P.cv);
    const double Vc   = pkpd_lognormal(r, P.Vc,   P.cv);
    const double IC50 = pkpd_lognormal(r, P.IC50, P.cv);
    // System-level PD constants shared across patients (clean baseline reference).
    const double kin  = P.kin, kout = P.kout, Imax = P.Imax;

    // (3) Initial conditions.
    const double R0 = kin / kout;    // biomarker steady-state baseline (kin=kout*R0)
    double Agut = P.dose;            // whole oral dose starts in the gut depot (mg)
    double Acen = 0.0;               // plasma is empty at t=0
    double R    = R0;                // biomarker at baseline at t=0

    // Running summaries. Cc at t=0 is 0 (empty plasma) -> Cc_prev seeds the AUC
    // trapezoid; Rmax starts at the baseline value.
    PatientResult out{0.0, 0.0, 0.0, R0, 0.0, 0.0};
    double Cc_prev = 0.0;

    // (4) March forward. s counts completed steps; time at step s is s*dt.
    for (int s = 1; s <= P.steps; ++s) {
        pkpd_rk4(Agut, Acen, R, ka, CL, Vc, kin, kout, Imax, IC50, P.dt);
        const double t  = s * P.dt;
        const double Cc = Acen / Vc;                 // plasma concentration now
        out.AUC += 0.5 * (Cc_prev + Cc) * P.dt;      // trapezoidal AUC increment
        if (Cc > out.Cmax) { out.Cmax = Cc; out.Tmax = t; }   // track PK peak
        if (R  > out.Rmax) { out.Rmax = R;  out.Tresp = t; }  // track PD peak
        Cc_prev = Cc;
    }

    // (5) Peak fractional response above baseline. R0 > 0 by construction, so no
    // divide-by-zero; effect > 0 means the drug drove the biomarker up.
    out.effect = (out.Rmax - R0) / R0;
    return out;
}
