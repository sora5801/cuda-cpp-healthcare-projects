// ===========================================================================
// src/heart.h  --  Shared (host + device) 0-D whole-heart twin + RK4 integrator
// ---------------------------------------------------------------------------
// Project 6.2 : Whole-Heart Digital Twin   (REDUCED-SCOPE TEACHING VERSION)
//
// WHY THIS FILE EXISTS (the single most important idiom in this repo)
//   The per-element physics of the heart model lives here as `__host__ __device__`
//   inline functions, so the CPU reference (reference_cpu.cpp, compiled by the
//   host compiler) and the GPU kernel (kernels.cu, compiled by nvcc) run the
//   *byte-for-byte identical math*. That is what makes GPU-vs-CPU verification
//   EXACT to round-off instead of approximate. HEART_HD expands to
//   `__host__ __device__` under nvcc and to nothing under the host compiler.
//   (PATTERNS.md section 2, "the shared HD core".)
//
// -------------------------------------------------------------------------
// WHAT A "WHOLE-HEART DIGITAL TWIN" IS, AND WHAT WE SIMPLIFY
// -------------------------------------------------------------------------
//   A real cardiac digital twin (openCARP, simcardems, TorchCor) couples FOUR
//   things on a patient-specific tetrahedral mesh of the heart:
//     1. Electrophysiology (EP): a reaction-diffusion PDE (mono/bidomain) that
//        propagates the action potential across millions of nodes.
//     2. Active mechanics: nonlinear elasticity where the tissue CONTRACTS in
//        response to the local electrical activation (active stress/strain).
//     3. Circulation: lumped Windkessel boundary conditions feeding the
//        ventricular cavity pressure/volume.
//     4. Inference: thousands of forward solves to fit parameters to a specific
//        patient's clinical measurements (ensemble Kalman filter / adjoint).
//
//   Solving (1)+(2) as full 3-D FEM is research-grade and needs a mesh, a sparse
//   solver, and days of compute -- out of scope for a single teaching project
//   (CLAUDE.md section 13: ship the simplest correct teaching version, describe the
//   full one in THEORY.md "real world"). So we keep the SAME FOUR PHYSICS BLOCKS
//   but collapse the spatial PDE to a 0-D (spatially-lumped, "single cell + single
//   chamber") ordinary-differential-equation (ODE) model:
//
//     EP        -> FitzHugh-Nagumo (FHN) 2-variable action-potential oscillator.
//     Mechanics -> a time-varying-elastance ventricle whose contractility is
//                  gated by the FHN activation variable (active contraction).
//     Circulation -> a 3-element Windkessel (R_c, C, R_p) arterial load.
//     Inference -> an ENSEMBLE sweep over a physiological parameter, from which
//                  main.cu picks the member that best matches a target stroke
//                  volume (a 1-parameter stand-in for twin calibration).
//
//   This is a genuine (if minimal) closed-loop electro-mechanical-hemodynamic
//   heart model: it produces a pressure-volume loop, a stroke volume, and an
//   ejection fraction -- the very quantities a twin is calibrated against.
//
// -------------------------------------------------------------------------
// THE STATE VECTOR  y = (v, w, V, P)     [4 doubles per heart]
// -------------------------------------------------------------------------
//   v : FHN fast (membrane-potential-like) variable  [dimensionless, ~ -0.2..1.1]
//   w : FHN slow recovery variable                    [dimensionless]
//   V : left-ventricular volume                       [mL]
//   P : arterial (aortic/windkessel) pressure         [mmHg]
//
//   The ventricular pressure is an ALGEBRAIC function of (V, activation), not a
//   state, so we do not integrate it -- see ventricular_pressure() below.
//
// READ THIS AFTER: util/cuda_check.cuh; BEFORE: reference_cpu.h, kernels.cuh.
// ===========================================================================
#pragma once

// HEART_HD: the host+device decorator that makes CPU and GPU share this math.
#ifdef __CUDACC__
#define HEART_HD __host__ __device__
#else
#define HEART_HD
#endif

// ---------------------------------------------------------------------------
// HeartParams -- the fixed physiological constants of ONE virtual heart.
//   Grouping them in a struct (rather than a long argument list) keeps the RK4
//   plumbing readable and lets us pass "one heart" by value into a kernel thread.
//   Units are documented per field; they are the currency of the whole model.
// ---------------------------------------------------------------------------
struct HeartParams {
    // --- FitzHugh-Nagumo electrophysiology (dimensionless, classic form) ----
    double fhn_a    = 0.13;   // excitation threshold parameter
    double fhn_b    = 0.013;  // recovery coupling
    double fhn_c1   = 0.26;   // cubic-term gain (shapes upstroke)
    double fhn_c2   = 0.10;   // recovery decay rate
    double fhn_d    = 1.0;    // recovery time-scale
    double stim_amp = 0.55;   // periodic stimulus amplitude (triggers each beat)
    double stim_dur = 2.0;    // stimulus duration [ms]
    double bcl_ms   = 800.0;  // basic cycle length (beat period) [ms] -> 75 bpm

    // --- Time-varying-elastance ventricle (active mechanics) ----------------
    // Ventricular pressure P_lv = E(t) * (V - V0), with E(t) ramped between a
    // diastolic (relaxed) and systolic (contracted) elastance by the FHN
    // activation. Contractility Emax is the classic index of pump strength.
    double E_min    = 0.06;   // diastolic elastance [mmHg/mL] (compliant, filling)
    double E_max    = 2.2;    // systolic elastance / contractility [mmHg/mL]
    double V0       = 10.0;   // unstressed ventricular volume [mL]

    // --- Mitral / aortic valve + filling ------------------------------------
    double P_venous = 8.0;    // left-atrial filling pressure [mmHg]
    double R_mitral = 0.006;  // mitral (filling) resistance [mmHg*s/mL]
    double R_aortic = 0.008;  // aortic (ejection) resistance [mmHg*s/mL]

    // --- 3-element Windkessel arterial load ---------------------------------
    // The classic lumped systemic circulation: a characteristic resistance R_c
    // in series with a parallel (compliance C, peripheral resistance R_p).
    double Rc       = 0.03;   // characteristic (aortic) resistance [mmHg*s/mL]
    double C_art    = 1.6;    // arterial compliance [mL/mmHg]
    double Rp       = 1.0;    // peripheral (systemic) resistance [mmHg*s/mL]
};

// ---------------------------------------------------------------------------
// activation(v) -- map the FHN fast variable v to a 0..1 contraction signal.
//   The mechanics do not care about the exact membrane voltage, only "how
//   activated is the muscle right now". We clamp and normalise v into [0,1];
//   v ~ 0 (resting) -> no contraction, v ~ 1 (plateau) -> full contraction.
// ---------------------------------------------------------------------------
HEART_HD inline double activation(double v) {
    double a = v;                       // FHN v already sits roughly in [0,1]
    if (a < 0.0) a = 0.0;               // clamp below (diastolic rest)
    if (a > 1.0) a = 1.0;               // clamp above (full plateau)
    return a;
}

// ---------------------------------------------------------------------------
// current_elastance -- the time-varying elastance E(t) of the ventricle.
//   E interpolates between the relaxed diastolic value E_min and the peak
//   systolic value E_max according to the EP activation. This is the coupling
//   knob that turns "the cell fired" into "the chamber squeezes".
// ---------------------------------------------------------------------------
HEART_HD inline double current_elastance(const HeartParams& p, double v) {
    const double act = activation(v);                 // 0 (relaxed) .. 1 (contracted)
    return p.E_min + (p.E_max - p.E_min) * act;       // linear elastance ramp
}

// ---------------------------------------------------------------------------
// ventricular_pressure -- P_lv = E(t) * (V - V0)  [mmHg].
//   The end-systolic pressure-volume relationship, the workhorse of 0-D cardiac
//   mechanics. Returns 0 (not negative) below V0 so a nearly-empty ventricle
//   does not generate suction in this teaching model.
// ---------------------------------------------------------------------------
HEART_HD inline double ventricular_pressure(const HeartParams& p, double V, double v) {
    const double E = current_elastance(p, v);
    const double Plv = E * (V - p.V0);
    return (Plv > 0.0) ? Plv : 0.0;
}

// ---------------------------------------------------------------------------
// stimulus_current -- the periodic pacing stimulus I_stim(t).
//   A real twin's EP is paced by a propagating wavefront; here a periodic pulse
//   of width stim_dur every bcl_ms models the sino-atrial "beat clock". `t_ms`
//   is elapsed time in milliseconds.
// ---------------------------------------------------------------------------
HEART_HD inline double stimulus_current(const HeartParams& p, double t_ms) {
    // Phase within the current beat: how far are we past the last stimulus?
    // fmod is available on both host (<cmath>) and device (CUDA math), so this
    // line compiles identically on CPU and GPU.
    double phase = t_ms - p.bcl_ms * (double)((long long)(t_ms / p.bcl_ms));
    return (phase < p.stim_dur) ? p.stim_amp : 0.0;   // on during the pulse only
}

// ---------------------------------------------------------------------------
// heart_deriv -- the right-hand side of the coupled ODE system dy/dt = f(t, y).
//   Writes the four derivatives (dv, dw, dV, dP). This is the ONE place the
//   physics is defined; RK4 below just samples it. `t_ms` is time in ms.
//
//   Coupling map (who drives whom):
//     FHN (v,w) --activation--> elastance E(t) --> ventricular pressure P_lv
//     P_lv vs P_venous  -> mitral inflow  (fills V)
//     P_lv vs P (aorta) -> aortic outflow (ejects V, charges the Windkessel)
//     Windkessel (P)    -> arterial pressure decays through R_p, charged by flow
// ---------------------------------------------------------------------------
HEART_HD inline void heart_deriv(const HeartParams& p, double t_ms,
                                 double v, double w, double V, double P,
                                 double& dv, double& dw, double& dV, double& dP) {
    // ---- (1) Electrophysiology: FitzHugh-Nagumo (time in ms) --------------
    // dv/dt = c1*v*(v-a)*(1-v) - c2*w + I_stim   (fast excitation, cubic)
    // dw/dt = b*(v - d*w)                        (slow recovery)
    // Rates are per-ms so the whole system integrates in milliseconds.
    const double Istim = stimulus_current(p, t_ms);
    dv = p.fhn_c1 * v * (v - p.fhn_a) * (1.0 - v) - p.fhn_c2 * w + Istim;
    dw = p.fhn_b * (v - p.fhn_d * w);

    // ---- (2) Ventricular pressure (algebraic, from mechanics) -------------
    const double Plv = ventricular_pressure(p, V, v);   // [mmHg]

    // ---- (3) Valve flows (diode-like: open only under a forward gradient) --
    // Convert resistances (mmHg*s/mL) to per-ms by dividing the pressure drop
    // by R and by 1000 (ms per s) so dV/dt is in mL/ms, matching dt in ms.
    double q_in  = 0.0;   // mitral inflow  (atrium -> ventricle) [mL/ms]
    double q_out = 0.0;   // aortic outflow (ventricle -> aorta)  [mL/ms]
    if (p.P_venous > Plv) {                     // mitral valve OPEN (filling)
        q_in = (p.P_venous - Plv) / (p.R_mitral * 1000.0);
    }
    if (Plv > P) {                              // aortic valve OPEN (ejection)
        q_out = (Plv - P) / (p.R_aortic * 1000.0);
    }
    dV = q_in - q_out;                          // ventricular volume balance

    // ---- (4) 3-element Windkessel arterial pressure -----------------------
    // The aorta is charged by ejected flow q_out and drains through R_p; the
    // characteristic resistance Rc adds the fast pressure kick at ejection.
    // C*dP/dt = q_out - P/R_p  (compliance charge balance), converted to /ms.
    dP = (q_out - P / (p.Rp * 1000.0)) / p.C_art;
    (void)w;  // w is a state but not needed algebraically here beyond dw above
}

// ---------------------------------------------------------------------------
// rk4_step -- one classical 4th-order Runge-Kutta step of size dt_ms (ms).
//   RK4 samples heart_deriv at four stages and combines them for O(dt^4) local
//   error -- accurate and stable for these smooth ODEs. Advancing (v,w,V,P) in
//   place. Time `t_ms` is the step's start time (the stimulus depends on it).
// ---------------------------------------------------------------------------
HEART_HD inline void rk4_step(const HeartParams& p, double t_ms, double dt_ms,
                              double& v, double& w, double& V, double& P) {
    double k1v, k1w, k1V, k1P;
    double k2v, k2w, k2V, k2P;
    double k3v, k3w, k3V, k3P;
    double k4v, k4w, k4V, k4P;
    const double h = dt_ms;

    // Stage 1: slope at the start of the interval.
    heart_deriv(p, t_ms,             v,             w,             V,             P,
                k1v, k1w, k1V, k1P);
    // Stage 2: slope at the midpoint using the stage-1 slope.
    heart_deriv(p, t_ms + 0.5*h,     v+0.5*h*k1v,   w+0.5*h*k1w,   V+0.5*h*k1V,   P+0.5*h*k1P,
                k2v, k2w, k2V, k2P);
    // Stage 3: slope at the midpoint using the stage-2 slope.
    heart_deriv(p, t_ms + 0.5*h,     v+0.5*h*k2v,   w+0.5*h*k2w,   V+0.5*h*k2V,   P+0.5*h*k2P,
                k3v, k3w, k3V, k3P);
    // Stage 4: slope at the end using the stage-3 slope.
    heart_deriv(p, t_ms + h,         v+h*k3v,       w+h*k3w,       V+h*k3V,       P+h*k3P,
                k4v, k4w, k4V, k4P);

    // Weighted average (Simpson-like) of the four slopes.
    v += (h / 6.0) * (k1v + 2.0*k2v + 2.0*k3v + k4v);
    w += (h / 6.0) * (k1w + 2.0*k2w + 2.0*k3w + k4w);
    V += (h / 6.0) * (k1V + 2.0*k2V + 2.0*k3V + k4V);
    P += (h / 6.0) * (k1P + 2.0*k2P + 2.0*k3P + k4P);
}

// ---------------------------------------------------------------------------
// TwinResult -- the per-heart summary the analysis (and twin fit) care about.
//   These are the clinically meaningful outputs of one forward simulation:
//   the pressure-volume loop is condensed into stroke volume, ejection
//   fraction, and peak pressure -- exactly the metrics a twin is calibrated to.
// ---------------------------------------------------------------------------
struct TwinResult {
    double edv;         // end-diastolic volume (max V over the last beat)   [mL]
    double esv;         // end-systolic  volume (min V over the last beat)   [mL]
    double stroke_vol;  // EDV - ESV                                          [mL]
    double ejection_fr; // stroke_vol / EDV                                   [-]
    double peak_plv;    // peak left-ventricular pressure over the last beat [mmHg]
    double peak_pao;    // peak arterial (aortic) pressure over the last beat[mmHg]
};

// ---------------------------------------------------------------------------
// simulate_heart -- integrate ONE virtual heart for `beats` cardiac cycles and
//   return its steady-state summary. Shared verbatim by the CPU reference and
//   the GPU kernel, so both produce identical numbers.
//
//   We run several beats to wash out the initial transient, then measure the
//   pressure-volume extremes over the FINAL beat only (that is the "converged"
//   loop a clinician would read). `dt_ms` is the RK4 step in milliseconds.
//
//   Determinism note: every operation here is plain double arithmetic in a
//   fixed order, so the result is bit-reproducible run to run (PATTERNS.md 3).
// ---------------------------------------------------------------------------
HEART_HD inline TwinResult simulate_heart(const HeartParams& p, double dt_ms, int beats) {
    // Physiological-ish initial condition: relaxed cell, a filled ventricle,
    // and a diastolic arterial pressure.
    double v = 0.0;      // resting membrane variable
    double w = 0.0;      // resting recovery variable
    double V = 120.0;    // start near end-diastolic volume [mL]
    double P = 75.0;     // start near mean arterial pressure [mmHg]

    const int    steps_per_beat = (int)(p.bcl_ms / dt_ms + 0.5);  // e.g. 800/0.1 = 8000
    const int    total_steps    = steps_per_beat * beats;
    const int    last_beat_start = steps_per_beat * (beats - 1);  // measure final beat

    // Running extremes over the LAST beat (initialise to opposite infinities).
    double edv = -1.0e300, esv = 1.0e300;   // max / min ventricular volume
    double peak_plv = 0.0, peak_pao = 0.0;  // peak LV and arterial pressure

    for (int s = 0; s < total_steps; ++s) {
        const double t_ms = s * dt_ms;      // absolute time (drives the stimulus)
        rk4_step(p, t_ms, dt_ms, v, w, V, P);

        if (s >= last_beat_start) {          // only the converged final beat counts
            if (V > edv) edv = V;
            if (V < esv) esv = V;
            const double Plv = ventricular_pressure(p, V, v);
            if (Plv > peak_plv) peak_plv = Plv;
            if (P   > peak_pao) peak_pao = P;
        }
    }

    TwinResult r;
    r.edv         = edv;
    r.esv         = esv;
    r.stroke_vol  = edv - esv;
    r.ejection_fr = (edv > 0.0) ? (edv - esv) / edv : 0.0;
    r.peak_plv    = peak_plv;
    r.peak_pao    = peak_pao;
    return r;
}
