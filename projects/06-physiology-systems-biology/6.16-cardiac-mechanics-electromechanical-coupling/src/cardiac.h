// ===========================================================================
// src/cardiac.h  --  Shared (host + device) 0-D cardiac electromechanics model
// ---------------------------------------------------------------------------
// Project 6.16 : Cardiac Mechanics & Electromechanical Coupling
//
// WHAT THIS PROJECT COMPUTES  (REDUCED-SCOPE TEACHING VERSION)
//   The full research problem (see ../THEORY.md "Where this sits in the real
//   world") couples a stiff ionic + cross-bridge ODE at EVERY Gauss point of a
//   3-D nonlinear finite-element mesh of hyperelastic myocardium, and solves a
//   global Newton-Raphson equilibrium at each timestep. That is a multi-week
//   FEM code. To TEACH the electromechanical-coupling idea on a GPU while
//   staying honest and reproducible, we collapse the tissue to a SINGLE 0-D
//   representative cell + a lumped-parameter ventricle (the classic
//   time-varying ELASTANCE model of Suga & Sagawa) closed by a Windkessel
//   circulation. The coupling CHAIN a learner must understand is all here:
//
//     electrical activation  ->  intracellular calcium transient  ->
//     cross-bridge recruitment (active state)  ->  ventricular ELASTANCE E(t)
//     ->  chamber pressure P = E(t)*(V - V0)  ->  valves open/close  ->
//     volume ejected into a Windkessel  ->  pressure-volume (PV) loop.
//
//   "Voltage causes calcium, calcium causes force (stiffness), force ejects
//   blood" -- that is the whole electromechanics story, at 0-D. The GPU angle
//   is the ENSEMBLE: we solve this same ODE for thousands of virtual hearts (a
//   sweep over contractility and afterload), one GPU thread per heart. That is
//   exactly the catalog's "batch ODE, one integration point per thread" pattern
//   -- our integration points are whole hearts instead of Gauss points, which
//   keeps the demo runnable on any machine (PATTERNS.md section 1, ensemble RK4).
//
//   The state vector y = (Ca, xb, V, Pa) integrated by RK4:
//     Ca : intracellular free calcium concentration      [micromolar, uM]
//     xb : cross-bridge / active-state variable           [dimensionless 0..~1]
//     V  : left-ventricular blood volume                  [mL]
//     Pa : arterial (Windkessel) pressure                 [mmHg]
//
//   The per-element PHYSICS lives here as __host__ __device__ inline functions,
//   so the CPU reference (reference_cpu.cpp) and the GPU kernel (kernels.cu)
//   evaluate BYTE-FOR-BYTE identical math -> results agree to round-off. CARD_HD
//   expands to `__host__ __device__` under nvcc and to nothing under the plain
//   host compiler (PATTERNS.md section 2, the HD-macro idiom).
//
//   NOT FOR CLINICAL USE. All numbers are synthetic and illustrative.
//
// READ THIS AFTER: nothing (start here), then reference_cpu.h, kernels.cuh.
// ===========================================================================
#pragma once

// -- The HD macro: portable "run on both CPU and GPU" decoration -------------
//   Under nvcc (__CUDACC__ defined) we compile these inline functions for BOTH
//   host and device. Under the plain host compiler (cl.exe / g++) compiling
//   reference_cpu.cpp the CUDA keywords do not exist, so the macro must vanish.
//   Keeping ZERO CUDA-only types in this header is what lets the host compiler
//   include it (PATTERNS.md section 2).
#ifdef __CUDACC__
#define CARD_HD __host__ __device__
#else
#define CARD_HD
#endif

// ---------------------------------------------------------------------------
// HeartParams -- the fixed physiology for one simulated ventricle.
//   Units are annotated on every field. These are lumped, illustrative values
//   in the physiological ballpark of a human left ventricle; they are NOT
//   patient-specific and must not be read as clinical measurements.
// ---------------------------------------------------------------------------
struct HeartParams {
    // ---- Electrical activation / stimulus timing -----------------------
    double t_activate  = 0.0;  // time within the cycle when the cell fires [ms]

    // ---- Calcium transient (phenomenological difference-of-exponentials) --
    //   A depolarisation at t_activate injects a calcium "pulse"; calcium then
    //   decays back to rest. This stands in for the L-type Ca influx + SERCA
    //   re-uptake of a full ionic model, captured by two time constants.
    double Ca_rest     = 0.0;  // diastolic resting calcium                [uM]
    double Ca_amp      = 0.0;  // peak calcium added by the release        [uM]
    double tau_rise_ms = 0.0;  // calcium upstroke time constant           [ms]
    double tau_decay_ms= 0.0;  // calcium decay (re-uptake) time constant  [ms]

    // ---- Cross-bridge / active state (Rice-Wang-Bers-inspired) ----------
    //   Calcium binds troponin and recruits cross-bridges; the recruited
    //   fraction `xb` relaxes toward a calcium-dependent steady state (a Hill
    //   curve). This recruited fraction scales the muscle's active stiffness.
    double Tref        = 0.0;  // peak active elastance scale (contractility) [mmHg/mL]
    double Ca50        = 0.0;  // calcium giving half-max activation        [uM]
    double nH          = 0.0;  // Hill coefficient (cooperativity)          [-]
    double k_xb_ms     = 0.0;  // cross-bridge rate constant                [1/ms]

    // ---- Chamber mechanics: time-varying elastance ---------------------
    //   Suga-Sagawa: chamber pressure P = E(t)*(V - V0), with the instantaneous
    //   elastance E(t) = Emin + Tref*xb. In diastole xb~0 so E~Emin (a soft,
    //   compliant chamber that fills easily); in systole xb rises so E climbs to
    //   Emin+Tref (a stiff chamber that pressurises and ejects). Tref is thus
    //   the CONTRACTILITY knob. V0 is the unloaded (zero-pressure) volume.
    double Emin        = 0.0;  // diastolic (passive) elastance             [mmHg/mL]
    double V0_mL       = 0.0;  // unloaded (zero-pressure) volume            [mL]

    // ---- Valves (smooth diode-like resistances) ------------------------
    //   Aortic valve: opens (ejects) when ventricular P exceeds arterial Pa.
    //   Mitral valve: opens (fills) when the venous filling pressure P_ven
    //   exceeds ventricular P. Each valve is a linear resistance while open.
    double R_ao        = 0.0;  // aortic (ejection) resistance        [mmHg*ms/mL]
    double R_mv        = 0.0;  // mitral (filling) resistance         [mmHg*ms/mL]
    double P_ven       = 0.0;  // venous filling (preload) pressure         [mmHg]

    // ---- Windkessel afterload (2-element: R_sys and C) -----------------
    //   The systemic arteries downstream of the aortic valve are a resistance
    //   R_sys (systemic vascular resistance = AFTERLOAD) draining a compliance
    //   C toward a diastolic floor. Raising R_sys models hypertension.
    double R_sys       = 0.0;  // systemic vascular resistance (afterload) [mmHg*ms/mL]
    double C_art       = 0.0;  // arterial compliance                     [mL/mmHg]
    double P_art_dias  = 0.0;  // diastolic arterial floor pressure         [mmHg]
};

// ---------------------------------------------------------------------------
// CycleResult -- the clinically meaningful summary of one PV loop.
//   These are the "clinical outputs" the catalog names (PV loop, ejection
//   fraction, wall stress), reduced to scalars so we can print + verify them.
// ---------------------------------------------------------------------------
struct CycleResult {
    double EDV_mL;        // end-diastolic volume  (max V over the last beat)  [mL]
    double ESV_mL;        // end-systolic volume   (min V over the last beat)  [mL]
    double SV_mL;         // stroke volume = EDV - ESV                          [mL]
    double EF_percent;    // ejection fraction = 100 * SV / EDV                 [%]
    double P_peak_mmHg;   // peak ventricular pressure over the last beat       [mmHg]
    double stress_peak;   // peak wall-stress proxy (Laplace P*V^(1/3))     [arb. units]
};

// ---------------------------------------------------------------------------
// hill_activation -- steady-state cross-bridge activation for a given calcium.
//   Classic Hill curve f(Ca) = Ca^n / (Ca50^n + Ca^n): monotone in Ca, in
//   [0,1]. This is the calcium sensitivity of the myofilament. pow() is
//   available on both host and device.
// ---------------------------------------------------------------------------
CARD_HD inline double hill_activation(double Ca, double Ca50, double nH) {
    const double c = (Ca > 0.0) ? Ca : 0.0;   // calcium is physically >= 0
    const double cn = pow(c, nH);             // Ca^n
    const double c50n = pow(Ca50, nH);        // Ca50^n
    return cn / (c50n + cn);                  // in [0,1)
}

// ---------------------------------------------------------------------------
// calcium_target -- the instantaneous "driven" calcium level at cycle-phase t.
//   Phenomenological transient: before activation calcium sits at rest; after
//   the cell fires at t_activate it rises then decays. We model the released
//   calcium as a difference-of-exponentials pulse (fast rise, slower decay), a
//   shape widely used to approximate the measured Ca transient. The ODE (below)
//   then relaxes the true state Ca toward this target for a smooth trajectory.
// ---------------------------------------------------------------------------
CARD_HD inline double calcium_target(double t_in_cycle, const HeartParams& p) {
    const double dt_since = t_in_cycle - p.t_activate;
    if (dt_since < 0.0) return p.Ca_rest;          // not yet activated this beat
    const double tr = p.tau_rise_ms, td = p.tau_decay_ms;
    const double e_dec = exp(-dt_since / td);
    const double e_ris = exp(-dt_since / tr);
    // Analytic peak of (e^{-s/td} - e^{-s/tr}) at s* = ln(td/tr)/(1/tr - 1/td);
    // divide by its height so the pulse peaks exactly at Ca_amp.
    const double s_star = log(td / tr) / (1.0 / tr - 1.0 / td);
    const double peak = exp(-s_star / td) - exp(-s_star / tr);
    const double shape = (e_dec - e_ris) / (peak > 0.0 ? peak : 1.0);  // 0..1
    return p.Ca_rest + p.Ca_amp * shape;
}

// ---------------------------------------------------------------------------
// elastance -- instantaneous chamber elastance E(t) = Emin + Tref*xb.
//   The single scalar that turns "how activated the muscle is" (xb) into "how
//   stiff the chamber is". Diastole: E~Emin (compliant, fills). Systole: E
//   climbs (stiff, pressurises). Tref is the contractility knob swept by the
//   ensemble.
// ---------------------------------------------------------------------------
CARD_HD inline double elastance(double xb, const HeartParams& p) {
    return p.Emin + p.Tref * xb;   // [mmHg/mL]
}

// ---------------------------------------------------------------------------
// ventricular_pressure -- chamber pressure from the time-varying elastance.
//   P = E(t) * (V - V0), clamped at 0 (the chamber cannot pull a vacuum). This
//   is the pressure that drives the valves and the PV loop.
// ---------------------------------------------------------------------------
CARD_HD inline double ventricular_pressure(double V, double xb, const HeartParams& p) {
    const double P = elastance(xb, p) * (V - p.V0_mL);   // [mmHg]
    return (P > 0.0) ? P : 0.0;
}

// ---------------------------------------------------------------------------
// State -- the 4-variable ODE state we integrate with RK4.
// ---------------------------------------------------------------------------
struct State {
    double Ca;    // intracellular calcium   [uM]
    double xb;    // cross-bridge activation  [-]
    double V;     // ventricular volume       [mL]
    double Pa;    // arterial pressure        [mmHg]
};

// ---------------------------------------------------------------------------
// deriv -- the right-hand side dy/dt of the coupled electromechanics ODE.
//   Every term is annotated; this is the heart of the model.
//     dCa/dt : calcium relaxes toward the driven target with the rise time
//              constant on the way up and the decay one on the way down.
//     dxb/dt : cross-bridges recruit toward their calcium-set steady state
//              hill_activation(Ca) at rate k_xb -- this LAG makes tension trail
//              calcium (the electromechanical delay).
//     dV/dt  : volume balance across two one-way valves:
//                * mitral (filling):  q_mv = (P_ven - P)/R_mv  when P_ven > P
//                * aortic (ejection): q_ao = (P - Pa)/R_ao     when P > Pa
//              dV = q_mv - q_ao.
//     dPa/dt : 2-element Windkessel. Ejected blood q_ao flows into the arterial
//              compliance C; the arteries drain toward P_art_dias through the
//              systemic resistance R_sys (the AFTERLOAD).
//                dPa = (q_ao - (Pa - P_art_dias)/R_sys) / C.
//   The whole state (including Pa) is integrated together so ventricle-artery
//   coupling is consistent. This formulation is numerically WELL-BEHAVED at the
//   demo's timestep (see THEORY numerics): flows are bounded by finite valve
//   resistances rather than the unbounded orifice used in a naive model.
// ---------------------------------------------------------------------------
CARD_HD inline void deriv(const State& y, double t_in_cycle,
                          const HeartParams& p, State& dy) {
    // --- Calcium: relax toward the phenomenological driven target ---------
    const double Ca_tgt = calcium_target(t_in_cycle, p);
    const double tau_ca = (Ca_tgt > y.Ca) ? p.tau_rise_ms : p.tau_decay_ms;
    dy.Ca = (Ca_tgt - y.Ca) / tau_ca;

    // --- Cross-bridges: first-order recruitment toward Hill(Ca) -----------
    const double xb_ss = hill_activation(y.Ca, p.Ca50, p.nH);   // 0..1 target
    dy.xb = p.k_xb_ms * (xb_ss - y.xb);

    // --- Pressures + valve flows ------------------------------------------
    const double P = ventricular_pressure(y.V, y.xb, p);        // ventricular [mmHg]
    const double q_ao = (P > y.Pa)   ? (P - y.Pa)   / p.R_ao : 0.0;   // ejection out
    const double q_mv = (p.P_ven > P) ? (p.P_ven - P) / p.R_mv : 0.0; // filling in
    dy.V = q_mv - q_ao;                                         // [mL/ms]

    // --- Windkessel arterial pressure -------------------------------------
    const double runoff = (y.Pa - p.P_art_dias) / p.R_sys;      // peripheral drain
    dy.Pa = (q_ao - runoff) / p.C_art;                          // [mmHg/ms]
}

// ---------------------------------------------------------------------------
// rk4_step -- one classical 4th-order Runge-Kutta step of the FULL coupled
//   system (Ca, xb, V, Pa), advancing time by dt. RK4 evaluates the derivative
//   at four stages and combines them with weights (1,2,2,1)/6 (O(dt^4) local
//   error). Shared verbatim by the CPU reference and the GPU kernel -> identical
//   arithmetic -> near-exact verification.
// ---------------------------------------------------------------------------
CARD_HD inline void rk4_step(State& y, double t_in_cycle,
                             const HeartParams& p, double dt) {
    State k1, k2, k3, k4;

    // Stage 1: derivative at the current point.
    deriv(y, t_in_cycle, p, k1);
    // Stage 2: derivative at the midpoint using k1.
    State y2 = { y.Ca + 0.5*dt*k1.Ca, y.xb + 0.5*dt*k1.xb,
                 y.V + 0.5*dt*k1.V,  y.Pa + 0.5*dt*k1.Pa };
    deriv(y2, t_in_cycle + 0.5*dt, p, k2);
    // Stage 3: derivative at the midpoint using k2.
    State y3 = { y.Ca + 0.5*dt*k2.Ca, y.xb + 0.5*dt*k2.xb,
                 y.V + 0.5*dt*k2.V,  y.Pa + 0.5*dt*k2.Pa };
    deriv(y3, t_in_cycle + 0.5*dt, p, k3);
    // Stage 4: derivative at the end using k3.
    State y4 = { y.Ca + dt*k3.Ca, y.xb + dt*k3.xb,
                 y.V + dt*k3.V,  y.Pa + dt*k3.Pa };
    deriv(y4, t_in_cycle + dt, p, k4);

    // Combine the four stages.
    y.Ca += (dt / 6.0) * (k1.Ca + 2.0*k2.Ca + 2.0*k3.Ca + k4.Ca);
    y.xb += (dt / 6.0) * (k1.xb + 2.0*k2.xb + 2.0*k3.xb + k4.xb);
    y.V  += (dt / 6.0) * (k1.V  + 2.0*k2.V  + 2.0*k3.V  + k4.V );
    y.Pa += (dt / 6.0) * (k1.Pa + 2.0*k2.Pa + 2.0*k3.Pa + k4.Pa);
}

// ---------------------------------------------------------------------------
// integrate_cycle -- integrate ONE virtual heart for `n_beats` cardiac cycles
//   and return the PV-loop summary of the LAST (steady-state) beat.
//
//   Why several beats: the ventricle-Windkessel system needs a few cycles to
//   settle into a LIMIT CYCLE (its transient depends on the initial volume). We
//   discard the warm-up and summarise the final beat -- exactly how a limit
//   cycle is extracted in cardiovascular modeling.
//
//   Deterministic: fixed dt, fixed step count -> byte-identical CPU & GPU.
// ---------------------------------------------------------------------------
CARD_HD inline CycleResult integrate_cycle(const HeartParams& p, double dt,
                                           int steps_per_beat, int n_beats) {
    // Initial condition: relaxed cell, a partly-filled chamber, arteries at the
    // diastolic floor. The exact start is washed out by the warm-up beats.
    State y = { p.Ca_rest, 0.0, p.V0_mL + 80.0, p.P_art_dias };

    double edv = -1.0e300, esv = 1.0e300, ppeak = 0.0, speak = 0.0;
    const int last_beat = n_beats - 1;

    for (int beat = 0; beat < n_beats; ++beat) {
        const bool record = (beat == last_beat);
        for (int s = 0; s < steps_per_beat; ++s) {
            const double t_in_cycle = s * dt;      // phase within this beat
            rk4_step(y, t_in_cycle, p, dt);
            if (record) {
                if (y.V > edv) edv = y.V;           // end-diastolic volume
                if (y.V < esv) esv = y.V;           // end-systolic volume
                const double P = ventricular_pressure(y.V, y.xb, p);
                if (P > ppeak) ppeak = P;
                // Wall-stress proxy via Laplace law for a sphere: sigma ~ P*r,
                // r ~ V^(1/3). Reported in arbitrary (relative) units.
                const double stress = P * pow(y.V, 1.0 / 3.0);
                if (stress > speak) speak = stress;
            }
        }
    }

    CycleResult out;
    out.EDV_mL      = edv;
    out.ESV_mL      = esv;
    out.SV_mL       = edv - esv;
    out.EF_percent  = (edv > 0.0) ? 100.0 * (edv - esv) / edv : 0.0;
    out.P_peak_mmHg = ppeak;
    out.stress_peak = speak;
    return out;
}
