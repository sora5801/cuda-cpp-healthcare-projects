// ===========================================================================
// src/defib.h  --  Shared (host + device) 1-D monodomain cable + FitzHugh-Nagumo
//                   ionic kinetics + defibrillation shock, as ONE physics core.
// ---------------------------------------------------------------------------
// Project 6.19 : Defibrillation & High-Voltage Shock Simulation
//                (REDUCED-SCOPE teaching version -- see ../THEORY.md "real world")
//
// WHAT THIS FILE IS
//   The single source of truth for the per-cell physics of a defibrillation
//   simulation, written so that BOTH the plain-C++ CPU reference AND the CUDA
//   kernel call the exact same inline functions. That "shared __host__ __device__
//   core" idiom (docs/PATTERNS.md section 2) is what lets us VERIFY the GPU
//   result against the CPU result to (near) bit-for-bit agreement instead of
//   hand-waving "close enough".
//
//   DEFIB_HD expands to `__host__ __device__` when compiled by nvcc (so the very
//   same code runs on the GPU) and to nothing when compiled by cl.exe / g++ (so
//   the CPU reference can include this header). No __global__ / CUDA-only types
//   appear here, so the host compiler is happy.
//
// THE SCIENCE IN ONE PARAGRAPH
//   Cardiac tissue is an excitable medium: a cell at rest, if pushed above a
//   threshold, fires an action potential (a spike of transmembrane voltage V)
//   and then recovers. Neighbouring cells are electrically coupled, so a spike
//   PROPAGATES as a travelling wave -- that is the heartbeat. In fibrillation
//   the wave breaks into self-sustaining re-entrant activity. A DEFIBRILLATION
//   shock delivers a strong extracellular electric field that forces (nearly)
//   every cell past threshold at once ("virtual electrode polarization", VEP),
//   erasing the re-entrant pattern; if the whole tissue is left refractory and
//   quiescent, the arrhythmia is terminated. The minimum shock strength that
//   reliably does this is the DEFIBRILLATION THRESHOLD (DFT).
//
// THE MATH (reduced to a 1-D cable so a learner can follow every term)
//   Monodomain reaction-diffusion for the transmembrane voltage V(x,t):
//       dV/dt = D * d2V/dx2  +  f(V,w) + I_stim(x,t)
//       dw/dt = eps * (V - gamma*w)                     (slow recovery variable)
//   with the cubic FitzHugh-Nagumo (FHN) reaction  f(V,w) = V(V-a)(1-V) - w.
//   D is the diffusion (gap-junction coupling) coefficient; a is the excitation
//   threshold; eps sets the recovery timescale; gamma sets the resting state.
//   The shock enters as I_stim -- a large transient current over a short window.
//
//   We non-dimensionalise (V, w, x, t all dimensionless) which is standard for
//   FHN teaching models. THEORY.md maps these back to physical bidomain terms.
//
// READ THIS AFTER: nothing (start here), then reference_cpu.h, kernels.cuh.
// ===========================================================================
#pragma once

#include <cstddef>   // std::size_t

// --- The host/device portability macro (docs/PATTERNS.md section 2) ---------
#ifdef __CUDACC__
#define DEFIB_HD __host__ __device__
#else
#define DEFIB_HD
#endif

// ---------------------------------------------------------------------------
// FhnParams -- every physical constant of the model, in one struct so the CPU
// and GPU are configured identically. All quantities are dimensionless (see the
// header comment); THEORY.md gives the physical interpretation of each.
//   The struct is trivially copyable so we can hand it straight to a kernel by
//   value (it lands in constant/parameter memory, broadcast to every thread).
// ---------------------------------------------------------------------------
struct FhnParams {
    int    ncell   = 0;      // number of cells along the 1-D cable
    int    nsteps  = 0;      // number of time steps to integrate
    double dt      = 0.0;    // time step (dimensionless); must satisfy the
                             //   diffusion stability limit dt <= dx^2/(2D) (§numerics)
    double dx      = 0.0;    // cell-to-cell spacing (dimensionless)
    double D       = 0.0;    // diffusion / gap-junction coupling coefficient
    double a       = 0.0;    // FHN excitation threshold (0<a<1); V must exceed a to fire
    double eps     = 0.0;    // recovery rate (small -> slow recovery, long refractory period)
    double gamma   = 0.0;    // recovery coupling (sets the resting equilibrium)

    // --- Initial condition: a partially-excited cable so there is activity for
    //     the shock to act on. We pre-excite the LEFT `initial_excited` cells to
    //     V=1 (mimicking an ongoing depolarised region / re-entrant wavefront). ---
    int    initial_excited = 0;   // number of left-hand cells started at V=1

    // --- Shock protocol (the "defibrillation" part) ---
    int    shock_start = 0;       // time step at which the shock turns on
    int    shock_len   = 0;       // shock duration in time steps
    // Biphasic shocks reverse polarity halfway through (clinically ~2x more
    // effective). If `biphasic` != 0 the second half of the shock is negated.
    int    biphasic    = 0;       // 0 = monophasic, 1 = biphasic
};

// ---------------------------------------------------------------------------
// fhn_reaction: the FitzHugh-Nagumo ionic current  f(V,w) = V(V-a)(1-V) - w.
//   This is the "reaction" (ionic) term of the reaction-diffusion PDE and the
//   heart of the excitable dynamics: the cubic gives a fast upstroke and the
//   -w term (fed by the slow recovery variable) pulls the cell back down.
//   Units: dimensionless current. Called once per cell per step by BOTH paths.
// ---------------------------------------------------------------------------
DEFIB_HD inline double fhn_reaction(double V, double w, double a) {
    return V * (V - a) * (1.0 - V) - w;
}

// ---------------------------------------------------------------------------
// fhn_recovery: the slow gate ODE right-hand side  dw/dt = eps*(V - gamma*w).
//   w is the recovery variable (a lumped stand-in for the slow repolarising
//   ionic gates). eps<<1 makes it slow, producing the action-potential plateau
//   and the refractory period that a shock must overcome.
// ---------------------------------------------------------------------------
DEFIB_HD inline double fhn_recovery(double V, double w, double eps, double gamma) {
    return eps * (V - gamma * w);
}

// ---------------------------------------------------------------------------
// shock_current: the extracellular-shock stimulus current seen by cell `i` at
// step `s`, given a shock amplitude. This is the REDUCED-SCOPE model of virtual
// electrode polarization (VEP): a real bidomain shock depolarises tissue near
// one electrode and HYPERPOLARISES it near the other (opposite-sign VEP). We
// capture that essential dipole structure cheaply: the left half of the cable
// feels +amp (depolarising) and the right half feels -amp (hyperpolarising),
// during the shock window. Biphasic protocols flip the sign at the midpoint.
//   Returns the stimulus current to ADD into dV/dt for this cell/step (0 when
//   the shock is off). THEORY.md derives this from the bidomain "activating
//   function" and explains why the sign flips across the tissue.
// ---------------------------------------------------------------------------
DEFIB_HD inline double shock_current(int i, int s, int ncell,
                                     double amp, const FhnParams& p) {
    // Outside the shock time window -> no shock current.
    if (s < p.shock_start || s >= p.shock_start + p.shock_len) return 0.0;

    // Spatial VEP structure: +amp on the left half, -amp on the right half.
    // (The boundary between depolarised and hyperpolarised regions is the
    //  virtual-electrode pattern that makes shocks work -- see THEORY.md.)
    double sign_space = (i < ncell / 2) ? 1.0 : -1.0;

    // Temporal structure: biphasic shocks reverse polarity at the halfway point
    // of the shock window (this is why biphasic waveforms defibrillate at lower
    // energy in the clinic -- the reversal recharges sodium channels).
    double sign_time = 1.0;
    if (p.biphasic && (s - p.shock_start) >= p.shock_len / 2) sign_time = -1.0;

    return amp * sign_space * sign_time;
}

// ---------------------------------------------------------------------------
// cable_step: advance the WHOLE 1-D cable by ONE time step, in place, using
// operator-split forward Euler (the standard teaching integrator):
//   1. DIFFUSION (the stencil): each cell reads its two neighbours,
//        d2V/dx2 ~ (V[i-1] - 2 V[i] + V[i+1]) / dx^2      (3-point Laplacian)
//      with zero-flux (Neumann) boundaries -> the wave reflects, it does not
//      leak out the ends of the cable.
//   2. REACTION + SHOCK: add the FHN ionic current and the shock stimulus.
//   3. RECOVERY: advance the slow gate w.
//
//   V_in / w_in  : state at the start of the step (READ ONLY)
//   V_out / w_out: state at the end of the step   (WRITTEN)
//   Using SEPARATE in/out buffers (ping-pong) means every cell's update depends
//   only on the OLD field -> there are no read-after-write races, which is
//   exactly what lets the GPU run all cells in parallel and still match the CPU.
//
//   This one function is the entire per-step physics. The CPU reference loops it
//   over steps; the GPU runs one thread per shock-strength trajectory, each of
//   which loops it over its own private cable. Byte-identical math either way.
// ---------------------------------------------------------------------------
DEFIB_HD inline void cable_step(int s, double amp, const FhnParams& p,
                                const double* V_in, const double* w_in,
                                double* V_out, double* w_out) {
    const double inv_dx2 = 1.0 / (p.dx * p.dx);
    for (int i = 0; i < p.ncell; ++i) {
        // --- 1. Diffusion via the 3-point Laplacian stencil, zero-flux ends. ---
        // Neumann boundary: a ghost cell equal to the edge cell -> no gradient
        // across the boundary -> current cannot flow out the end of the cable.
        const double Vl = (i > 0)            ? V_in[i - 1] : V_in[i];
        const double Vr = (i < p.ncell - 1)  ? V_in[i + 1] : V_in[i];
        const double lap = (Vl - 2.0 * V_in[i] + Vr) * inv_dx2;

        // --- 2. Reaction (FHN ionic current) + shock stimulus. ---
        const double react = fhn_reaction(V_in[i], w_in[i], p.a);
        const double stim  = shock_current(i, s, p.ncell, amp, p);

        // Forward-Euler update of the voltage: V += dt*(D*lap + f + I_stim).
        V_out[i] = V_in[i] + p.dt * (p.D * lap + react + stim);

        // --- 3. Recovery variable (slow gate), also forward Euler. ---
        w_out[i] = w_in[i] + p.dt * fhn_recovery(V_in[i], w_in[i], p.eps, p.gamma);
    }
}

// ---------------------------------------------------------------------------
// activity_metric: a single deterministic number summarising how much
// electrical activity remains in the cable AFTER the shock. We use the mean of
// V clamped to the excited range, i.e. the fraction of "how depolarised" the
// tissue is on average. A successfully defibrillated cable settles toward rest
// (V ~ 0) so this metric is small; a cable still carrying a wave keeps it high.
//   Returned as a plain double so both paths compute it identically.
// ---------------------------------------------------------------------------
DEFIB_HD inline double activity_metric(int ncell, const double* V) {
    double acc = 0.0;
    for (int i = 0; i < ncell; ++i) {
        // Count only genuinely depolarised voltage (V>0.1) so tiny numerical
        // ringing near rest does not register as "activity".
        double v = V[i];
        acc += (v > 0.1) ? v : 0.0;
    }
    return acc / static_cast<double>(ncell);
}
