// ===========================================================================
// src/purkinje.h  --  Shared (host + device) Purkinje cable model + solver
// ---------------------------------------------------------------------------
// Project 6.17 : Purkinje System & Conduction System Modeling
//
// WHAT THIS PROJECT COMPUTES (reduced-scope teaching version -- see THEORY.md
// "Where this sits in the real world" for the full research problem)
//   The cardiac conduction system (SA node -> AV node -> His bundle -> bundle
//   branches -> Purkinje fibre network) carries the electrical impulse that
//   triggers a coordinated ventricular contraction. Each Purkinje fibre behaves
//   like a 1-D excitable CABLE: a voltage wave (the action potential) ignites at
//   one end and PROPAGATES along the fibre at a well-defined conduction velocity
//   (CV). The CV of a cable is set by its passive electrical properties (fibre
//   diameter -> axial/diffusion coefficient D) and its active membrane kinetics.
//
//   We simulate an ENSEMBLE of independent 1-D Purkinje cables (think: the many
//   fascicles / bundle branches of the tree, each with its own diameter and
//   coupling). For each cable we solve the monodomain CABLE EQUATION
//
//         dV/dt = D * d2V/dx2  +  f(V, w)                 (reaction-diffusion)
//         dw/dt = g(V, w)                                 (recovery variable)
//
//   with a compact 2-variable excitable membrane model (a FitzHugh-Nagumo /
//   Aliev-Panfilov-style reaction term -- a didactic stand-in for the 20-ODE
//   Stewart-Zhang Purkinje ionic model). We stimulate the LEFT end, watch the
//   activation front sweep to the RIGHT end, and MEASURE the conduction velocity
//   from the arrival times -- exactly the quantity clinicians calibrate for the
//   His-Purkinje system. Finally a small tree graph turns per-branch CVs into a
//   total ventricular activation time via graph-based conduction delays.
//
//   GPU MAPPING: each cable is an INDEPENDENT PDE solve (space loop x time loop),
//   so we give each cable its own GPU thread -- the same "ensemble of solvers"
//   pattern as the epidemiology flagship 9.02, applied to a spatial PDE. Because
//   the per-cable stepper lives here as __host__ __device__ inline code, the CPU
//   reference (reference_cpu.cpp) and the GPU kernel (kernels.cu) run BYTE-FOR-
//   BYTE identical arithmetic, so verification is exact to round-off.
//
//   PK_HD expands to __host__ __device__ under nvcc, and to nothing under the
//   plain host compiler (so reference_cpu.cpp can include this header too). This
//   is the "HD-macro" idiom from docs/PATTERNS.md section 2.
//
// READ THIS AFTER: nothing (start here), then reference_cpu.h, kernels.cuh.
// ===========================================================================
#pragma once

// --- The host/device portability macro -------------------------------------
// Under nvcc (__CUDACC__ defined) we mark the shared math callable from BOTH the
// host and a kernel. Under cl.exe/g++ these decorators do not exist, so we blank
// them out. NOTHING CUDA-specific (no __global__, no <cuda_runtime.h>) may live
// in this header, or the host compiler could not include it.
#ifdef __CUDACC__
#define PK_HD __host__ __device__
#else
#define PK_HD
#endif

// Fixed maximum number of spatial nodes per cable. We size the per-thread work
// arrays to this constant so the GPU kernel can keep the two voltage buffers in
// LOCAL memory (per-thread private storage) with no dynamic allocation. A real
// solver would use one thread per NODE and shared/global buffers; here one
// thread owns a WHOLE cable, which is the simplest correct ensemble mapping and
// keeps the teaching focus on "many independent PDE solves in parallel".
#define PK_MAX_NODES 256

// ---------------------------------------------------------------------------
// CableParams  --  everything that defines ONE Purkinje cable's simulation.
//   Kept as plain-old-data so it copies trivially to the device (by value into
//   the kernel, or inside an array). Units are noted per field; SI-ish but
//   nondimensionalised where the reaction model is nondimensional.
// ---------------------------------------------------------------------------
struct CableParams {
    int    n_nodes;     // number of spatial grid points along the cable (<= PK_MAX_NODES)
    double length_mm;   // physical cable length (mm); dx = length_mm/(n_nodes-1)
    double D;           // diffusion (axial) coefficient (mm^2/ms) -- larger fibre => larger D => faster CV
    double dt_ms;       // explicit-Euler time step (ms). Must satisfy the stability limit (THEORY §numerics)
    int    n_steps;     // number of time steps to integrate (total time = n_steps*dt_ms ms)
    double stim_amp;    // stimulus current amplitude added to the left nodes during the stimulus window
    double stim_dur_ms; // how long (ms) the left-end stimulus is applied
    int    stim_width;  // how many left-end nodes receive the stimulus (a small pacing electrode)
    double thresh;      // activation threshold on V used to timestamp the front (dimensionless V units)
    int    parent;      // index of the parent cable in the tree (-1 for the His-bundle root)
    double delay_ms;    // fixed junction/gap-junction delay (ms) added when the impulse enters this cable
};

// ---------------------------------------------------------------------------
// CableResult  --  the per-cable measured outputs (all deterministic).
//   These are integers/doubles produced by identical host+device arithmetic, so
//   they compare EXACTLY between CPU and GPU (activation-step indices are ints;
//   the tree activation time is a deterministic sum). See main.cu for the check.
// ---------------------------------------------------------------------------
struct CableResult {
    int    activate_step_in;   // time step at which the LEFT (proximal) end first crossed threshold
    int    activate_step_out;  // time step at which the RIGHT (distal) end first crossed threshold
    double cv_mm_per_ms;       // measured conduction velocity = length / (time between the two crossings)
    int    captured;           // 1 if the distal end activated (wave propagated), else 0 (conduction block)
};

// ---------------------------------------------------------------------------
// pk_reaction  --  the active membrane reaction term f(V,w) and recovery g(V,w).
//   We use the Aliev-Panfilov two-variable excitable model, a compact, widely
//   used didactic surrogate for detailed cardiac ionic models. It reproduces the
//   qualitative action-potential shape (fast upstroke, plateau, recovery) with
//   only two state variables:
//
//       f(V,w) = -k*V*(V-a)*(V-1) - V*w          (fast excitation + repolarisation)
//       g(V,w) =  eps(V,w)*(-w - k*V*(V-a-1))    (slow recovery)
//
//   V is a nondimensional transmembrane potential in [0,1] (0 = rest, ~1 = peak);
//   w is a nondimensional recovery current. The constants below are the standard
//   Aliev-Panfilov values. We keep them as compile-time constants (not tunable
//   per cable) so the reaction kinetics are identical across the ensemble and the
//   ONLY thing that changes CV is the passive diffusion D -- making the CV-vs-D
//   relationship (the teaching point) clean to read off.
// ---------------------------------------------------------------------------
PK_HD inline void pk_reaction(double V, double w, double& fV, double& gw) {
    // Aliev-Panfilov constants (dimensionless). a = excitation threshold,
    // k sets the upstroke steepness, mu0/mu1/eps0 shape the recovery time scale.
    const double a    = 0.15;
    const double k    = 8.0;
    const double eps0 = 0.002;
    const double mu1  = 0.2;
    const double mu2  = 0.3;

    // Reaction (excitation) current: cubic in V gives the bistable "flip" from
    // rest to excited, and the -V*w term pulls V back down during recovery.
    fV = -k * V * (V - a) * (V - 1.0) - V * w;

    // Voltage-dependent recovery rate eps(V,w): slow at high V (plateau), faster
    // as V falls -- this asymmetry is what gives a realistic AP duration.
    const double eps = eps0 + (mu1 * w) / (V + mu2);

    // Recovery variable derivative.
    gw = eps * (-w - k * V * (V - a - 1.0));
}

// ---------------------------------------------------------------------------
// pk_simulate_cable  --  integrate ONE cable's monodomain equation and measure
//                        its conduction velocity. Shared by CPU + GPU.
//
//   ALGORITHM (explicit finite differences, forward Euler in time):
//     * State: V[i], w[i] for i in [0, n_nodes). Two V buffers (Vcur/Vnew) so the
//       Laplacian reads the OLD field while we write the NEW field (no in-place
//       aliasing -- the classic "ping-pong" for an explicit stencil).
//     * Each step:
//         - interior nodes: d2V/dx2 ~= (V[i-1] - 2V[i] + V[i+1]) / dx^2   (3-point stencil)
//         - boundaries: zero-flux (Neumann) -> copy the neighbour (sealed-end cable)
//         - V[i]_new = V[i] + dt*( D*d2V/dx2 + f(V[i],w[i]) ) + stimulus
//         - w[i]_new = w[i] + dt*g(V[i],w[i])
//     * We inject a stimulus at the left `stim_width` nodes for `stim_dur_ms`.
//     * We timestamp when node 0 and node n_nodes-1 first cross `thresh` -> the
//       front's entry and exit times -> conduction velocity.
//
//   The two V buffers are passed in by the caller (Vbuf_a, Vbuf_b) so this
//   function allocates nothing -- essential for the GPU where each thread keeps
//   its buffers in local memory (see kernels.cu).
//
//   COMPLEXITY: O(n_steps * n_nodes) per cable. The whole ensemble is
//   O(n_cables * n_steps * n_nodes); the GPU parallelises across cables.
// ---------------------------------------------------------------------------
PK_HD inline CableResult pk_simulate_cable(const CableParams& p,
                                           double* Vbuf_a, double* Vbuf_b,
                                           double* wbuf) {
    const int    n  = p.n_nodes;
    const double dx = p.length_mm / (double)(n - 1);   // node spacing (mm)
    const double inv_dx2 = 1.0 / (dx * dx);            // precomputed 1/dx^2 for the stencil

    // Initialise: cable at rest (V=0, w=0). Vcur points at buffer A to start.
    double* Vcur = Vbuf_a;
    double* Vnew = Vbuf_b;
    for (int i = 0; i < n; ++i) { Vcur[i] = 0.0; wbuf[i] = 0.0; }

    // How many steps the stimulus is held on (ceil of stim_dur/dt).
    const int stim_steps = (int)(p.stim_dur_ms / p.dt_ms + 0.5);

    // Sentinel -1 means "never crossed threshold yet".
    int step_in  = -1;   // activation step of the proximal (left) end, node 0
    int step_out = -1;   // activation step of the distal (right) end, node n-1

    // --- time-stepping loop -------------------------------------------------
    for (int s = 0; s < p.n_steps; ++s) {
        const bool stim_on = (s < stim_steps);

        // Spatial sweep: compute the new V at every node from the OLD field.
        for (int i = 0; i < n; ++i) {
            // Zero-flux (Neumann) boundaries model a SEALED cable end: the ghost
            // node equals the interior neighbour, so no current leaks out the end.
            const double vL = (i == 0)     ? Vcur[i + 1] : Vcur[i - 1];
            const double vR = (i == n - 1) ? Vcur[i - 1] : Vcur[i + 1];
            const double lap = (vL - 2.0 * Vcur[i] + vR) * inv_dx2;   // discrete d2V/dx2

            double fV, gw;
            pk_reaction(Vcur[i], wbuf[i], fV, gw);

            // Pacing electrode on the left `stim_width` nodes during the window.
            const double stim = (stim_on && i < p.stim_width) ? p.stim_amp : 0.0;

            // Forward-Euler update of the reaction-diffusion PDE.
            Vnew[i]  = Vcur[i] + p.dt_ms * (p.D * lap + fV + stim);
            wbuf[i] += p.dt_ms * gw;   // recovery variable advances in place (no spatial coupling)
        }

        // Record first threshold crossings at the two ends (front entry/exit).
        if (step_in  < 0 && Vnew[0]     >= p.thresh) step_in  = s;
        if (step_out < 0 && Vnew[n - 1] >= p.thresh) step_out = s;

        // Ping-pong: the buffer we just wrote becomes "current" for the next step.
        double* tmp = Vcur; Vcur = Vnew; Vnew = tmp;
    }

    // --- assemble the result ------------------------------------------------
    CableResult r;
    r.activate_step_in  = step_in;
    r.activate_step_out = step_out;
    r.captured          = (step_out >= 0) ? 1 : 0;

    // Conduction velocity = distance / time-of-flight between the two ends. We
    // use the WHOLE-cable length as distance and the crossing-step gap as time.
    // If the wave never reached the far end (block), CV is 0 by convention.
    if (step_out >= 0 && step_in >= 0 && step_out > step_in) {
        const double tof_ms = (double)(step_out - step_in) * p.dt_ms;   // time of flight (ms)
        r.cv_mm_per_ms = p.length_mm / tof_ms;                           // mm/ms  (= m/s)
    } else {
        r.cv_mm_per_ms = 0.0;
    }
    return r;
}
