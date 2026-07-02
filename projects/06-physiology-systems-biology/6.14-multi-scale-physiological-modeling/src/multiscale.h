// ===========================================================================
// src/multiscale.h  --  Shared (host + device) multi-scale physiology core
// ---------------------------------------------------------------------------
// Project 6.14 : Multi-Scale Physiological Modeling
//
// WHAT THIS PROJECT COMPUTES  (the teaching version of "VPH multiscale")
//   A 1-D strand of cardiac tissue (a "cable") on which an electrical action
//   potential ("AP") is born at one end and PROPAGATES to the other -- the
//   phenomenon behind every heartbeat. Two physical scales are coupled:
//
//     * CELL scale (fine, sub-grid): at EACH node of the tissue mesh lives a
//       tiny ionic-model ODE describing that patch of membrane. We use the
//       2-variable FitzHugh-Nagumo (FHN) model -- a didactic reduction of the
//       Hodgkin-Huxley / ten-Tusscher cell models. State per node = (v, w):
//         v = fast excitation variable (a stand-in for transmembrane voltage)
//         w = slow recovery variable   (a stand-in for gating / repolarization)
//       This is the "millions of cell ODEs at quadrature points" in the catalog
//       deep dive, shrunk to a strand so it runs on a laptop GPU.
//
//     * TISSUE scale (coarse, mesh): the nodes are NOT independent -- current
//       diffuses between neighbours, which is what makes the AP travel. On the
//       1-D cable this is the diffusion (cable) term  D * d^2v/dx^2, discretized
//       as the classic 3-point Laplacian stencil.
//
//   Coupling the two scales is done by OPERATOR SPLITTING (Strang/Godunov), the
//   workhorse of the "heterogeneous multiscale method" (HMM): within one global
//   step dt we (1) advance every cell ODE with its LOCAL reaction only (the fine
//   scale, solved node-by-node in parallel), then (2) apply the tissue diffusion
//   coupling (the coarse scale, a stencil sweep). Splitting lets each scale use
//   the method that suits it and is exactly the "co-simulation coupling" idea.
//
//   The governing PDE is the MONODOMAIN cable equation:
//       dv/dt = D * d^2v/dx^2 + f_react(v, w)      (reaction-diffusion)
//       dw/dt =                 g_react(v, w)       (cell-local recovery)
//   with f_react, g_react the FHN reaction below.
//
// WHY A SHARED __host__ __device__ HEADER (PATTERNS.md section 2)
//   The per-node reaction (RK4 on the FHN ODE) and the per-node diffusion
//   stencil are written ONCE here as `__host__ __device__` inline functions.
//   The CPU reference (reference_cpu.cpp) loops over nodes calling them; the GPU
//   kernel (kernels.cu) calls the SAME functions from one thread per node. Same
//   arithmetic on both sides => the GPU result matches the CPU reference to a
//   documented tolerance (see main.cu / THEORY.md section on verification).
//   MSM_HD expands to `__host__ __device__` under nvcc, to nothing under the
//   plain host compiler (so reference_cpu.cpp can include this file too).
//
// READ THIS AFTER: util/*, and BEFORE reference_cpu.h / kernels.cuh.
// ===========================================================================
#pragma once

// --- The host/device portability macro (the HD-macro idiom) ----------------
// Under nvcc, __CUDACC__ is defined, so these functions are compiled for BOTH
// the host and the device. Under cl.exe/g++ (compiling reference_cpu.cpp), the
// CUDA keywords do not exist, so the macro must vanish.
#ifdef __CUDACC__
#define MSM_HD __host__ __device__
#else
#define MSM_HD
#endif

// ---------------------------------------------------------------------------
// FhnParams -- the fixed parameters of the FitzHugh-Nagumo cell model plus the
// tissue diffusion coefficient. Grouped in a struct so the SAME bundle can be
// passed to the host loop and copied by-value into the kernel (it is small and
// trivially copyable, so passing it as a kernel argument is fine).
//
//   a, eps (epsilon), b : shape the FHN excitation/recovery dynamics.
//       a   ~ excitation threshold (0<a<1): higher a is harder to excite.
//       eps ~ time-scale separation (small => w is much slower than v).
//       b   ~ recovery coupling: how strongly v drives w back down.
//   D : diffusion coefficient (tissue conductivity / capacitance), units
//       length^2 / time. Larger D => faster conduction velocity.
// ---------------------------------------------------------------------------
struct FhnParams {
    double a   = 0.13;   // FHN excitation threshold (dimensionless)
    double eps = 0.01;   // FHN recovery time-scale (dimensionless, small)
    double b   = 0.50;   // FHN recovery coupling (dimensionless)
    double D   = 1.0;    // tissue diffusion coefficient (space^2 / time)
};

// ---------------------------------------------------------------------------
// fhn_f : the FAST reaction term  f(v,w) = v*(v-a)*(1-v) - w.
//   This is the cubic nullcline that gives FHN its excitable "all-or-none"
//   spike: small perturbations decay, but a supra-threshold kick launches a
//   full excursion of v before w pulls it back. This term is the CELL-scale
//   physics evaluated independently at every node.
// ---------------------------------------------------------------------------
MSM_HD inline double fhn_f(double v, double w, const FhnParams& p) {
    return v * (v - p.a) * (1.0 - v) - w;
}

// ---------------------------------------------------------------------------
// fhn_g : the SLOW recovery term  g(v,w) = eps * (v - b*w).
//   eps << 1 makes w evolve on a much slower time scale than v; this
//   separation of time scales is *itself* a multi-scale feature (the "us-ms"
//   vs "ms-s" split in the catalog deep dive), captured in one cell model.
// ---------------------------------------------------------------------------
MSM_HD inline double fhn_g(double v, double w, const FhnParams& p) {
    return p.eps * (v - p.b * w);
}

// ---------------------------------------------------------------------------
// react_rk4_step : advance ONE cell's (v, w) by dt using the LOCAL reaction
//   only (no diffusion here -- that is the tissue scale, applied separately by
//   operator splitting). Classical 4th-order Runge-Kutta: evaluate the RHS at
//   four stages and combine, giving O(dt^4) accuracy and good stability for the
//   smooth-but-stiff FHN dynamics. Updates v and w in place.
//
//   This is the "fine-scale sub-grid ODE solve at a quadrature point" -- the
//   exact role SUNDIALS batch-CVODE plays in a production VPH stack; here we
//   hand-roll RK4 so nothing is a black box (CLAUDE.md section 6.1.6).
// ---------------------------------------------------------------------------
MSM_HD inline void react_rk4_step(double& v, double& w, const FhnParams& p, double dt) {
    // Stage 1: slope at the current state.
    const double k1v = fhn_f(v, w, p);
    const double k1w = fhn_g(v, w, p);
    // Stage 2: slope at the midpoint predicted by k1.
    const double k2v = fhn_f(v + 0.5 * dt * k1v, w + 0.5 * dt * k1w, p);
    const double k2w = fhn_g(v + 0.5 * dt * k1v, w + 0.5 * dt * k1w, p);
    // Stage 3: slope at the midpoint predicted by k2.
    const double k3v = fhn_f(v + 0.5 * dt * k2v, w + 0.5 * dt * k2w, p);
    const double k3w = fhn_g(v + 0.5 * dt * k2v, w + 0.5 * dt * k2w, p);
    // Stage 4: slope at the endpoint predicted by k3.
    const double k4v = fhn_f(v + dt * k3v, w + dt * k3w, p);
    const double k4w = fhn_g(v + dt * k3v, w + dt * k3w, p);
    // Weighted average of the four slopes (Simpson-like 1-2-2-1 weights).
    v += (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
    w += (dt / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w);
}

// ---------------------------------------------------------------------------
// diffusion_laplacian : the 3-point 1-D Laplacian at node i on a uniform grid,
//   (v[i-1] - 2*v[i] + v[i+1]) / dx^2 , with a ZERO-FLUX (Neumann) boundary --
//   at the ends we mirror the missing neighbour (reflecting boundary), the
//   physically correct condition for a sealed strand of tissue (no current
//   leaves the ends). `left`/`right` are the neighbour values the caller has
//   already selected (mirrored at the boundary), so this stays branch-free and
//   identical on host and device.
//
//   This is the TISSUE-scale coupling: it is what turns a field of independent
//   cell oscillators into a medium that PROPAGATES a wave. It is the classic
//   stencil pattern (see flagship 6.04 / 14.02).
// ---------------------------------------------------------------------------
MSM_HD inline double diffusion_laplacian(double left, double center, double right, double dx) {
    const double inv_dx2 = 1.0 / (dx * dx);
    return (left - 2.0 * center + right) * inv_dx2;
}

// ---------------------------------------------------------------------------
// mirror_left / mirror_right : pick a node's neighbour value, applying the
//   zero-flux boundary by reflection at the two ends. Kept as one-liners so the
//   host reference and the device kernel index the cable IDENTICALLY (any
//   mismatch here would break CPU==GPU agreement). `v` is the voltage array,
//   `n` its length, `i` the node.
// ---------------------------------------------------------------------------
MSM_HD inline double mirror_left(const double* v, int n, int i) {
    (void)n;
    return (i > 0) ? v[i - 1] : v[i + 1];       // reflect at the left end
}
MSM_HD inline double mirror_right(const double* v, int n, int i) {
    return (i < n - 1) ? v[i + 1] : v[i - 1];   // reflect at the right end
}

// ---------------------------------------------------------------------------
// CableConfig -- the whole simulation problem in one struct (loaded from the
//   sample file). Small and trivially copyable, so it can be passed BY VALUE as
//   a kernel argument (the arrays live in separate device buffers).
//
//   n      : number of tissue nodes on the cable
//   dx     : spatial step between nodes (space units)
//   dt     : global (split) time step (time units)
//   steps  : number of global steps to run (total time = steps*dt)
//   stim_nodes : the first `stim_nodes` nodes are held excited (v=1) at t=0 to
//                launch the AP from the left end.
//   p      : the FHN + diffusion parameters above.
// ---------------------------------------------------------------------------
struct CableConfig {
    int    n = 0;            // number of nodes
    double dx = 0.0;         // spatial step
    double dt = 0.0;         // time step (per global split step)
    int    steps = 0;        // number of global steps
    int    stim_nodes = 0;   // how many left-end nodes are stimulated at t=0
    FhnParams p;             // cell + tissue parameters
};

// Total simulated time (time units) -- shared by host and device reporting.
MSM_HD inline double total_time(const CableConfig& c) { return c.steps * c.dt; }
