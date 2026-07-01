// ===========================================================================
// src/cardiac_cell.h  --  Shared (host + device) cardiac monodomain physics
// ---------------------------------------------------------------------------
// Project 6.1 : Cardiac Electrophysiology Simulation
//
// THE MODEL  (read ../THEORY.md for the full science->math->GPU derivation)
//   Cardiac tissue is an EXCITABLE MEDIUM: a resting cell that is nudged past a
//   threshold fires an "action potential" -- a fast upstroke, a plateau, then a
//   slow recovery -- and, because cells are electrically coupled, that spike
//   PROPAGATES to neighbours as a travelling wave (the heartbeat's electrical
//   front). We simulate this with the MONODOMAIN reaction-diffusion PDE:
//
//       dV/dt = D * laplacian(V)  -  I_ion(V, w)          (transmembrane voltage)
//       dw/dt = f(V, w)                                    (a recovery variable)
//
//   * V(x,y,t)  is the (dimensionless) transmembrane voltage at a tissue point.
//   * w(x,y,t)  is a slow "recovery"/"gating" variable that turns the cell off
//               again (its biophysical analogue: the aggregate of ion-channel
//               gates that repolarise and refractory-block the cell).
//   * D         is the diffusion coefficient (electrotonic coupling / area).
//   * I_ion     is the local ionic current -- the REACTION term (per-cell ODE).
//
//   REACTION term: we use the FITZHUGH-NAGUMO (FHN) cell model -- the canonical
//   two-variable teaching reduction of the full 50-200-variable ionic models
//   (ten Tusscher-Panfilov, O'Hara-Rudy). FHN keeps the qualitative excitable
//   dynamics (rest / threshold / upstroke / plateau / refractoriness) with just
//   V and w, so a learner can SEE the mechanism without drowning in Markov gates.
//   The real models plug in HERE, unchanged elsewhere -- see THEORY "real world".
//
//       I_ion(V,w) = V*(V - a)*(V - 1) + w          (cubic N-shaped nonlinearity)
//       f(V,w)     = eps * (V - b*w)                 (slow linear recovery)
//
//   OPERATOR SPLITTING (Godunov): we advance one dt in TWO decoupled half-steps
//   per timestep -- first the pointwise REACTION ODE (embarrassingly parallel,
//   one thread per cell), then the DIFFUSION stencil (nearest-neighbour). This
//   is exactly how production cardiac solvers (openCARP, MonoAlg3D) decouple the
//   stiff per-cell ODE from the sparse spatial coupling.
//
//   WHY ONE SHARED HEADER: the per-cell physics below is decorated
//   __host__ __device__ (via CARDIAC_HD), so the CPU reference and the GPU
//   kernels call the SAME functions and produce byte-for-byte identical math.
//   That is what makes GPU-vs-CPU verification exact rather than approximate
//   (docs/PATTERNS.md section 2). Keep CUDA-only constructs (no __global__) out
//   of this file so the plain host compiler can include it too.
//
// READ THIS AFTER: reference_cpu.h (the parameter struct), then kernels.cuh.
// ===========================================================================
#pragma once

#include <cstddef>   // std::size_t

// CARDIAC_HD expands to __host__ __device__ under nvcc (so the function exists
// on both CPU and GPU) and to nothing under the plain host compiler (which does
// not understand those decorators). This is the HD-macro idiom.
#ifdef __CUDACC__
#define CARDIAC_HD __host__ __device__
#else
#define CARDIAC_HD
#endif

// ---------------------------------------------------------------------------
// MonodomainParams -- the full problem definition (grid + model + stimulus).
//   Kept as plain-old-data so it can be copied to the device by value and is
//   safe to include from both reference_cpu.cpp (host) and kernels.cu (nvcc).
//   Units are dimensionless (FHN is a nondimensional caricature); THEORY maps
//   them to physical ms / mV / mm.
// ---------------------------------------------------------------------------
struct MonodomainParams {
    // --- Grid (a square patch of tissue, row-major, nx across, ny down) ------
    int nx = 0;          // grid columns (x)
    int ny = 0;          // grid rows    (y)
    int steps = 0;       // number of full reaction+diffusion timesteps

    // --- Numerics ------------------------------------------------------------
    double dt = 0.0;     // time step (explicit; must satisfy the CFL bound below)
    double dx = 0.0;     // spatial step (grid spacing; same in x and y)
    double D  = 0.0;     // diffusion coefficient (electrotonic coupling)

    // --- FitzHugh-Nagumo reaction parameters ---------------------------------
    double a   = 0.0;    // excitation threshold (0<a<1); cell fires if V>a
    double eps = 0.0;    // recovery time-scale (small => slow recovery, long AP)
    double b   = 0.0;    // recovery coupling (controls refractory/return to rest)

    // --- Stimulus (S1): a small square patch is depolarised at t=0 -----------
    int stim_x0 = 0, stim_y0 = 0;   // top-left corner of the stimulus patch
    int stim_w  = 0, stim_h  = 0;   // patch width/height (cells)
    double stim_v = 0.0;            // voltage the patch is clamped to at t=0
};

// Flat row-major index of grid cell (x,y): consecutive x are contiguous, so a
// warp of threads with consecutive x reads/writes coalesced global memory.
CARDIAC_HD inline std::size_t cell_idx(int x, int y, int nx) {
    return static_cast<std::size_t>(y) * nx + x;
}

// ---------------------------------------------------------------------------
// react_step: advance ONE cell's (V,w) by one dt using ONLY the local reaction
//   ODE (no spatial coupling). Forward Euler on the FHN system:
//
//       I_ion = V*(V-a)*(V-1) + w
//       V <- V + dt * ( -I_ion )      (reaction half of dV/dt = D*lap(V) - I_ion)
//       w <- w + dt * eps*(V - b*w)
//
//   Uses the OLD V inside f(V,w) (explicit/simultaneous update) so the CPU loop
//   and the GPU one-thread-per-cell version compute identically. This is the
//   per-cell ODE the GPU parallelises across ~10^8 cells in a real heart; here
//   it is the REACTION half-step of the operator split.
//
//   Parameters:
//     V,w : in/out cell state (voltage, recovery) -- updated in place.
//     p   : model parameters (a, eps, b, dt).
//   Returns: nothing (mutates *V,*w). Complexity: O(1) per cell.
// ---------------------------------------------------------------------------
CARDIAC_HD inline void react_step(double* V, double* w, const MonodomainParams& p) {
    const double v = *V;                       // snapshot: use OLD V in both eqns
    const double I_ion = v * (v - p.a) * (v - 1.0) + *w;   // cubic N-shaped current
    const double v_new = v + p.dt * (-I_ion);              // reaction half of dV/dt
    const double w_new = *w + p.dt * p.eps * (v - p.b * (*w));  // slow recovery
    *V = v_new;
    *w = w_new;
}

// ---------------------------------------------------------------------------
// diffuse_cell: compute the DIFFUSION half-step for one cell (x,y). Explicit
//   forward-Euler on dV/dt = D * laplacian(V), using the standard 5-point
//   Laplacian on a uniform grid with NO-FLUX (Neumann) boundaries:
//
//       lap(V) ~ ( V_left + V_right + V_up + V_down - 4*V_center ) / dx^2
//
//   No-flux boundary: a neighbour that would fall off the tissue edge is
//   replaced by the centre value (mirror), so V_left-V_center = 0 there -- i.e.
//   no current leaves the tissue (insulated heart boundary). This is a nearest-
//   neighbour STENCIL: each cell reads only its 4 neighbours from the read-only
//   input buffer and writes its own output -> no races, no atomics, perfect for
//   the ping-pong GPU pattern (docs/PATTERNS.md section 1, like 6.04 / 14.02).
//
//   Parameters:
//     x,y    : this cell's grid coordinates.
//     Vin    : read-only voltage field (size nx*ny) BEFORE this half-step.
//     p      : parameters (nx, ny, dt, dx, D).
//   Returns: the new voltage at (x,y) after the diffusion half-step.
//   Complexity: O(1) per cell (4 neighbour reads).
// ---------------------------------------------------------------------------
CARDIAC_HD inline double diffuse_cell(int x, int y, const double* Vin,
                                      const MonodomainParams& p) {
    const std::size_t c = cell_idx(x, y, p.nx);
    const double vc = Vin[c];
    // Mirror (no-flux) neighbours: clamp the index to stay on the grid, which
    // makes an off-grid neighbour equal to the centre -> zero gradient there.
    const double vl = (x > 0)         ? Vin[cell_idx(x - 1, y, p.nx)] : vc;
    const double vr = (x < p.nx - 1)  ? Vin[cell_idx(x + 1, y, p.nx)] : vc;
    const double vu = (y > 0)         ? Vin[cell_idx(x, y - 1, p.nx)] : vc;
    const double vd = (y < p.ny - 1)  ? Vin[cell_idx(x, y + 1, p.nx)] : vc;
    const double lap = (vl + vr + vu + vd - 4.0 * vc) / (p.dx * p.dx);
    return vc + p.dt * p.D * lap;              // explicit Euler diffusion update
}

// ---------------------------------------------------------------------------
// cfl_limit: the largest stable dt for the EXPLICIT diffusion stencil on this
//   grid: 2-D forward Euler diffusion is stable only for
//       dt <= dx^2 / (4*D).
//   We expose it so main.cu can print it and callers can sanity-check the
//   sample. (The reaction ODE adds its own, looser, stability constraint.)
// ---------------------------------------------------------------------------
CARDIAC_HD inline double cfl_limit(const MonodomainParams& p) {
    return (p.dx * p.dx) / (4.0 * p.D);
}
