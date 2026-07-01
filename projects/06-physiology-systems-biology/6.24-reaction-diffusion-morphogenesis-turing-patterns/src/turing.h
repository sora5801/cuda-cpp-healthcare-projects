// ===========================================================================
// src/turing.h  --  Shared (host + device) Gierer-Meinhardt Turing update
// ---------------------------------------------------------------------------
// Project 6.24 : Reaction-Diffusion Morphogenesis (Turing Patterns)
//
// WHAT THIS PROJECT COMPUTES
//   Alan Turing's 1952 insight: two chemicals ("morphogens") that DIFFUSE at
//   different speeds and REACT nonlinearly can spontaneously break a uniform
//   state into a stationary spatial PATTERN -- spots, stripes, labyrinths.
//   This is the leading mathematical model for how a featureless embryo lays
//   down periodic structure: leopard spots, zebrafish stripes, hair-follicle
//   spacing, digit patterning. We simulate the canonical activator-inhibitor
//   system on a 2-D grid and watch a pattern emerge from a nearly-uniform seed.
//
// THE MODEL (Gierer-Meinhardt 1972, the textbook Turing activator-inhibitor)
//   Let a(x,y,t) = ACTIVATOR concentration, h(x,y,t) = INHIBITOR concentration.
//
//     da/dt = Da * lap(a)  +  rho * a^2 / h  -  mu_a * a  +  rho_a
//     dh/dt = Dh * lap(h)  +  rho * a^2      -  mu_h * h
//
//   Reading the reaction terms as biology:
//     * a^2/h  : the activator AUTOCATALYSES (a^2, positive feedback) but is
//                damped by the inhibitor (division by h). "Short-range
//                activation."
//     * a^2    : the activator PRODUCES the inhibitor, which then diffuses away
//                fast (Dh >> Da) and suppresses activation elsewhere.
//                "Long-range inhibition."
//     * -mu*   : linear decay of each species.
//     * rho_a  : a small basal activator source (breaks the trivial a=0 state).
//
//   The Turing condition is Dh/Da >> 1 (inhibitor diffuses much faster than the
//   activator). Under it, tiny noise around the uniform steady state grows into
//   a pattern with a characteristic wavelength -- see THEORY.md "dispersion
//   relation." lap() is the 5-point Laplacian (the discrete diffusion operator).
//
// WHY A GPU
//   Every grid cell's next value depends only on itself and its 4 neighbours --
//   a pure STENCIL (PATTERNS.md §1; cf. lattice-Boltzmann 6.04 and reaction-
//   diffusion 14.02). All cells advance independently, so we give each cell its
//   own thread and DOUBLE-BUFFER (ping-pong) the fields: read the frozen old
//   state, write the new state, swap. No two threads write the same cell, so
//   there are no data races and no atomics.
//
//   The per-cell physics lives here as `__host__ __device__` inline functions so
//   the CPU reference (reference_cpu.cpp, host compiler) and the GPU kernel
//   (kernels.cu, nvcc) run BYTE-FOR-BYTE identical arithmetic. That makes the
//   GPU-vs-CPU verification meaningful. TU_HD expands to `__host__ __device__`
//   under nvcc and to nothing under the plain host compiler (HD-macro idiom,
//   PATTERNS.md §2).
//
// READ THIS AFTER: README.md and THEORY.md (the "why").
// READ THIS BEFORE: kernels.cu (the GPU twin) and reference_cpu.cpp (the CPU twin).
// ===========================================================================
#pragma once

// -- The host/device portability shim -------------------------------------
// When compiled by nvcc, __CUDACC__ is defined and we decorate the shared
// functions so they can run on BOTH the host and the device. When this header
// is pulled into reference_cpu.cpp by the ordinary C++ compiler, __CUDACC__ is
// NOT defined, the decorators would be a syntax error, so we blank them out.
#ifdef __CUDACC__
#define TU_HD __host__ __device__
#else
#define TU_HD
#endif

// ---------------------------------------------------------------------------
// TuringParams  --  everything that defines one simulation run.
//
// Grouped so a single struct can be passed by value into a kernel (it is small
// and trivially copyable, so this is cheap and avoids pointer bookkeeping).
// All fields are read from the one-line sample file (see data/README.md).
// ---------------------------------------------------------------------------
struct TuringParams {
    int    nx, ny;     // grid width / height in cells (domain is nx*ny cells)
    double Da, Dh;     // diffusion coefficients: activator (small), inhibitor (large)
    double rho;        // reaction strength (scales a^2/h production and a^2 source)
    double mu_a;       // activator linear decay rate
    double mu_h;       // inhibitor linear decay rate
    double rho_a;      // basal (source) production of activator; breaks a==0
    double dt;         // explicit-Euler timestep (must be small for stability)
    int    steps;      // number of timesteps to integrate
    int    noise_seed; // seed for the deterministic initial perturbation (below)
};

// ---------------------------------------------------------------------------
// tu_baseline_activator  --  the spatially-uniform steady state a* (the fixed
//                            point we perturb to start a pattern).
//
// The homogeneous steady state (a*, h*) sets both reaction terms to zero with
// no spatial variation (lap()==0):
//     0 = rho*a*^2 / h*  -  mu_a*a*  +  rho_a          (activator balance)
//     0 = rho*a*^2       -  mu_h*h*                     (inhibitor balance)
// From the second equation:  h* = (rho/mu_h) * a*^2.
// Substituting into the first, the rho*a*^2 cancels neatly:
//     rho*a*^2 / ((rho/mu_h)*a*^2) = mu_h,  so the first becomes
//     mu_h - mu_a*a* + rho_a = 0   =>   a* = (mu_h + rho_a) / mu_a.
// This is an EXACT closed form (see THEORY.md §"The math"). We seed the fields
// at (a*, h*) plus tiny noise, so the simulation starts from the true unstable
// equilibrium -- exactly the setup Turing's linear-stability theory analyzes,
// which lets main.cu's dispersion-relation check match the observed pattern.
//
// Returns a* = (mu_h + rho_a)/mu_a. Kept deterministic and identical on host and
// device so the CPU and GPU seed from the same numbers.
// ---------------------------------------------------------------------------
TU_HD inline double tu_baseline_activator(const TuringParams& P) {
    const double denom = (P.mu_a > 1e-12) ? P.mu_a : 1e-12;   // guard mu_a -> 0
    return (P.mu_h + P.rho_a) / denom;
}

// ---------------------------------------------------------------------------
// tu_baseline_inhibitor  --  the matching steady inhibitor level h* for a given
//                            activator level a*, from h* = (rho/mu_h)*a*^2.
// Used to seed the inhibitor field at the true fixed point (not just a flat
// guess), so the initial state is a genuine homogeneous equilibrium.
// ---------------------------------------------------------------------------
TU_HD inline double tu_baseline_inhibitor(const TuringParams& P, double a_star) {
    const double mu_h = (P.mu_h > 1e-12) ? P.mu_h : 1e-12;   // guard mu_h -> 0
    return P.rho * a_star * a_star / mu_h;
}

// ---------------------------------------------------------------------------
// tu_laplacian  --  5-point discrete Laplacian of field f at cell (x,y).
//
// The Laplacian is the diffusion operator: it measures how much a cell's value
// differs from the average of its 4 neighbours. On a unit grid spacing:
//   lap(f)[x,y] = f(x-1,y)+f(x+1,y)+f(x,y-1)+f(x,y+1) - 4*f(x,y).
//
// BOUNDARY: we use PERIODIC (toroidal) wrap so the domain has no edges -- a
// common, artefact-free choice for pattern-formation studies (a leopard's flank
// has no special boundary). The modulo arithmetic wraps index -1 -> nx-1 and
// nx -> 0. (Neumann/zero-flux is the biological alternative; see THEORY.)
//
// Parameters:
//   f       : pointer to an nx*ny field, row-major (index = y*nx + x)
//   x, y    : the cell whose Laplacian we want (0 <= x < nx, 0 <= y < ny)
//   nx, ny  : grid dimensions
// Returns the scalar Laplacian value (units: concentration, since h=1 grid unit).
// ---------------------------------------------------------------------------
TU_HD inline double tu_laplacian(const double* f, int x, int y, int nx, int ny) {
    const int xm = (x - 1 + nx) % nx;   // left neighbour column (wrapped)
    const int xp = (x + 1)      % nx;   // right neighbour column (wrapped)
    const int ym = (y - 1 + ny) % ny;   // upper neighbour row (wrapped)
    const int yp = (y + 1)      % ny;   // lower neighbour row (wrapped)
    return f[y * nx + xm] + f[y * nx + xp]     // left  + right
         + f[ym * nx + x] + f[yp * nx + x]     // up    + down
         - 4.0 * f[y * nx + x];                // minus 4x the centre
}

// ---------------------------------------------------------------------------
// tu_update  --  ONE explicit-Euler Gierer-Meinhardt step for cell (x,y).
//
// This is the single source of truth for the physics. Both the CPU loop and the
// GPU kernel call it, so their results agree to floating-point rounding.
//
// It reads the CURRENT activator/inhibitor fields (a, h) -- including the 4
// neighbours via the Laplacian -- and writes the NEXT state (an, hn). Because
// input and output are SEPARATE buffers (ping-pong), the whole grid updates as
// if simultaneously, which is what the PDE semantics require.
//
// Parameters:
//   x, y : cell coordinates
//   P    : all model + grid parameters (by const-ref; trivially copyable)
//   a, h : CURRENT activator / inhibitor fields (nx*ny, row-major) -- read only
//   an,hn: NEXT activator / inhibitor fields (nx*ny) -- written at index (x,y)
// Side effects: writes an[i] and hn[i] where i = y*nx + x. No other cell touched.
// Complexity: O(1) per call (4 neighbour reads + a handful of flops).
// ---------------------------------------------------------------------------
TU_HD inline void tu_update(int x, int y, const TuringParams& P,
                            const double* a, const double* h,
                            double* an, double* hn) {
    const int i = y * P.nx + x;          // this cell's flat, row-major index
    const double av = a[i];              // current activator concentration here
    const double hv = h[i];              // current inhibitor concentration here

    // Diffusion (spatial coupling): each species spreads by its own D * lap.
    const double la = tu_laplacian(a, x, y, P.nx, P.ny);
    const double lh = tu_laplacian(h, x, y, P.nx, P.ny);

    // Reaction (local kinetics). Guard the 1/h division so a transient h -> 0
    // (numerically) cannot produce a NaN; hv stays positive in practice because
    // the inhibitor has a source term and starts positive.
    const double h_safe = (hv > 1e-12) ? hv : 1e-12;
    const double autocat = P.rho * av * av / h_safe;   // rho * a^2 / h : self-activation, inhibitor-damped
    const double a_source = P.rho * av * av;           // rho * a^2     : activator drives inhibitor production

    // Explicit (forward) Euler: new = old + dt * (diffusion + reaction).
    // Kept in the SAME term order on host and device so rounding matches.
    an[i] = av + P.dt * (P.Da * la + autocat - P.mu_a * av + P.rho_a);
    hn[i] = hv + P.dt * (P.Dh * lh + a_source - P.mu_h * hv);
}
