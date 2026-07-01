// ===========================================================================
// src/tumor.h  --  Shared (host + device) tumor-growth + treatment physics
// ---------------------------------------------------------------------------
// Project 6.8 : Tumor Growth & Treatment-Response Modeling (reduced-scope teaching version)
//
// WHAT THIS PROJECT COMPUTES
//   A continuum model of an avascular tumor spreading through tissue, plus its
//   response to fractionated radiotherapy. The state is a single scalar field
//   u(x,y) in [0,1]: the LOCAL TUMOR CELL DENSITY normalized to the tissue's
//   carrying capacity (u = 1 means "packed solid with tumor cells", u = 0 means
//   "no tumor here"). Two well-known pieces of mathematical oncology drive it:
//
//   (1) GROWTH + SPREAD -- the Fisher-KPP reaction-diffusion equation:
//         du/dt = D * lap(u)  +  rho * u * (1 - u)
//                 \________/     \______________/
//                  diffusion       logistic growth
//       * D    [mm^2/day]  : how fast cells infiltrate neighbouring tissue
//                            (random motility -> a diffusion term).
//       * rho  [1/day]     : the net proliferation rate. u*(1-u) is LOGISTIC:
//                            fast growth while sparse, saturating at u = 1 (the
//                            carrying capacity -- cells run out of room/nutrient).
//       This is THE canonical PDE of glioma modeling (the "PI" model of
//       Swanson et al.); its travelling front moves at speed c = 2*sqrt(D*rho),
//       which we check analytically in THEORY / main.cu.
//
//   (2) TREATMENT -- the linear-quadratic (LQ) radiobiological survival model.
//       A radiotherapy fraction delivering dose d [Gy] kills a fraction of the
//       cells at every voxel; the SURVIVING FRACTION is
//         S(d) = exp( -(alpha*d + beta*d^2) ).
//       alpha [1/Gy], beta [1/Gy^2] are tissue radiosensitivities (their ratio
//       alpha/beta ~ 10 Gy for tumor, ~3 Gy for late-responding normal tissue,
//       is the single most-used number in the radiotherapy clinic). At each
//       scheduled fraction we simply do  u <- S(d) * u  everywhere in the field.
//
// WHY A GPU
//   Growth+spread is a pure STENCIL: each grid cell's next value depends only on
//   itself and its 4 nearest neighbours (the 5-point Laplacian). Every cell is
//   independent within a timestep, so we map ONE THREAD PER CELL and advance the
//   whole field in parallel, double-buffered (ping-pong) so there are no races.
//   This is exactly the pattern of the lattice-Boltzmann flagship (6.04) and the
//   reaction-diffusion project (14.02). The treatment step is an embarrassingly
//   parallel per-cell multiply.
//
//   The per-cell physics lives here as __host__ __device__ inline functions so
//   the CPU reference (reference_cpu.cpp) and the GPU kernel (kernels.cu) run
//   BYTE-FOR-BYTE identical math -- that is what makes verification exact. The
//   TUMOR_HD macro expands to __host__ __device__ under nvcc and to nothing
//   under the plain host compiler (which does not know those keywords). This is
//   the "HD-macro idiom" from docs/PATTERNS.md section 2.
//
// READ THIS AFTER: nothing (start here) -- then reference_cpu.h, kernels.cuh.
// ===========================================================================
#pragma once

// --- The __host__ __device__ portability shim (PATTERNS.md section 2) -------
// Under nvcc, __CUDACC__ is defined and we decorate the shared functions so they
// compile for BOTH the CPU and the GPU. Under the host compiler (cl.exe / g++)
// compiling reference_cpu.cpp, the macro vanishes and they are ordinary inlines.
#ifdef __CUDACC__
#define TUMOR_HD __host__ __device__
#else
#define TUMOR_HD
#include <cmath>   // std::exp for the host build of lq_survival() below
#endif

// ---------------------------------------------------------------------------
// TumorParams -- everything that defines one simulation.
//   Kept a plain C struct (no CUDA types) so it can be passed BY VALUE into the
//   kernel (cheap: a handful of doubles/ints) and #included by the host compiler.
//   Units are documented per field because getting them wrong is the #1 source
//   of nonsense results in PDE code.
// ---------------------------------------------------------------------------
struct TumorParams {
    int    nx, ny;      // grid size in cells (periodic boundaries; see rd_laplacian)
    double dx;          // spatial step [mm] between adjacent grid cells
    double D;           // tumor-cell diffusion (infiltration) coefficient [mm^2/day]
    double rho;         // net proliferation rate [1/day] (logistic growth)
    double dt;          // explicit-Euler timestep [day]
    int    steps;       // number of growth timesteps to integrate
    // --- treatment schedule (fractionated radiotherapy) ---
    double alpha;       // LQ linear radiosensitivity   [1/Gy]
    double beta;        // LQ quadratic radiosensitivity [1/Gy^2]
    double dose;        // dose PER fraction [Gy]
    int    n_fractions; // how many fractions to deliver (0 = untreated control)
    int    fx_interval; // deliver a fraction every this many timesteps
    // --- initial condition (a small seeded tumor) ---
    double seed_radius; // radius [mm] of the initial tumor spot at grid centre
    double seed_u;      // initial density inside the seed (0..1)
};

// ---------------------------------------------------------------------------
// tumor_laplacian: 5-point Laplacian of field f at (x,y) with PERIODIC
//   (toroidal) boundaries, in units of [field]/cell^2 (the 1/dx^2 factor is
//   applied by the caller so this stays a pure discrete operator).
//
//   lap(f) ~ f(x-1,y) + f(x+1,y) + f(x,y-1) + f(x,y+1) - 4 f(x,y)
//
//   Periodic wrap ( (x-1+nx) % nx ) keeps the tumor front from hitting a hard
//   wall; the committed sample keeps the tumor away from the edges over the run,
//   so the boundary choice does not affect the science here. The identical index
//   math on host and device is what guarantees bit-matching results.
// ---------------------------------------------------------------------------
TUMOR_HD inline double tumor_laplacian(const double* f, int x, int y, int nx, int ny) {
    const int xm = (x - 1 + nx) % nx, xp = (x + 1) % nx;   // left / right neighbours
    const int ym = (y - 1 + ny) % ny, yp = (y + 1) % ny;   // up / down neighbours
    return f[y * nx + xm] + f[y * nx + xp]
         + f[ym * nx + x] + f[yp * nx + x]
         - 4.0 * f[y * nx + x];
}

// ---------------------------------------------------------------------------
// tumor_grow_update: ONE explicit-Euler Fisher-KPP step for cell (x,y).
//   Reads u (and its neighbours) from the input buffer `u`, writes the next
//   density to the output buffer `un`. This is the per-cell kernel body shared
//   by the CPU loop and the GPU thread.
//
//   Discretization:
//     du/dt = D * lap(u)/dx^2 + rho * u * (1 - u)
//     u_new = u + dt * ( D/dx^2 * lap5(u) + rho * u * (1 - u) )
//
//   Stability (explicit Euler, 2-D): needs dt <= dx^2 / (4 D). The loader checks
//   this and the sample honours it; see THEORY "numerical considerations".
// ---------------------------------------------------------------------------
TUMOR_HD inline void tumor_grow_update(int x, int y, const TumorParams& P,
                                       const double* u, double* un) {
    const int i = y * P.nx + x;                       // row-major flat index
    const double ui  = u[i];                          // this cell's current density
    const double lap = tumor_laplacian(u, x, y, P.nx, P.ny);
    const double diffusion = (P.D / (P.dx * P.dx)) * lap;   // D * lap(u)/dx^2
    const double reaction  = P.rho * ui * (1.0 - ui);       // logistic growth
    double next = ui + P.dt * (diffusion + reaction);
    // Clamp into the physical range [0,1]: tiny explicit-Euler overshoots near
    // the front can nudge u slightly outside, which is non-physical (a density).
    if (next < 0.0) next = 0.0;
    if (next > 1.0) next = 1.0;
    un[i] = next;
}

// ---------------------------------------------------------------------------
// lq_survival: the linear-quadratic SURVIVING FRACTION after a single dose d.
//   S(d) = exp(-(alpha*d + beta*d^2)). A number in (0,1]: the fraction of cells
//   still viable after the fraction. Used by the treatment step below and quoted
//   directly in the report so the learner can sanity-check the biology.
// ---------------------------------------------------------------------------
TUMOR_HD inline double lq_survival(double alpha, double beta, double d) {
#ifdef __CUDACC__
    return exp(-(alpha * d + beta * d * d));   // device exp() (double precision)
#else
    return std::exp(-(alpha * d + beta * d * d));
#endif
}

// ---------------------------------------------------------------------------
// tumor_treat_update: apply ONE radiotherapy fraction to cell (x,y).
//   Multiplies the local density by the LQ surviving fraction. Purely local (no
//   neighbours), so it is an embarrassingly parallel per-cell multiply. We pass
//   the precomputed survival S in (it is identical for every cell) so we do not
//   recompute exp() nx*ny times -- a small but honest optimization.
// ---------------------------------------------------------------------------
TUMOR_HD inline void tumor_treat_update(int i, double survival, double* u) {
    u[i] *= survival;   // instantaneous cell kill: u <- S * u
}
