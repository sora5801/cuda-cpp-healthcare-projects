// ===========================================================================
// src/nse_channel.h  --  Shared (host + device) incompressible Navier-Stokes core
// ---------------------------------------------------------------------------
// Project 6.3 : Hemodynamics / Blood-Flow CFD   (REDUCED-SCOPE teaching version)
//
// WHAT THIS FILE IS
//   The single source of truth for the *per-cell physics* of a 2-D
//   incompressible Navier-Stokes solver on a rigid, straight channel. Both the
//   CPU reference (reference_cpu.cpp) and the GPU kernels (kernels.cu) include
//   THIS header and call THE SAME inline functions, so the two implementations
//   run byte-for-byte identical arithmetic. That is what lets us verify the GPU
//   result against the CPU result to machine precision (PATTERNS.md §2).
//
//   NSE_HD expands to `__host__ __device__` when compiled by nvcc (so the same
//   function can run on the CPU and inside a kernel) and to nothing when
//   compiled by the plain host compiler (which does not know those keywords).
//   Keep this header free of `__global__`, kernel launches, and CUDA-only types
//   so the host compiler can include it (PATTERNS.md §2).
//
// WHY A REDUCED SCOPE (honesty up front; see ../THEORY.md "Where this sits")
//   The catalog entry (6.3) describes full 3-D, patient-specific, non-Newtonian,
//   fluid-structure-interaction CFD on unstructured meshes with a multigrid
//   pressure solve — a research-grade effort (SimVascular, OpenFOAM, HemeLB).
//   To stay a *readable study project* (CLAUDE.md §13) we solve the SAME
//   governing equations in a smaller, fully-verifiable setting:
//     * 2-D structured (uniform) grid instead of a 3-D unstructured mesh,
//     * a rigid straight channel instead of compliant patient geometry (no FSI),
//     * Chorin's fractional-step (projection) method for incompressibility,
//     * a Jacobi-iterated pressure Poisson solve instead of algebraic multigrid,
//     * Carreau-Yasuda non-Newtonian viscosity kept as an OPTION (blood shear
//       thinning) — the clinically-relevant physics — but validated against the
//       analytic Newtonian Poiseuille solution so correctness is checkable.
//   We still compute the quantity clinicians care about: WALL SHEAR STRESS (WSS),
//   the near-wall velocity gradient times viscosity — a known atherosclerosis
//   risk factor. THEORY.md explains what the full version adds.
//
// THE MATH (see THEORY.md for the derivation)
//   Incompressible Navier-Stokes, constant density rho, velocity u=(u,v):
//       du/dt + (u . grad) u = -(1/rho) grad p + div( nu grad u ) + g
//       div u = 0                                    (incompressibility)
//   Chorin projection per time step dt:
//     1. u* = u^n + dt ( -advect(u^n) + diffuse(u^n) + g )      (no pressure yet)
//     2. solve  laplacian(p) = (rho/dt) div(u*)                 (Poisson)
//     3. u^{n+1} = u* - (dt/rho) grad(p)                        (projection)
//   Step 2 makes u^{n+1} divergence-free. Steps 1 and 3 are pure stencils; step
//   2 is a Jacobi relaxation (also a stencil). This is why the whole solver maps
//   onto the GPU as a sequence of nearest-neighbour cell updates (PATTERNS.md §1,
//   "stencil + ping-pong", exemplar 6.04).
//
// GRID & LAYOUT
//   A collocated (co-located) uniform grid of nx * ny cells, spacing h in both
//   directions. Fields u, v, p are flat row-major arrays of length nx*ny indexed
//   by idx(x,y) = y*nx + x. x is the streamwise (flow) direction, y is across
//   the channel. Boundary conditions:
//     * top/bottom walls (y=0, y=ny-1): no-slip  u=v=0,
//     * left/right (x=0, x=nx-1): periodic (an infinitely long channel),
//   and a constant body force gx drives the flow (a stand-in for the pressure
//   gradient dp/dx along the vessel). This develops toward the parabolic
//   Poiseuille profile, whose peak and wall-shear we can check analytically.
//
// READ THIS AFTER: reference_cpu.h (the ChannelParams struct + solver API).
// ===========================================================================
#pragma once

#include <cstddef>   // std::size_t
#include <cmath>     // std::pow / std::sqrt on the HOST path (device uses the
                     // built-in pow/sqrt provided by nvcc's device math library)

#ifdef __CUDACC__
#define NSE_HD __host__ __device__
#else
#define NSE_HD
#endif

// ---------------------------------------------------------------------------
// idx: flat row-major index of cell (x,y) in an nx*ny field.
//   Row-major means consecutive x are contiguous in memory, so a GPU warp
//   walking x reads coalesced (adjacent threads -> adjacent addresses).
// ---------------------------------------------------------------------------
NSE_HD inline std::size_t idx(int x, int y, int nx) {
    return static_cast<std::size_t>(y) * nx + x;
}

// ---------------------------------------------------------------------------
// wrap_x: apply PERIODIC wrap-around in the streamwise (x) direction.
//   A neighbour just past the right edge re-enters on the left and vice-versa,
//   modelling an infinitely long straight channel. Branch-light (two ifs) so it
//   is cheap on the GPU where all threads in a warp usually take the same path.
// ---------------------------------------------------------------------------
NSE_HD inline int wrap_x(int x, int nx) {
    if (x < 0)    return x + nx;
    if (x >= nx)  return x - nx;
    return x;
}

// ---------------------------------------------------------------------------
// is_wall: true for the top and bottom rows, which are the rigid no-slip walls.
//   Interior rows (0 < y < ny-1) are fluid; y==0 and y==ny-1 are solid walls
//   where velocity is clamped to zero. Keeping this test in one place means the
//   CPU and GPU agree on exactly which cells are walls.
// ---------------------------------------------------------------------------
NSE_HD inline bool is_wall(int y, int ny) {
    return (y == 0) || (y == ny - 1);
}

// ---------------------------------------------------------------------------
// carreau_yasuda: shear-rate-dependent kinematic viscosity nu(gamma_dot).
//   Blood is a NON-NEWTONIAN, shear-thinning fluid: it is thick (high viscosity)
//   where it moves slowly and thin where it is sheared fast. The Carreau-Yasuda
//   model captures this:
//        nu(g) = nu_inf + (nu_0 - nu_inf) * [ 1 + (lambda*g)^a ]^((n-1)/a)
//   where g = shear rate (1/s), nu_0 = zero-shear viscosity, nu_inf = infinite-
//   shear viscosity, lambda = relaxation time (s), n = power-law index (<1 for
//   shear thinning), a = Yasuda transition parameter.
//   When nu_0 == nu_inf the fluid is NEWTONIAN (constant nu) — that is the mode
//   we verify against the analytic Poiseuille solution. Enabling shear thinning
//   (nu_0 != nu_inf) is the clinically-interesting mode (an exercise / option).
//   Uses only +,*,pow so it is identical on host and device (pow is available in
//   both <cmath> and CUDA device math).
// ---------------------------------------------------------------------------
NSE_HD inline double carreau_yasuda(double gamma_dot,
                                    double nu0, double nu_inf,
                                    double lambda, double n, double a) {
    // Newtonian shortcut: nothing to compute, avoids a pow() and is exact.
    if (nu0 == nu_inf) return nu0;
    const double lg = lambda * gamma_dot;
#ifdef __CUDACC__
    const double bracket = 1.0 + pow(lg, a);
    return nu_inf + (nu0 - nu_inf) * pow(bracket, (n - 1.0) / a);
#else
    const double bracket = 1.0 + std::pow(lg, a);
    return nu_inf + (nu0 - nu_inf) * std::pow(bracket, (n - 1.0) / a);
#endif
}

// ---------------------------------------------------------------------------
// local_shear_rate: magnitude of the velocity gradient at cell (x,y).
//   For our nearly-unidirectional channel flow the dominant term is du/dy
//   (streamwise velocity varying across the channel). We use a central
//   difference in y (one-sided against a wall) and add the small du/dx, dv/*
//   terms for completeness. Units: 1/s (velocity/length). Feeds carreau_yasuda.
// ---------------------------------------------------------------------------
NSE_HD inline double local_shear_rate(int x, int y, int nx, int ny, double h,
                                      const double* u, const double* v) {
    const int xm = wrap_x(x - 1, nx), xp = wrap_x(x + 1, nx);
    const int ym = (y > 0)      ? y - 1 : y;   // clamp at walls (one-sided)
    const int yp = (y < ny - 1) ? y + 1 : y;
    const double inv2h_y = 1.0 / ((yp - ym) * h + 1e-30);  // guard degenerate
    const double du_dy = (u[idx(x, yp, nx)] - u[idx(x, ym, nx)]) * inv2h_y;
    const double dv_dx = (v[idx(xp, y, nx)] - v[idx(xm, y, nx)]) / (2.0 * h);
    const double du_dx = (u[idx(xp, y, nx)] - u[idx(xm, y, nx)]) / (2.0 * h);
    const double dv_dy = (v[idx(x, yp, nx)] - v[idx(x, ym, nx)]) * inv2h_y;
    // Second invariant of the strain-rate tensor -> a scalar shear rate.
    const double g2 = 2.0 * (du_dx * du_dx + dv_dy * dv_dy)
                    + (du_dy + dv_dx) * (du_dy + dv_dx);
#ifdef __CUDACC__
    return sqrt(g2);
#else
    return std::sqrt(g2);
#endif
}

// ---------------------------------------------------------------------------
// STEP 1 -- predictor: compute the provisional velocity u* at cell (x,y).
//   Discretizes  u* = u^n + dt*( -(u.grad)u + nu*laplacian(u) + g )  with:
//     * advection  (u.grad)u  by first-order UPWIND differencing (stable for the
//       low Reynolds numbers of small-vessel blood flow; upwind picks the
//       difference from the direction the flow comes from),
//     * diffusion  nu*laplacian(u)  by the standard 5-point central stencil,
//     * body force g = (gx, 0)  (the driving pressure gradient),
//   with nu evaluated from Carreau-Yasuda at the local shear rate. Walls stay at
//   zero (no-slip). Writes u*,v* into (us,vs). Pure nearest-neighbour reads ->
//   a stencil, so one GPU thread per cell has no data races (it only writes its
//   own cell and reads the read-only u^n,v^n buffers).
// ---------------------------------------------------------------------------
NSE_HD inline void predictor_cell(int x, int y, int nx, int ny,
                                  double h, double dt, double gx,
                                  double nu0, double nu_inf,
                                  double lambda, double n_cy, double a_cy,
                                  const double* u, const double* v,
                                  double* us, double* vs) {
    const std::size_t c = idx(x, y, nx);
    // No-slip walls: velocity pinned to zero, nothing to advance.
    if (is_wall(y, ny)) { us[c] = 0.0; vs[c] = 0.0; return; }

    const int xm = wrap_x(x - 1, nx), xp = wrap_x(x + 1, nx);
    const int ym = y - 1, yp = y + 1;          // interior cell -> neighbours exist

    const double uc = u[c],           vc = v[c];
    const double uw = u[idx(xm,y,nx)], ue = u[idx(xp,y,nx)];
    const double us_ = u[idx(x,ym,nx)], un = u[idx(x,yp,nx)];
    const double vw = v[idx(xm,y,nx)], ve = v[idx(xp,y,nx)];
    const double vs_ = v[idx(x,ym,nx)], vn = v[idx(x,yp,nx)];

    // --- Non-Newtonian viscosity at this cell's shear rate.
    const double gdot = local_shear_rate(x, y, nx, ny, h, u, v);
    const double nu   = carreau_yasuda(gdot, nu0, nu_inf, lambda, n_cy, a_cy);

    // --- Advection by first-order upwind: choose the upwind neighbour per sign.
    //   d(phi)/dx ~ (phi_c - phi_upwind)/h in the direction the flow comes from.
    const double dudx = (uc >= 0.0) ? (uc - uw) / h : (ue - uc) / h;
    const double dudy = (vc >= 0.0) ? (uc - us_) / h : (un - uc) / h;
    const double dvdx = (uc >= 0.0) ? (vc - vw) / h : (ve - vc) / h;
    const double dvdy = (vc >= 0.0) ? (vc - vs_) / h : (vn - vc) / h;
    const double adv_u = uc * dudx + vc * dudy;
    const double adv_v = uc * dvdx + vc * dvdy;

    // --- Diffusion by the 5-point Laplacian: (sum of 4 neighbours - 4*center)/h^2.
    const double inv_h2 = 1.0 / (h * h);
    const double lap_u = (uw + ue + us_ + un - 4.0 * uc) * inv_h2;
    const double lap_v = (vw + ve + vs_ + vn - 4.0 * vc) * inv_h2;

    // --- Explicit Euler predictor (pressure added later in the projection step).
    us[c] = uc + dt * (-adv_u + nu * lap_u + gx);
    vs[c] = vc + dt * (-adv_v + nu * lap_v);
}

// ---------------------------------------------------------------------------
// divergence_cell: div(u*) at cell (x,y) by central differences.
//   The pressure Poisson RHS is (rho/dt)*div(u*); this returns div(u*) so the
//   caller scales it. Walls have zero velocity so div there is well-defined.
// ---------------------------------------------------------------------------
NSE_HD inline double divergence_cell(int x, int y, int nx, int ny, double h,
                                     const double* us, const double* vs) {
    const int xm = wrap_x(x - 1, nx), xp = wrap_x(x + 1, nx);
    const int ym = (y > 0)      ? y - 1 : y;
    const int yp = (y < ny - 1) ? y + 1 : y;
    const double dudx = (us[idx(xp,y,nx)] - us[idx(xm,y,nx)]) / (2.0 * h);
    const double dvdy = (vs[idx(x,yp,nx)] - vs[idx(x,ym,nx)]) / ((yp - ym) * h + 1e-30);
    return dudx + dvdy;
}

// ---------------------------------------------------------------------------
// STEP 2 -- one Jacobi sweep of the pressure Poisson equation at cell (x,y).
//   Solving laplacian(p) = rhs with the 5-point stencil, the update that drives
//   the residual to zero is:
//        p_new = ( p_W + p_E + p_S + p_N - h^2 * rhs ) / 4
//   Jacobi reads the OLD p everywhere and writes a NEW p (double buffer / ping-
//   pong) so every cell is independent within a sweep -> perfect for the GPU
//   (PATTERNS.md §1, "Jacobi projection + double buffer", exemplar 10.02).
//   Boundary pressure uses homogeneous Neumann (dp/dn = 0): a wall/edge neighbour
//   is replaced by this cell's own value, which is the natural BC for the
//   projection method. Returns the new pressure for cell (x,y).
// ---------------------------------------------------------------------------
NSE_HD inline double pressure_jacobi_cell(int x, int y, int nx, int ny, double h,
                                          const double* p, double rhs) {
    const int xm = wrap_x(x - 1, nx), xp = wrap_x(x + 1, nx);
    // Neumann at the walls: mirror this cell's pressure across the wall.
    const double pW = p[idx(xm, y, nx)];
    const double pE = p[idx(xp, y, nx)];
    const double pS = (y > 0)      ? p[idx(x, y - 1, nx)] : p[idx(x, y, nx)];
    const double pN = (y < ny - 1) ? p[idx(x, y + 1, nx)] : p[idx(x, y, nx)];
    return 0.25 * (pW + pE + pS + pN - h * h * rhs);
}

// ---------------------------------------------------------------------------
// STEP 3 -- corrector/projection: subtract the pressure gradient at cell (x,y).
//        u^{n+1} = u* - (dt/rho) * grad(p)
//   This is the step that removes the divergence introduced by the predictor,
//   enforcing incompressibility. Walls stay at zero (no-slip). grad(p) by
//   central differences (periodic in x, Neumann-mirrored at walls).
// ---------------------------------------------------------------------------
NSE_HD inline void corrector_cell(int x, int y, int nx, int ny,
                                  double h, double dt, double rho,
                                  const double* us, const double* vs,
                                  const double* p,
                                  double* u_new, double* v_new) {
    const std::size_t c = idx(x, y, nx);
    if (is_wall(y, ny)) { u_new[c] = 0.0; v_new[c] = 0.0; return; }
    const int xm = wrap_x(x - 1, nx), xp = wrap_x(x + 1, nx);
    const int ym = y - 1, yp = y + 1;
    const double dpdx = (p[idx(xp,y,nx)] - p[idx(xm,y,nx)]) / (2.0 * h);
    const double dpdy = (p[idx(x,yp,nx)] - p[idx(x,ym,nx)]) / (2.0 * h);
    const double coef = dt / rho;
    u_new[c] = us[c] - coef * dpdx;
    v_new[c] = vs[c] - coef * dpdy;
}

// ---------------------------------------------------------------------------
// wall_shear_stress: WSS at the bottom wall for streamwise column x.
//   WSS = mu * (du/dy)|_wall, with mu = rho*nu the DYNAMIC viscosity. We take a
//   one-sided difference of u between the first fluid row (y=1) and the wall
//   (y=0, where u=0): du/dy ~ u[x,1]/h. This is THE clinically-relevant output:
//   low/oscillatory WSS marks atheroprone regions of a vessel. Returned in Pa
//   when inputs are SI (kept dimensionless in the demo and labelled as such).
// ---------------------------------------------------------------------------
NSE_HD inline double wall_shear_stress(int x, int nx, double h, double rho,
                                       double nu, const double* u) {
    const double du_dy = (u[idx(x, 1, nx)] - u[idx(x, 0, nx)]) / h;  // u[x,0]=0
    return rho * nu * du_dy;   // mu * shear rate at the wall
}
