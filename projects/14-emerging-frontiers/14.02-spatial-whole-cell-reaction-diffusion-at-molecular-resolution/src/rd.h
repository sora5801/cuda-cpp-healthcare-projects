// ===========================================================================
// src/rd.h  --  Shared (host + device) Gray-Scott reaction-diffusion update
// ---------------------------------------------------------------------------
// Project 14.02 : Spatial / Whole-Cell Reaction-Diffusion (reduced-scope teaching version)
//
// WHAT THIS PROJECT COMPUTES
//   A 2-D reaction-diffusion PDE -- the Gray-Scott model -- on a periodic grid.
//   Two virtual chemicals U and V diffuse and react; from a tiny seed they
//   self-organize into Turing PATTERNS (spots, stripes, mazes). This is the
//   continuum, grid (stencil) form of spatial reaction-diffusion; the full
//   project (catalog 14.2) is PARTICLE-based molecular-resolution RD -- a 🔴
//   frontier problem we describe in THEORY "real world".
//
//   Gray-Scott (Du, Dv diffusion; F feed; k kill):
//     dU/dt = Du*lap(U) - U*V^2 + F*(1 - U)
//     dV/dt = Dv*lap(V) + U*V^2 - (F + k)*V
//   integrated with explicit Euler; lap() is the 5-point Laplacian.
//
// WHY A GPU
//   Each grid cell updates from its 4 neighbours only -- a pure STENCIL (cf.
//   the lattice-Boltzmann project 6.04). One thread per cell; double-buffered
//   so all cells advance in parallel with no races.
//
//   The per-cell update lives here as a __host__ __device__ function so the CPU
//   reference and the GPU kernel are byte-for-byte identical. RD_HD = __host__
//   __device__ under nvcc, nothing under the host compiler.
// ===========================================================================
#pragma once

#ifdef __CUDACC__
#define RD_HD __host__ __device__
#else
#define RD_HD
#endif

struct RdParams {
    int nx, ny;       // grid size
    double Du, Dv;    // diffusion coefficients for U, V
    double F, k;      // feed and kill rates (these select the pattern)
    double dt;        // explicit-Euler timestep
    int steps;        // number of timesteps
    int seed_half;    // half-size of the central V-seed square
};

// 5-point Laplacian of field f at (x,y), with PERIODIC (toroidal) boundaries.
RD_HD inline double rd_laplacian(const double* f, int x, int y, int nx, int ny) {
    const int xm = (x - 1 + nx) % nx, xp = (x + 1) % nx;
    const int ym = (y - 1 + ny) % ny, yp = (y + 1) % ny;
    return f[y * nx + xm] + f[y * nx + xp] + f[ym * nx + x] + f[yp * nx + x]
         - 4.0 * f[y * nx + x];
}

// One explicit-Euler reaction-diffusion update for cell (x,y): read U,V (and
// neighbours) from the input buffers, write the next U,V to the output buffers.
RD_HD inline void rd_update(int x, int y, const RdParams& P,
                           const double* U, const double* V,
                           double* Un, double* Vn) {
    const int i = y * P.nx + x;
    const double u = U[i], v = V[i];
    const double lu = rd_laplacian(U, x, y, P.nx, P.ny);
    const double lv = rd_laplacian(V, x, y, P.nx, P.ny);
    const double uvv = u * v * v;               // the nonlinear reaction term
    Un[i] = u + P.dt * (P.Du * lu - uvv + P.F * (1.0 - u));
    Vn[i] = v + P.dt * (P.Dv * lv + uvv - (P.F + P.k) * v);
}
