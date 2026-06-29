// ===========================================================================
// src/lbm_d2q9.h  --  Shared (host + device) D2Q9 lattice-Boltzmann update
// ---------------------------------------------------------------------------
// Project 6.04 : Lattice-Boltzmann Blood/Airflow Solver
//
// THE MODEL
//   LBM replaces the Navier-Stokes PDEs with a mesoscale kinetic model: at each
//   lattice node we track 9 "populations" f_i (the D2Q9 stencil) -- the density
//   of fictitious particles moving in 9 discrete directions. Each timestep is:
//     COLLIDE : relax the populations toward a local equilibrium (BGK).
//     STREAM  : move each population to its neighbour in direction i.
//   Macroscopic density and velocity are moments of f. The beauty for GPUs: a
//   node updates using ONLY its nearest neighbours -- a pure stencil, no global
//   communication.
//
//   The whole per-node update lives here as ONE __host__ __device__ function so
//   the CPU reference and the GPU kernel run byte-for-byte identical math (the
//   key to exact verification). LBM_HD expands to __host__ __device__ under nvcc
//   and to nothing under the host compiler.
//
//   D2Q9 directions (index : (e_x,e_y)):
//     0:( 0, 0)  1:( 1, 0)  2:( 0, 1)  3:(-1, 0)  4:( 0,-1)
//     5:( 1, 1)  6:(-1, 1)  7:(-1,-1)  8:( 1,-1)
// ===========================================================================
#pragma once

#include <cstddef>

#ifdef __CUDACC__
#define LBM_HD __host__ __device__
#else
#define LBM_HD
#endif

// --- D2Q9 lattice constants, via tiny accessor functions (host+device safe) ---
LBM_HD inline int e_x(int i) { const int v[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1}; return v[i]; }
LBM_HD inline int e_y(int i) { const int v[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1}; return v[i]; }
LBM_HD inline int opp(int i) { const int v[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6}; return v[i]; } // opposite dir
LBM_HD inline double w_i(int i) {
    const double v[9] = {4.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/36, 1.0/36, 1.0/36, 1.0/36};
    return v[i];
}

// Flat index of population i at node (x,y): structure-of-arrays per direction so
// that, for a fixed (i,y), consecutive x are contiguous -> coalesced GPU reads.
LBM_HD inline std::size_t lbm_idx(int i, int x, int y, int nx, int ny) {
    return (static_cast<std::size_t>(i) * ny + y) * nx + x;
}

// One full collide+stream update for node (x,y), reading f_old and writing f_new.
//   Geometry: periodic in x; solid walls above/below (rows beyond [0,ny-1]) with
//   halfway BOUNCE-BACK for no-slip. A constant body force gx (in +x) drives the
//   flow via an equilibrium velocity shift (a simple, stable forcing).
// This is the entire physics -- the CPU loops it over all nodes, the GPU runs
// one thread per node; both produce identical results.
LBM_HD inline void lbm_collide_stream(int x, int y, int nx, int ny,
                                     double tau, double gx,
                                     const double* f_old, double* f_new) {
    // --- STREAM (pull): gather the population that arrives at (x,y) in each dir.
    double f[9];
    for (int i = 0; i < 9; ++i) {
        int xs = x - e_x(i);                 // upstream source node
        int ys = y - e_y(i);
        if (xs < 0) xs += nx; else if (xs >= nx) xs -= nx;   // periodic in x
        if (ys < 0 || ys >= ny) {
            // Source is a wall: halfway bounce-back -> take the OPPOSITE
            // population at THIS node (gives no-slip at the wall halfway out).
            f[i] = f_old[lbm_idx(opp(i), x, y, nx, ny)];
        } else {
            f[i] = f_old[lbm_idx(i, xs, ys, nx, ny)];
        }
    }

    // --- Macroscopic moments: density rho and velocity (ux, uy).
    double rho = 0.0, ux = 0.0, uy = 0.0;
    for (int i = 0; i < 9; ++i) { rho += f[i]; ux += e_x(i) * f[i]; uy += e_y(i) * f[i]; }
    ux /= rho; uy /= rho;

    // --- Body force by equilibrium velocity shift (drives Poiseuille flow in +x).
    const double uxeq = ux + tau * gx;
    const double uyeq = uy;
    const double usqr = uxeq * uxeq + uyeq * uyeq;

    // --- COLLIDE (BGK): relax toward the local Maxwell-Boltzmann equilibrium.
    for (int i = 0; i < 9; ++i) {
        const double ciu = e_x(i) * uxeq + e_y(i) * uyeq;     // c_i . u
        const double feq = w_i(i) * rho *
            (1.0 + 3.0 * ciu + 4.5 * ciu * ciu - 1.5 * usqr); // truncated Maxwellian
        f_new[lbm_idx(i, x, y, nx, ny)] = f[i] - (f[i] - feq) / tau;  // BGK relaxation
    }
}

// Macroscopic x-velocity at node (x,y) from a population field (for reporting /
// comparison). Shared so CPU and GPU velocities are computed identically.
LBM_HD inline double lbm_ux(int x, int y, int nx, int ny, const double* f) {
    double rho = 0.0, ux = 0.0;
    for (int i = 0; i < 9; ++i) {
        const double fi = f[lbm_idx(i, x, y, nx, ny)];
        rho += fi; ux += e_x(i) * fi;
    }
    return ux / rho;
}
