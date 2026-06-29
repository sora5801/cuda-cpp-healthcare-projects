// ===========================================================================
// src/pbd.h  --  Shared (host + device) Position-Based Dynamics on a grid mesh
// ---------------------------------------------------------------------------
// Project 10.02 : Real-Time Soft-Tissue Deformation for Surgical Simulation
//
// WHAT THIS PROJECT COMPUTES
//   A deformable sheet (a stand-in for soft tissue) modelled as a grid of mass
//   PARTICLES linked by distance CONSTRAINTS (structural + shear springs).
//   Position-Based Dynamics (PBD) advances it each frame by: predict positions
//   under gravity, then iteratively PROJECT positions to satisfy the distance
//   constraints, then derive velocities from the position change. The top edge
//   is pinned, so the sheet drapes under gravity.
//
// WHY A GPU
//   Surgical simulators need sub-10 ms updates on meshes of 10^5+ elements for
//   haptic feedback. PBD's constraint projections are data-parallel: with a
//   JACOBI scheme, every particle computes its correction from the (read-only)
//   current positions of its neighbours, independently -> one thread per
//   particle. (Production solvers use XPBD and graph colouring; THEORY.md.)
//
//   The per-particle projection + the predict/finalize steps live here as
//   __host__ __device__ functions so the CPU reference and the GPU kernels do
//   byte-for-byte identical math -> exact verification. PBD_HD = __host__
//   __device__ under nvcc, nothing under the host compiler.
// ===========================================================================
#pragma once

#include <cmath>

#ifdef __CUDACC__
#define PBD_HD __host__ __device__
#else
#define PBD_HD
#endif

// --- Minimal 3-vector with just the operations PBD needs (host+device) ---
struct Vec3 { double x, y, z; };
PBD_HD inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
PBD_HD inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
PBD_HD inline Vec3 operator*(Vec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
PBD_HD inline double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
PBD_HD inline double length(Vec3 a) { return sqrt(dot(a, a)); }

// Simulation parameters (read from the data file).
struct PbdParams {
    int R, C;            // grid rows x columns of particles
    double spacing;      // initial rest spacing between adjacent particles
    double dt;           // timestep
    double gravity;      // gravitational acceleration (applied in -y)
    double stiffness;    // constraint stiffness in [0,1]
    double omega;        // Jacobi relaxation factor (~1.0)
    int iters;           // constraint-projection iterations per step
    int steps;           // number of timesteps
};

// The 8 spring neighbours: 4 structural (axis) + 4 shear (diagonal).
// (Declared as a function so it is usable in both host and device code.)
PBD_HD inline void neighbour_offset(int k, int& dr, int& dc) {
    const int off[8][2] = {{-1,0},{1,0},{0,-1},{0,1},{-1,-1},{-1,1},{1,-1},{1,1}};
    dr = off[k][0]; dc = off[k][1];
}

// Predict a particle's position under gravity (the explicit part of a PBD step).
// Pinned particles (inverse mass w <= 0) do not move.
PBD_HD inline Vec3 pbd_predict(Vec3 x, Vec3 v, double w, double dt, double gravity) {
    if (w <= 0.0) return x;
    const Vec3 g = {0.0, -gravity, 0.0};
    return x + v * dt + g * (dt * dt);     // x + v*dt + a*dt^2
}

// Jacobi constraint projection for ONE particle (r,c): sum the position
// corrections from each incident distance constraint, computed from the
// READ-ONLY positions `p`, and return the averaged correction to apply.
//   For a constraint between i and j with rest length L:
//     C   = |p_i - p_j| - L              (violation)
//     dx_i = -(w_i/(w_i+w_j)) * C * (p_i-p_j)/|p_i-p_j| * stiffness
//   We average over the incident constraints (Jacobi) with relaxation omega.
PBD_HD inline Vec3 pbd_correction(int r, int c, const PbdParams& P,
                                 const Vec3* p, const double* w) {
    const int i = r * P.C + c;
    const Vec3 pi = p[i];
    const double wi = w[i];
    Vec3 acc = {0.0, 0.0, 0.0};
    int count = 0;
    const double diag_rest = P.spacing * 1.4142135623730951;   // sqrt(2)*spacing

    for (int k = 0; k < 8; ++k) {
        int dr, dc; neighbour_offset(k, dr, dc);
        const int nr = r + dr, nc = c + dc;
        if (nr < 0 || nr >= P.R || nc < 0 || nc >= P.C) continue;   // edge of mesh
        const int j = nr * P.C + nc;
        const double wj = w[j];
        const double denom = wi + wj;
        if (denom <= 0.0) continue;                  // both pinned: nothing to do
        const double rest = (k < 4) ? P.spacing : diag_rest;
        const Vec3 d = pi - p[j];
        const double len = length(d);
        if (len < 1e-12) continue;                   // coincident: skip (no direction)
        const double Cviol = len - rest;
        const Vec3 grad = d * (1.0 / len);           // unit constraint gradient at i
        const double s = -(wi / denom) * Cviol * P.stiffness;
        acc = acc + grad * s;
        ++count;
    }
    if (count > 0) acc = acc * (P.omega / count);    // Jacobi average + relaxation
    return acc;
}

// Derive the new velocity from the position change (the PBD velocity update).
// Pinned particles stay at rest. Caller then commits x = p.
PBD_HD inline Vec3 pbd_new_velocity(Vec3 p, Vec3 x, double w, double dt) {
    if (w <= 0.0) return {0.0, 0.0, 0.0};
    return (p - x) * (1.0 / dt);
}
