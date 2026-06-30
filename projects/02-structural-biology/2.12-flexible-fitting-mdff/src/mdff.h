// ===========================================================================
// src/mdff.h  --  Shared (host + device) Molecular Dynamics Flexible Fitting
// ---------------------------------------------------------------------------
// Project 2.12 : Flexible Fitting / MDFF   (reduced-scope teaching version)
//
// WHAT THIS PROJECT COMPUTES
//   MDFF (Molecular Dynamics Flexible Fitting) deforms an atomic model so it
//   matches an experimental cryo-EM DENSITY MAP. The map is a 3-D grid of
//   "electron density" values; a fitted model is one whose atoms sit on the
//   ridges of that density. MDFF adds, to an ordinary MD force field, an extra
//   "density-derived" force that pulls each atom UPHILL along the local density
//   gradient -- i.e. toward denser regions -- while the MD force field keeps
//   bonds/angles/sterics sane so the structure does not tear apart.
//
//   We teach the heart of that idea with the smallest faithful model:
//
//     force on atom a  =  w * grad(rho)(x_a)            <- the MDFF (fitting) force
//                       - k * (x_a - x_ref_a)           <- a harmonic restraint
//                                                          (our stand-in for the
//                                                           MD force field that
//                                                           prevents runaway)
//
//   and we move atoms by overdamped STEEPEST DESCENT (one tiny step along the
//   total force each iteration). Production MDFF instead runs full Langevin MD
//   in NAMD/OpenMM; we replace that with a restraint + steepest descent so the
//   whole thing is a few lines a learner can follow (see ../THEORY.md "Where
//   this sits in the real world").
//
// THE TWO GPU-RELEVANT PIECES
//   (1) Sampling rho and grad(rho) at an arbitrary atom position requires
//       TRILINEAR INTERPOLATION from the 8 surrounding grid voxels -- the exact
//       "gather + interpolate" pattern used in CT backprojection (project 4.01).
//   (2) Every atom's force is INDEPENDENT given the (read-only) density map and
//       the current positions, so we give each atom its own GPU thread, double
//       buffer the positions, and iterate (the Jacobi/ensemble pattern of 9.02
//       and 10.02). No atomics, no races.
//
//   Both the per-atom physics AND the trilinear sampler live here as
//   __host__ __device__ inline functions, so the CPU reference (reference_cpu)
//   and the GPU kernel (kernels.cu) run BYTE-FOR-BYTE IDENTICAL math. That makes
//   verification a tight numeric comparison instead of a hand-wave.
//
//   MDFF_HD expands to "__host__ __device__" under nvcc and to nothing under the
//   plain host compiler, so this one header compiles in both worlds. Keep CUDA-
//   only constructs (no __global__, no <<<>>>) OUT of this file for that reason.
//
// READ THIS AFTER: ../THEORY.md (the "why"); BEFORE kernels.cu / reference_cpu.
// ===========================================================================
#pragma once

#include <cmath>     // sqrt, fabs (resolved to the device intrinsics under nvcc)

// One macro to make every function below dual-targeted. Under nvcc __CUDACC__ is
// defined and we tag the functions for both host and device; under cl.exe/g++
// the decorators do not exist, so we expand to nothing.
#ifdef __CUDACC__
#define MDFF_HD __host__ __device__
#else
#define MDFF_HD
#endif

// ---------------------------------------------------------------------------
// Vec3 : a minimal double-precision 3-vector with only the ops MDFF needs.
//   We use double (not float) because the fit is a long iterative descent and we
//   want the CPU and GPU to agree to many digits; double also keeps the trilinear
//   interpolation and gradient stable. (THEORY "Numerical considerations".)
// ---------------------------------------------------------------------------
struct Vec3 { double x, y, z; };

MDFF_HD inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
MDFF_HD inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
MDFF_HD inline Vec3 operator*(Vec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
MDFF_HD inline double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
MDFF_HD inline double length(Vec3 a) { return sqrt(dot(a, a)); }

// ---------------------------------------------------------------------------
// MdffParams : everything that defines the problem instance.
//   The density map is a regular grid: NX x NY x NZ voxels, isotropic spacing
//   `vox` (Angstrom/voxel), with the grid origin at the world coordinate origin
//   (0,0,0). A world point p maps to grid coordinates g = p / vox.
// ---------------------------------------------------------------------------
struct MdffParams {
    int    nx, ny, nz;   // density-map grid dimensions (voxels along x, y, z)
    double vox;          // voxel size (world units per voxel); grid is isotropic
    int    natoms;       // number of atoms in the model being fitted
    double w_dens;       // weight on the density (fitting) force  [force / density-slope]
    double k_rest;       // harmonic-restraint stiffness (MD-force-field stand-in)
    double step;         // steepest-descent step size (overdamped "dt/gamma")
    int    iters;        // number of fitting iterations
};

// ---------------------------------------------------------------------------
// grid_index : flatten 3-D voxel coordinates (ix,iy,iz) into the 1-D array.
//   Layout is x-fastest, then y, then z (row-major in (z,y,x)):
//       idx = (iz * ny + iy) * nx + ix
//   We keep this in ONE place so every sampler agrees on the memory order.
// ---------------------------------------------------------------------------
MDFF_HD inline int grid_index(int ix, int iy, int iz, int nx, int ny) {
    return (iz * ny + iy) * nx + ix;
}

// ---------------------------------------------------------------------------
// clampi : clamp an integer index into [0, hi]. Used so that interpolation near
//   the map boundary reads edge voxels instead of running off the array (a
//   "clamp-to-edge" boundary condition, the same trick texture units use).
// ---------------------------------------------------------------------------
MDFF_HD inline int clampi(int v, int hi) {
    if (v < 0) return 0;
    if (v > hi) return hi;
    return v;
}

// ---------------------------------------------------------------------------
// sample_density : TRILINEAR INTERPOLATION of the scalar density at world point
//   p, returned via the function value. This is the read side of the "gather"
//   pattern: from a continuous position we gather the 8 surrounding voxels and
//   blend them by the fractional offsets.
//
//   Steps:
//     1. Convert world -> grid coordinates: g = p / vox.
//     2. Let (ix,iy,iz) = floor(g); the fractional parts (fx,fy,fz) in [0,1) are
//        the interpolation weights toward the +1 neighbour.
//     3. Read the 8 corner voxels c000..c111 (clamped to the edge) and blend:
//        a standard separable lerp along x, then y, then z.
//
//   Why trilinear (not nearest)? A nearest-voxel read gives a piecewise-constant
//   field whose gradient is zero almost everywhere -> no usable fitting force.
//   Trilinear gives a continuous field with a well-defined gradient.
//
//   `rho` is the density array (length nx*ny*nz). Returned value is unitless
//   "density"; the matching gradient is computed by sample_gradient below.
// ---------------------------------------------------------------------------
MDFF_HD inline double sample_density(const double* rho, Vec3 p, const MdffParams& P) {
    // 1. World -> grid coordinates (voxel units).
    const double gx = p.x / P.vox;
    const double gy = p.y / P.vox;
    const double gz = p.z / P.vox;

    // 2. Lower corner voxel + fractional offsets within the cell.
    //    floor() picks the voxel whose corner is at-or-below g; the fraction is
    //    how far we are toward the next voxel along each axis.
    const int ix = (int)floor(gx), iy = (int)floor(gy), iz = (int)floor(gz);
    const double fx = gx - ix, fy = gy - iy, fz = gz - iz;

    // Clamp the 2 sample planes per axis so boundary atoms read edge voxels.
    const int ix0 = clampi(ix, P.nx - 1), ix1 = clampi(ix + 1, P.nx - 1);
    const int iy0 = clampi(iy, P.ny - 1), iy1 = clampi(iy + 1, P.ny - 1);
    const int iz0 = clampi(iz, P.nz - 1), iz1 = clampi(iz + 1, P.nz - 1);

    // 3. The 8 corner densities of the enclosing voxel cube.
    const double c000 = rho[grid_index(ix0, iy0, iz0, P.nx, P.ny)];
    const double c100 = rho[grid_index(ix1, iy0, iz0, P.nx, P.ny)];
    const double c010 = rho[grid_index(ix0, iy1, iz0, P.nx, P.ny)];
    const double c110 = rho[grid_index(ix1, iy1, iz0, P.nx, P.ny)];
    const double c001 = rho[grid_index(ix0, iy0, iz1, P.nx, P.ny)];
    const double c101 = rho[grid_index(ix1, iy0, iz1, P.nx, P.ny)];
    const double c011 = rho[grid_index(ix0, iy1, iz1, P.nx, P.ny)];
    const double c111 = rho[grid_index(ix1, iy1, iz1, P.nx, P.ny)];

    // Separable lerp: blend along x, then y, then z.
    const double c00 = c000 * (1 - fx) + c100 * fx;
    const double c10 = c010 * (1 - fx) + c110 * fx;
    const double c01 = c001 * (1 - fx) + c101 * fx;
    const double c11 = c011 * (1 - fx) + c111 * fx;
    const double c0  = c00 * (1 - fy) + c10 * fy;
    const double c1  = c01 * (1 - fy) + c11 * fy;
    return c0 * (1 - fz) + c1 * fz;
}

// ---------------------------------------------------------------------------
// sample_gradient : the SPATIAL GRADIENT grad(rho) of the (trilinearly
//   interpolated) density at world point p. This vector is the direction of
//   steepest density INCREASE -- exactly the direction MDFF wants to push atoms.
//
//   We use a symmetric finite difference along each axis with a half-voxel
//   probe `h`:
//       d rho / d x  ~=  [rho(x + h) - rho(x - h)] / (2h)
//   evaluated through sample_density() so it stays consistent with the field the
//   atom actually feels. (An analytic trilinear gradient is possible and faster;
//   the finite difference is used here because it is the most transparent to a
//   learner and shares one code path. THEORY discusses the analytic form.)
// ---------------------------------------------------------------------------
MDFF_HD inline Vec3 sample_gradient(const double* rho, Vec3 p, const MdffParams& P) {
    const double h = 0.5 * P.vox;          // half-voxel probe distance (world units)
    const double inv = 1.0 / (2.0 * h);    // 1/(2h) finite-difference denominator

    const Vec3 px0 = {p.x - h, p.y, p.z}, px1 = {p.x + h, p.y, p.z};
    const Vec3 py0 = {p.x, p.y - h, p.z}, py1 = {p.x, p.y + h, p.z};
    const Vec3 pz0 = {p.x, p.y, p.z - h}, pz1 = {p.x, p.y, p.z + h};

    Vec3 g;
    g.x = (sample_density(rho, px1, P) - sample_density(rho, px0, P)) * inv;
    g.y = (sample_density(rho, py1, P) - sample_density(rho, py0, P)) * inv;
    g.z = (sample_density(rho, pz1, P) - sample_density(rho, pz0, P)) * inv;
    return g;
}

// ---------------------------------------------------------------------------
// mdff_step_atom : advance ONE atom by one overdamped steepest-descent step.
//   This is the per-element physics shared by the CPU loop and the GPU kernel.
//
//   total force F = w_dens * grad(rho)(x)        (uphill on the density -> fit)
//                 - k_rest * (x - x_ref)         (pull back toward the reference
//                                                 model -> our MD-field stand-in;
//                                                 it stops atoms from sliding off
//                                                 to the global density maximum
//                                                 and tearing the structure)
//   overdamped update:  x_new = x + step * F
//
//   Parameters:
//     rho   : density map (read-only)
//     x     : this atom's current position
//     x_ref : this atom's reference (restraint anchor) position
//     P     : problem parameters (weights, step, grid)
//   Returns the atom's new position. Pure function -> identical on host/device.
// ---------------------------------------------------------------------------
MDFF_HD inline Vec3 mdff_step_atom(const double* rho, Vec3 x, Vec3 x_ref,
                                   const MdffParams& P) {
    const Vec3 g = sample_gradient(rho, x, P);          // density gradient at x
    const Vec3 f_dens = g * P.w_dens;                   // fitting force (uphill)
    const Vec3 f_rest = (x - x_ref) * (-P.k_rest);      // harmonic restraint
    const Vec3 F = f_dens + f_rest;                     // total force
    return x + F * P.step;                              // overdamped SD step
}
