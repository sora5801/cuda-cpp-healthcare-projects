// ===========================================================================
// src/docking_core.h  --  The ONE TRUE per-pose scoring physics (CPU == GPU)
// ---------------------------------------------------------------------------
// Project 1.3 : Molecular Docking Engine  (reduced-scope teaching version)
//
// WHY THIS HEADER EXISTS  (the most important idiom in this repo, PATTERNS.md S2)
//   Molecular docking scores a candidate ligand POSE by summing a precomputed
//   energy grid over the ligand's atoms. That per-pose arithmetic must be
//   *byte-for-byte identical* on the CPU reference and the GPU kernel, otherwise
//   "GPU agrees with CPU" verification would only be approximate. So we put the
//   physics here, ONCE, as `__host__ __device__` inline functions:
//
//       * reference_cpu.cpp  (host C++ compiler)  loops these over every pose;
//       * kernels.cu         (nvcc)               calls these from one thread
//                                                  per pose.
//
//   Both sides therefore execute the SAME source -> the same FMA-free `double`
//   arithmetic -> identical results (verified to ~1e-12, see ../THEORY.md S"verify").
//
// THE HD MACRO
//   When compiled by nvcc (__CUDACC__ defined) we decorate every shared function
//   with `__host__ __device__` so it can run on BOTH processors. When compiled by
//   the plain host compiler, those CUDA keywords do not exist, so DOCK_HD expands
//   to nothing. Keep CUDA-only constructs (__global__, <<<>>>, threadIdx) OUT of
//   this header so the host compiler can include it cleanly.
//
// WHAT THE PHYSICS IS (full derivation in ../THEORY.md)
//   The receptor's binding pocket is precomputed as a regular 3D ENERGY GRID:
//   grid[z][y][x] holds the interaction energy (kcal/mol) that a single probe
//   atom would feel if placed at that grid point. (In real AutoDock there is one
//   such grid per atom type plus an electrostatics grid; we use ONE grid for a
//   generic probe to keep the teaching version legible -- THEORY S"real world".)
//   To score a pose we:
//     1. rigidly transform each ligand atom by the pose (rotate then translate),
//     2. read the grid energy at the atom's position via TRILINEAR INTERPOLATION
//        (the grid is discrete; atoms land between grid points), and
//     3. sum those per-atom energies. Lower (more negative) total = better fit.
//   This is the "grid-based energy evaluation" the catalog names; sampling many
//   poses and scoring each is the massively-parallel "scoring function" step.
//
// READ THIS AFTER: reference_cpu.h (the data model).  Then kernels.cu / reference_cpu.cpp.
// ===========================================================================
#pragma once

#include <cstdint>

// --- the host/device portability macro (PATTERNS.md S2) --------------------
#ifdef __CUDACC__
#define DOCK_HD __host__ __device__   // nvcc: this function runs on CPU and GPU
#else
#define DOCK_HD                       // host compiler: the decorators don't exist
#endif

// ---------------------------------------------------------------------------
// GridDims: the geometry of the precomputed receptor energy grid.
//   A regular axis-aligned lattice of nx*ny*nz points. origin is the world-space
//   coordinate (Angstrom) of grid point (0,0,0); spacing is the distance between
//   adjacent points along each axis (AutoDock's default is 0.375 A). A point
//   with integer index (ix,iy,iz) sits at world position
//       origin + (ix,iy,iz) * spacing.
//   Storage is row-major with x fastest: value at (ix,iy,iz) is
//       data[(iz*ny + iy)*nx + ix]   (see grid_at below).
//   This struct is a plain POD so it can be copied to the GPU by value/memcpy.
// ---------------------------------------------------------------------------
struct GridDims {
    int    nx, ny, nz;        // number of grid points along x, y, z
    double ox, oy, oz;        // world-space origin (Angstrom) of point (0,0,0)
    double spacing;           // grid step (Angstrom) -- isotropic for simplicity

    // Total number of scalar energy values in the grid.
    DOCK_HD long long count() const {
        return static_cast<long long>(nx) * ny * nz;
    }
};

// ---------------------------------------------------------------------------
// grid_index: flatten an integer grid coordinate (ix,iy,iz) to a 1-D offset.
//   Row-major with x fastest -> consecutive ix are adjacent in memory, which is
//   what we want because trilinear interpolation reads the 8 corners of a cell
//   and the four x-neighbours then become coalesced/cache-friendly reads.
//   No bounds checking here: callers (grid_at) clamp first.
// ---------------------------------------------------------------------------
DOCK_HD inline long long grid_index(const GridDims& g, int ix, int iy, int iz) {
    return (static_cast<long long>(iz) * g.ny + iy) * g.nx + ix;
}

// ---------------------------------------------------------------------------
// clampi: clamp an integer to [lo, hi]. Used so a ligand atom that drifts to (or
//   just past) the grid boundary reads the edge value instead of out-of-bounds
//   memory. Returning the edge energy is the standard "wall" behaviour: poses
//   that push atoms outside the mapped pocket get a poor (high) score and are
//   naturally rejected. (A branchy min/max is fine; this is not the hot path.)
// ---------------------------------------------------------------------------
DOCK_HD inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// ---------------------------------------------------------------------------
// trilinear_energy: sample the energy grid at an arbitrary world-space point
//   (wx,wy,wz) using TRILINEAR INTERPOLATION of the 8 surrounding grid corners.
//
//   Why interpolate: the grid is discrete but ligand atoms land at continuous
//   positions. Nearest-point lookup would make the score a step function of pose
//   (bad for any optimiser and physically wrong); trilinear interpolation makes
//   the energy continuous and once-differentiable-enough for local search. This
//   is exactly what AutoDock does for its force-field grids.
//
//   Steps:
//     (a) convert world coords -> fractional grid coords gx = (wx-ox)/spacing;
//     (b) split into the lower integer corner (ix,iy,iz) and the fractional
//         offset (fx,fy,fz) in [0,1) within that grid cell;
//     (c) clamp corners into range (boundary "wall", see clampi);
//     (d) blend the 8 corner energies with weights (1-f) and f along each axis.
//
//   This is the GPU's "gather with interpolation" pattern (PATTERNS.md table):
//   each thread reads 8 scattered grid values and combines them. On real GPUs
//   this grid would live in TEXTURE memory, whose hardware does trilinear
//   interpolation for free -- we do it by hand here so the math is visible and
//   identical on the CPU (THEORY S"real world").
//
//   All arithmetic is `double` and FMA-free-by-construction (the compiler may
//   still fuse, but identically on both paths for this short expression), so the
//   CPU and GPU results match to ~1e-12.
// ---------------------------------------------------------------------------
DOCK_HD inline double trilinear_energy(const double* grid, const GridDims& g,
                                       double wx, double wy, double wz) {
    // (a) world -> fractional grid coordinate
    const double gx = (wx - g.ox) / g.spacing;
    const double gy = (wy - g.oy) / g.spacing;
    const double gz = (wz - g.oz) / g.spacing;

    // (b) lower corner index (floor) and in-cell fraction
    // floor via cast is safe here because we clamp the indices next; negative
    // out-of-grid coords clamp to 0 anyway.
    int ix = static_cast<int>(gx); if (gx < 0) ix -= 1;  // floor toward -inf
    int iy = static_cast<int>(gy); if (gy < 0) iy -= 1;
    int iz = static_cast<int>(gz); if (gz < 0) iz -= 1;
    double fx = gx - ix;   // fractional position in the cell along x, in [0,1)
    double fy = gy - iy;
    double fz = gz - iz;

    // (c) the 8 corner indices, each clamped into [0, n-1] (boundary wall)
    const int x0 = clampi(ix,     0, g.nx - 1), x1 = clampi(ix + 1, 0, g.nx - 1);
    const int y0 = clampi(iy,     0, g.ny - 1), y1 = clampi(iy + 1, 0, g.ny - 1);
    const int z0 = clampi(iz,     0, g.nz - 1), z1 = clampi(iz + 1, 0, g.nz - 1);

    // fetch the 8 corner energies (c<x><y><z>)
    const double c000 = grid[grid_index(g, x0, y0, z0)];
    const double c100 = grid[grid_index(g, x1, y0, z0)];
    const double c010 = grid[grid_index(g, x0, y1, z0)];
    const double c110 = grid[grid_index(g, x1, y1, z0)];
    const double c001 = grid[grid_index(g, x0, y0, z1)];
    const double c101 = grid[grid_index(g, x1, y0, z1)];
    const double c011 = grid[grid_index(g, x0, y1, z1)];
    const double c111 = grid[grid_index(g, x1, y1, z1)];

    // (d) blend: interpolate along x, then y, then z (order is irrelevant to the
    // result; this nesting minimises multiplies). Weights are (1-f) and f.
    const double c00 = c000 * (1 - fx) + c100 * fx;   // bottom-front edge
    const double c10 = c010 * (1 - fx) + c110 * fx;   // bottom-back  edge
    const double c01 = c001 * (1 - fx) + c101 * fx;   // top-front    edge
    const double c11 = c011 * (1 - fx) + c111 * fx;   // top-back     edge
    const double c0  = c00 * (1 - fy) + c10 * fy;     // bottom face
    const double c1  = c01 * (1 - fy) + c11 * fy;     // top face
    return c0 * (1 - fz) + c1 * fz;                   // final interpolated energy
}

// ---------------------------------------------------------------------------
// Pose: a rigid-body placement of the ligand = a translation + a rotation.
//   We parameterise the rotation by three Euler-like angles (a,b,c) about the
//   x, y, z axes. This is the simplest pose parameterisation a learner can
//   reason about; production docking uses quaternions to avoid gimbal lock and
//   torsions for flexibility (THEORY S"real world" / Exercises). Translation
//   (tx,ty,tz) is in Angstrom.
//   POD so it copies trivially to the GPU.
// ---------------------------------------------------------------------------
struct Pose {
    double tx, ty, tz;   // translation of the ligand centroid (Angstrom)
    double a, b, c;      // rotation angles about x, y, z axes (radians)
};

// ---------------------------------------------------------------------------
// rotate_point: apply the pose's rotation (Rz * Ry * Rx) to a ligand-local atom
//   offset (lx,ly,lz), writing the rotated offset to (ox_,oy_,oz_).
//
//   We build the rotation as three elementary axis rotations applied in order
//   X then Y then Z. Each uses sin/cos of the corresponding angle. We compute
//   the trig ONCE per pose in the caller and pass it in, because sin/cos are
//   expensive and identical for every atom of a pose -- hoisting them out of the
//   per-atom loop is the single most important optimisation here (and keeps CPU
//   and GPU doing the exact same operations in the same order).
//
//   The combined matrix R = Rz*Ry*Rx is written out explicitly so there is no
//   hidden library call and the host/device math is provably identical.
// ---------------------------------------------------------------------------
DOCK_HD inline void rotate_point(double lx, double ly, double lz,
                                 double sa, double ca,   // sin/cos of angle a (x)
                                 double sb, double cb,   // sin/cos of angle b (y)
                                 double sc, double cc,   // sin/cos of angle c (z)
                                 double& ox_, double& oy_, double& oz_) {
    // R = Rz(c) * Ry(b) * Rx(a). Expanded once, by hand:
    const double r00 = cb * cc;
    const double r01 = sa * sb * cc - ca * sc;
    const double r02 = ca * sb * cc + sa * sc;
    const double r10 = cb * sc;
    const double r11 = sa * sb * sc + ca * cc;
    const double r12 = ca * sb * sc - sa * cc;
    const double r20 = -sb;
    const double r21 = sa * cb;
    const double r22 = ca * cb;
    ox_ = r00 * lx + r01 * ly + r02 * lz;
    oy_ = r10 * lx + r11 * ly + r12 * lz;
    oz_ = r20 * lx + r21 * ly + r22 * lz;
}

// ---------------------------------------------------------------------------
// score_pose: THE per-pose scoring function -- the unit of parallel work.
//   Given the energy grid, a rigid ligand (its atom offsets relative to the
//   ligand centroid, in ligand-local coordinates), and a Pose, return the total
//   interaction energy (sum over atoms of the grid energy at each transformed
//   atom). LOWER is better (more negative = stronger predicted binding).
//
//   Parameters:
//     grid     : [nx*ny*nz] precomputed energies (kcal/mol), see GridDims.
//     g        : grid geometry.
//     lx,ly,lz : [n_atoms] ligand atom offsets from its centroid (Angstrom).
//     weight   : [n_atoms] per-atom probe weight (e.g. partial charge magnitude
//                or atom-type scale). Lets some atoms count more, mirroring the
//                per-atom-type grids of real force fields. All-ones is fine.
//     n_atoms  : number of ligand atoms.
//     pose     : the rigid placement to score.
//   Returns: summed energy (double).
//
//   Complexity: O(n_atoms) per pose (8 grid reads + a few FLOPs each). The trig
//   is computed once up front, NOT per atom. The GPU runs one of these per
//   thread; the CPU loops it over all poses. Same code -> same numbers.
// ---------------------------------------------------------------------------
DOCK_HD inline double score_pose(const double* grid, const GridDims& g,
                                 const double* lx, const double* ly, const double* lz,
                                 const double* weight, int n_atoms,
                                 const Pose& pose) {
    // Hoist the six trig values out of the per-atom loop (see rotate_point).
    // sincos would be marginally faster; we keep separate sin/cos for clarity and
    // because the CPU std::sin/std::cos and device sin/cos agree to ~1e-15 here.
    const double sa = sin(pose.a), ca = cos(pose.a);
    const double sb = sin(pose.b), cb = cos(pose.b);
    const double sc = sin(pose.c), cc = cos(pose.c);

    double energy = 0.0;
    for (int k = 0; k < n_atoms; ++k) {
        // 1) rotate the ligand-local atom offset by the pose's rotation
        double rx, ry, rz;
        rotate_point(lx[k], ly[k], lz[k], sa, ca, sb, cb, sc, cc, rx, ry, rz);
        // 2) translate into the receptor (world) frame
        const double wx = rx + pose.tx;
        const double wy = ry + pose.ty;
        const double wz = rz + pose.tz;
        // 3) accumulate the (weighted) interpolated grid energy at that point
        energy += weight[k] * trilinear_energy(grid, g, wx, wy, wz);
    }
    return energy;
}
