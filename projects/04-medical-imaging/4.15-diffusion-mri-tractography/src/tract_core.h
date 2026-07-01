// ===========================================================================
// src/tract_core.h  --  Shared (host + device) deterministic tractography core
// ---------------------------------------------------------------------------
// Project 4.15 : Diffusion MRI & Tractography
//
// WHAT THIS FILE IS
//   The per-streamline stepping math, shared by the CPU reference and the GPU
//   kernel via the DTI_HD macro (defined in dti_core.h). Keeping it in one HD
//   header means both paths trace BYTE-FOR-BYTE identical streamlines, so we can
//   verify the GPU streamlines against the CPU ones exactly. (PATTERNS.md §2.)
//
// THE ALGORITHM: DETERMINISTIC ("streamline") TRACTOGRAPHY
//   Once every voxel has a principal fiber direction v1 (from the DTI fit), we
//   reconstruct white-matter pathways by INTEGRATING that direction field:
//     start at a seed point, look up the local fiber direction (trilinearly
//     interpolated from the 8 surrounding voxels), take a small step along it,
//     repeat. This is Euler integration of  dr/ds = v1(r).
//   We stop when:
//     * the path leaves the volume,
//     * the local anisotropy FA drops below fa_min (we left the white matter),
//     * or the direction turns more sharply than acos(cos_min) in one step
//       (a curvature limit; real fibers do not kink).
//   This is the DETERMINISTIC cousin of the PROBABILISTIC iFOD2 tractography in
//   the catalog: iFOD2 samples the fiber ORIENTATION DISTRIBUTION with a random
//   number generator (cuRAND) to explore crossing fibers. We use the single
//   principal direction and no randomness, which keeps the demo's output
//   reproducible (a hard requirement -- see THEORY "Numerical considerations").
//
//   WHY TRILINEAR INTERPOLATION IS THE GPU TEACHING POINT
//   Every step reads the direction field at a NON-integer position, so we blend
//   the 8 neighbouring voxels' vectors by their volume weights. On the GPU this
//   is exactly what TEXTURE MEMORY hardware does for free; here we spell the
//   interpolation out by hand (identically on host and device) so the math is
//   visible and verifiable, and we note in THEORY where textures would take over.
//
// READ THIS AFTER: dti_core.h. USED BY: reference_cpu.cpp and kernels.cu.
// ===========================================================================
#pragma once

#include "dti_core.h"   // DTI_HD, VoxelResult
#include <cmath>        // std::floor, std::sqrt, std::fabs

// ---------------------------------------------------------------------------
// clampi: clamp an integer index into [0, hi] (inclusive). Used to keep the 8
//   trilinear neighbours inside the grid at the boundary (clamp-to-edge, the
//   same rule a CUDA texture with cudaAddressModeClamp uses).
// ---------------------------------------------------------------------------
DTI_HD inline int clampi(int v, int hi) {
    if (v < 0)  return 0;
    if (v > hi) return hi;
    return v;
}

// ---------------------------------------------------------------------------
// sample_dir: trilinearly interpolate the principal direction (and FA) at a
//   continuous voxel position (px,py,pz).
//
//   fit[]         : per-voxel VoxelResult array (has v1x/v1y/v1z and fa).
//   nx,ny,nz      : grid dimensions.
//   ref(dx,dy,dz) : the direction of the voxel at the LAST step, used to keep
//                   the interpolated eigenvectors CONSISTENTLY oriented before
//                   blending (an eigenvector and its negation are equivalent, so
//                   naively averaging can cancel; we flip each neighbour to align
//                   with `ref` first). This is the standard fix for eigenvector
//                   sign ambiguity in tractography.
//   Outputs the interpolated unit direction (odx,ody,odz) and FA (ofa).
//   Returns false if the position is outside the padded grid (never happens
//   after clamping, but kept for clarity).
// ---------------------------------------------------------------------------
DTI_HD inline bool sample_dir(const VoxelResult* fit, int nx, int ny, int nz,
                              double px, double py, double pz,
                              double refx, double refy, double refz,
                              double& odx, double& ody, double& odz, double& ofa) {
    // Integer base corner of the enclosing cell and the fractional offsets.
    const int x0 = (int)std::floor(px), y0 = (int)std::floor(py), z0 = (int)std::floor(pz);
    const double fx = px - x0, fy = py - y0, fz = pz - z0;

    double ax = 0, ay = 0, az = 0, af = 0;   // weighted accumulators
    // Loop over the 8 corners of the trilinear cell.
    DTI_UNROLL
    for (int c = 0; c < 8; ++c) {
        const int dx = c & 1, dy = (c >> 1) & 1, dz = (c >> 2) & 1;
        const int xi = clampi(x0 + dx, nx - 1);
        const int yi = clampi(y0 + dy, ny - 1);
        const int zi = clampi(z0 + dz, nz - 1);
        // Trilinear weight = product of the per-axis blend factors.
        const double wx = dx ? fx : (1.0 - fx);
        const double wy = dy ? fy : (1.0 - fy);
        const double wz = dz ? fz : (1.0 - fz);
        const double w  = wx * wy * wz;

        const VoxelResult& R = fit[(size_t)zi * ny * nx + (size_t)yi * nx + xi];
        // Align this neighbour's eigenvector with the reference direction so the
        // vectors reinforce instead of cancel (sign-ambiguity fix). dot<0 => flip.
        double vx = R.v1x, vy = R.v1y, vz = R.v1z;
        const double dot = vx*refx + vy*refy + vz*refz;
        if (dot < 0.0) { vx = -vx; vy = -vy; vz = -vz; }
        ax += w * vx; ay += w * vy; az += w * vz;
        af += w * R.fa;
    }
    // Renormalise the blended direction to unit length.
    const double n = std::sqrt(ax*ax + ay*ay + az*az);
    if (n <= 1e-12) { odx = refx; ody = refy; odz = refz; ofa = af; return true; }
    odx = ax / n; ody = ay / n; odz = az / n; ofa = af;
    return true;
}

// ---------------------------------------------------------------------------
// nearest_dir: the principal direction of the voxel CONTAINING (px,py,pz), used
//   to seed the streamline's reference direction on the very first step (there is
//   no previous step to align against yet).
// ---------------------------------------------------------------------------
DTI_HD inline void nearest_dir(const VoxelResult* fit, int nx, int ny, int nz,
                               double px, double py, double pz,
                               double& dx, double& dy, double& dz) {
    const int xi = clampi((int)(px + 0.5), nx - 1);
    const int yi = clampi((int)(py + 0.5), ny - 1);
    const int zi = clampi((int)(pz + 0.5), nz - 1);
    const VoxelResult& R = fit[(size_t)zi * ny * nx + (size_t)yi * nx + xi];
    dx = R.v1x; dy = R.v1y; dz = R.v1z;
}
