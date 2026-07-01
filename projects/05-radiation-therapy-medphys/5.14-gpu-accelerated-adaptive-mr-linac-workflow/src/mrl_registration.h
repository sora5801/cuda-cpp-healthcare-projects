// ===========================================================================
// src/mrl_registration.h  --  Shared (host + device) per-voxel oART physics
// ---------------------------------------------------------------------------
// Project 5.14 : GPU-Accelerated Adaptive MR-Linac Workflow
//                (REDUCED-SCOPE TEACHING VERSION -- see ../THEORY.md "Where this
//                 sits in the real world" for the full 5-stage clinical chain)
//
// WHAT THIS HEADER IS
//   The single source of truth for the *per-voxel arithmetic* of the two GPU
//   stages we teach here:
//     (A) Demons deformable image registration (align today's MR to the planning
//         MR), and
//     (B) dose warping (carry the reference dose through the deformation onto the
//         daily anatomy).
//   The CPU reference (reference_cpu.cpp, host compiler) AND the GPU kernels
//   (kernels.cu, nvcc) both #include this file, so they run byte-for-byte
//   identical math. That is the whole trick behind exact verification
//   (PATTERNS.md section 2, the "__host__ __device__ core" idiom).
//
//   Keep this header free of CUDA-only constructs (no __global__, no launch
//   syntax) so the plain host compiler can also include it. MRL_HD expands to
//   `__host__ __device__` under nvcc and to nothing under the host compiler.
//
// THE MEDICAL PICTURE (why any of this exists)
//   An MR-Linac images the patient's anatomy *at the moment of treatment*. Since
//   tumours and organs move day to day (a full bladder, gas in the bowel, weight
//   loss), the plan made on an earlier "planning" scan may no longer fit. Online
//   adaptive radiotherapy (oART) re-derives the delivery from the daily image
//   inside the ~30-90 min the patient is on the couch. Two load-bearing steps of
//   that chain are registration and dose mapping -- exactly what we implement.
//
// COORDINATE / MEMORY CONVENTIONS (used everywhere below)
//   * Images are 2-D, row-major, size nx by ny. Flat index of voxel (x,y) is
//     idx = y*nx + x. Consecutive x are contiguous -> coalesced GPU reads.
//   * A displacement field is two float/double images u (x-shift) and v (y-shift),
//     in *voxel units*. "Deform image I by (u,v)" means: the output at (x,y) is
//     sampled from I at the moved-back location (x+u(x,y), y+v(x,y)) -- the
//     standard backward-warp / pull convention (no scatter, no races).
//
// READ THIS AFTER: util/cuda_check.cuh; BEFORE: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#pragma once

#include <cstddef>   // std::size_t
#include <cmath>     // std::floor (host); nvcc maps to device floor in device code

// --- Portable host/device decorator (the HD-macro idiom) --------------------
#ifdef __CUDACC__
#define MRL_HD __host__ __device__
#else
#define MRL_HD
#endif

// ---------------------------------------------------------------------------
// clampi: clamp an integer index into [0, n-1].
//   Used at image borders so a sample near the edge reads the nearest valid
//   voxel instead of walking off the array (a "clamp-to-edge" boundary). We
//   clamp rather than wrap (periodic) because anatomy is not periodic.
// ---------------------------------------------------------------------------
MRL_HD inline int clampi(int i, int n) {
    if (i < 0) return 0;
    if (i >= n) return n - 1;
    return i;
}

// ---------------------------------------------------------------------------
// flat_idx: row-major flat index of voxel (x,y) in an nx-by-ny image.
//   Kept as one function so CPU and GPU index identically (a common source of
//   subtle mismatches is one side transposing x and y -- centralizing it here
//   makes that impossible).
// ---------------------------------------------------------------------------
MRL_HD inline std::size_t flat_idx(int x, int y, int nx) {
    return static_cast<std::size_t>(y) * nx + x;
}

// ---------------------------------------------------------------------------
// sample_bilinear: read image `img` at a *fractional* location (fx, fy) using
//   bilinear interpolation, with clamp-to-edge borders.
//
//   WHY BILINEAR: a displacement field moves voxel (x,y) to a sub-voxel location.
//   Nearest-neighbour sampling there would be blocky and would make the Demons
//   optimisation jitter; bilinear is the cheapest smooth reconstruction and is
//   differentiable enough for the registration to converge.
//
//   Bilinear = weighted average of the 4 surrounding grid samples:
//       x0,y0 --- x1,y0
//         |          |          w = (1-a)(1-b), a(1-b), (1-a)b, a b
//       x0,y1 --- x1,y1        with a = fx-x0 (fractional x), b = fy-y0.
//
//   Parameters:
//     img : row-major image, size nx*ny (units: arbitrary MR intensity, ~[0,1])
//     nx,ny : image dimensions in voxels
//     fx,fy : fractional sample location in voxel coordinates
//   Returns: interpolated intensity (same units as img).
// ---------------------------------------------------------------------------
MRL_HD inline double sample_bilinear(const double* img, int nx, int ny,
                                     double fx, double fy) {
    // Lower-left integer corner of the enclosing cell.
    const double ffx = std::floor(fx);
    const double ffy = std::floor(fy);
    const int x0 = static_cast<int>(ffx);
    const int y0 = static_cast<int>(ffy);
    // Fractional offsets inside the cell, in [0,1).
    const double a = fx - ffx;   // horizontal blend weight
    const double b = fy - ffy;   // vertical blend weight

    // Clamp the 4 corner indices so edge samples stay in-bounds.
    const int x0c = clampi(x0,     nx);
    const int x1c = clampi(x0 + 1, nx);
    const int y0c = clampi(y0,     ny);
    const int y1c = clampi(y0 + 1, ny);

    // Gather the 4 neighbours.
    const double i00 = img[flat_idx(x0c, y0c, nx)];
    const double i10 = img[flat_idx(x1c, y0c, nx)];
    const double i01 = img[flat_idx(x0c, y1c, nx)];
    const double i11 = img[flat_idx(x1c, y1c, nx)];

    // Bilinear blend: interpolate along x on both rows, then along y.
    const double top = i00 * (1.0 - a) + i10 * a;   // y = y0 row
    const double bot = i01 * (1.0 - a) + i11 * a;   // y = y1 row
    return top * (1.0 - b) + bot * b;
}

// ---------------------------------------------------------------------------
// grad_x / grad_y: central finite-difference spatial gradient of an image at
//   the integer voxel (x,y). These are the "which way is intensity increasing"
//   directions that the Demons force pushes along.
//     d/dx I ~= (I[x+1] - I[x-1]) / 2       (central difference, O(h^2) accurate)
//   Border voxels use clamped neighbours (one-sided in effect).
// ---------------------------------------------------------------------------
MRL_HD inline double grad_x(const double* img, int nx, int ny, int x, int y) {
    const int xm = clampi(x - 1, nx);
    const int xp = clampi(x + 1, nx);
    return 0.5 * (img[flat_idx(xp, y, nx)] - img[flat_idx(xm, y, nx)]);
}
MRL_HD inline double grad_y(const double* img, int nx, int ny, int x, int y) {
    const int ym = clampi(y - 1, ny);
    const int yp = clampi(y + 1, ny);
    return 0.5 * (img[flat_idx(x, yp, nx)] - img[flat_idx(x, ym, nx)]);
}

// ---------------------------------------------------------------------------
// demons_force: Thirion's "optical-flow" demons update for ONE voxel.
//
//   THE IDEA. We want a displacement (du,dv) that nudges the *moving* image M
//   toward the *fixed* (planning) image F at this voxel. Formally we descend the
//   sum-of-squared-differences energy E = 1/2 (M(x+u) - F(x))^2. Its gradient
//   w.r.t. u is (M-F)*grad(M(x+u)); Thirion's key approximation replaces the
//   (expensive, warp-dependent) moving gradient by the FIXED image gradient
//   grad(F). A gradient-descent step is therefore MINUS that:
//
//              -(M(warped) - F) * grad(F)
//     (du,dv) = -------------------------------------
//               |grad(F)|^2 + (M(warped) - F)^2 / K
//
//   where:
//     * (M(warped) - F) is the intensity mismatch after the current warp,
//     * grad(F) = (gx, gy) is the fixed-image gradient (the push direction),
//     * the MINUS sign makes this a descent step: it moves the sampling point so
//       that next iteration M(warped) looks MORE like F (mismatch shrinks),
//     * the denominator is Thirion's normaliser: |grad F|^2 makes the step behave
//       like a unit displacement toward the matching isocontour, and the
//       (mismatch^2 / K) term ("K" ~ squared voxel spacing) keeps the step finite
//       in near-flat regions where grad(F) ~= 0 (otherwise it would divide by ~0
//       and explode).
//
//   This function returns the *incremental* velocity (du,dv) for this iteration;
//   the caller adds it to the running displacement field and then Gaussian-
//   smooths the whole field (the "diffusion" regulariser -- see kernels.cu). The
//   accumulate-then-smooth loop converges u to the true motion (here ~(3,2) vox).
//
//   Parameters:
//     m_warped : moving-image intensity already sampled at the current warp
//     f        : fixed-image intensity at this voxel
//     gx,gy    : fixed-image gradient components at this voxel
//     k_norm   : Thirion normaliser K (squared spacing; ~1 in voxel units here)
//   Outputs (by pointer): *du,*dv incremental voxel displacement.
// ---------------------------------------------------------------------------
MRL_HD inline void demons_force(double m_warped, double f,
                                double gx, double gy, double k_norm,
                                double* du, double* dv) {
    const double diff = m_warped - f;                 // intensity mismatch
    const double gmag2 = gx * gx + gy * gy;           // |grad F|^2
    // Denominator per Thirion (2003, "active" demons variant used by ITK).
    const double denom = gmag2 + (diff * diff) / k_norm;
    if (denom <= 0.0) { *du = 0.0; *dv = 0.0; return; } // fully flat, no info
    // MINUS: descend the SSD energy (see derivation above). If M is too bright
    // vs F (diff>0), step DOWN the fixed-image gradient toward a darker match.
    const double scale = -diff / denom;               // shared scalar factor
    *du = scale * gx;
    *dv = scale * gy;
}
