// ===========================================================================
// src/ct_geometry.h  --  The ONE shared parallel-beam projection geometry
// ---------------------------------------------------------------------------
// Project 4.2 : Iterative / Model-Based CT Reconstruction
//
// WHY THIS FILE EXISTS  (the "HD-macro" idiom -- PATTERNS.md §2)
//   An iterative reconstruction repeatedly applies two linear operators:
//     * FORWARD projection  A : image  -> sinogram   (simulate the scan)
//     * BACKprojection      A^T : sinogram -> image  (smear residual back)
//   Both operators sweep the SAME ray geometry. If the CPU reference and the GPU
//   kernel computed that geometry with even slightly different code, their
//   reconstructions would drift apart over ~50 iterations and verification would
//   become meaningless. So we put the per-ray math in ONE header, tagged
//   `__host__ __device__`, and include it from BOTH the host reference
//   (reference_cpu.cpp, compiled by cl.exe) and the GPU kernels (kernels.cu,
//   compiled by nvcc). Same source -> same arithmetic -> exact CPU/GPU parity.
//
//   The trick: `__host__ __device__` only exists when nvcc is compiling. When
//   the plain host compiler sees this header (via reference_cpu.cpp) we must
//   erase those decorators, which is what the CT_HD macro below does.
//
//   Keep this header free of CUDA-only constructs (no __global__, no <<< >>>,
//   no cudaXxx) so the host compiler can include it unharmed.
//
// THE GEOMETRY  (2-D parallel beam -- the same one Project 4.01 uses)
//   * The object lives in a square world [-W, W] x [-W, W], W = world_half.
//   * It is sampled on an N x N pixel grid; pixel (px,py) sits at world
//         (wx, wy) = (-W + px*pix, -W + py*pix),  pix = 2W/(N-1).
//   * We take `n_angles` projections at angles theta_k = k * pi / n_angles
//     (a 180-degree parallel-beam scan is sufficient for full sampling).
//   * A detector has `n_det` bins of width `ds`; bin j measures the ray whose
//     signed distance from the origin is  s_j = (j - center) * ds, where
//     center = (n_det-1)/2. For a pixel at (wx,wy), the ray through it at angle
//     theta lands at detector coordinate  s = wx*cos(theta) + wy*sin(theta),
//     i.e. fractional bin  fidx = s/ds + center.
//
//   Forward and backprojection are TRANSPOSES of one another and MUST use this
//   identical mapping + interpolation weight so that A^T is the exact adjoint of
//   A. That adjoint relationship is what makes SIRT converge (THEORY.md §math).
//
// READ THIS AFTER: nothing (this is the foundation); BEFORE reference_cpu.h.
// ===========================================================================
#pragma once

// --- The HD decorator ------------------------------------------------------
// When nvcc compiles this header (it defines __CUDACC__) we want the functions
// to be callable from BOTH host and device. When the host compiler compiles it
// (for reference_cpu.cpp) the decorators are meaningless, so we blank them out.
#ifdef __CUDACC__
#define CT_HD __host__ __device__
#else
#define CT_HD
#endif

// ---------------------------------------------------------------------------
// detector_coord(wx, wy, cos_t, sin_t) -> fractional detector bin index.
//   This is the heart of the geometry: project the world point (wx,wy) onto the
//   detector oriented at angle theta (whose cos/sin we pass in precomputed so
//   the host and device use bit-identical trig -- see compute_trig()).
//   Returns fidx = (wx*cos + wy*sin)/ds + center, a *fractional* bin.
//   Units: wx,wy,ds in world units; cos_t,sin_t dimensionless; center in bins.
// ---------------------------------------------------------------------------
CT_HD inline float detector_coord(float wx, float wy,
                                  float cos_t, float sin_t,
                                  float ds, float center) {
    const float s = wx * cos_t + wy * sin_t;   // signed distance of the ray
    return s / ds + center;                     // -> fractional detector bin
}

// ---------------------------------------------------------------------------
// A "linear interpolation stencil" for one ray hitting the detector between two
// bins. Given the fractional index `fidx`, the ray's contribution is split
// between bin j0 = floor(fidx) with weight (1-w) and bin j0+1 with weight w.
//   Both FORWARD and BACKprojection reuse this SAME stencil:
//     * forward: value += img_pixel scattered to (j0, j0+1) with (1-w, w)
//     * back   : pixel += sino[j0]*(1-w) + sino[j0+1]*w
//   Using one stencil for both is exactly what makes them adjoint operators.
//
//   Fills j0 (the lower bin) and w (the upper-bin weight). Returns `true` only
//   when BOTH bins are inside [0, n_det); rays that miss the detector return
//   false and are skipped by the caller (they contribute nothing either way).
// ---------------------------------------------------------------------------
CT_HD inline bool interp_stencil(float fidx, int n_det, int* j0, float* w) {
    // floorf works on host and device; casting to int truncates toward zero,
    // so we floor first to get the correct lower bin for negative fidx too.
    const int lo = (int)floorf(fidx);
    *j0 = lo;
    *w  = fidx - (float)lo;                      // fractional part in [0,1)
    return (lo >= 0 && lo + 1 < n_det);          // both bins in range?
}

// ---------------------------------------------------------------------------
// pixel_world_x / pixel_world_y: map a pixel index to its world coordinate.
//   Shared so the CPU loop and the GPU thread agree on where each pixel lives.
//   pix = world units per pixel = 2*W/(N-1); W = world_half.
// ---------------------------------------------------------------------------
CT_HD inline float pixel_world(int idx, float world_half, float pix) {
    return -world_half + (float)idx * pix;
}
