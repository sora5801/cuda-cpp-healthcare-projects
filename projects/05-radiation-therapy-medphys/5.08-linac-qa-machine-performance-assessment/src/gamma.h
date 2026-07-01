// ===========================================================================
// src/gamma.h  --  The ONE TRUE gamma-index math, shared by CPU and GPU
// ---------------------------------------------------------------------------
// Project 5.8 : Linac QA & Machine Performance Assessment  (catalog ID 5.8)
//
// WHY THIS HEADER EXISTS  (the "__host__ __device__ core" idiom, PATTERNS.md §2)
//   The single most useful pattern in this repo: put the per-element physics in
//   ONE header, decorated `__host__ __device__`, so the CPU reference
//   (reference_cpu.cpp, compiled by cl.exe / g++) and the GPU kernel
//   (kernels.cu, compiled by nvcc) run BYTE-FOR-BYTE IDENTICAL arithmetic. That
//   turns verification from "agree within a fudge factor" into "agree exactly",
//   which is the honest, teachable outcome for an integer/deterministic problem.
//
//   Everything here is plain scalar math on `float`/`int` -- NO CUDA types, NO
//   __global__ -- so the host compiler can include it happily. The GPU-only
//   launch machinery lives in kernels.cuh / kernels.cu.
//
// WHAT THE GAMMA INDEX IS  (the science lives in THEORY.md §1-2)
//   Radiotherapy QA asks: "does the dose the linac actually delivered match the
//   dose the treatment plan intended?" A naive pixel-by-pixel dose difference is
//   too harsh -- in a steep dose gradient (a beam edge), a sub-millimetre spatial
//   shift produces a huge dose difference even when the machine is fine. Low
//   (2018, Med Phys) introduced the GAMMA INDEX to fix this: it accepts a
//   measured point if there exists ANY nearby reference point that is close in
//   BOTH space AND dose, trading the two off through two tolerances:
//       dose-difference criterion   DD   (e.g. 3% of the prescription dose)
//       distance-to-agreement       DTA  (e.g. 3 mm)
//   The generalised distance between a measured point m and a reference point r
//   in this combined dose/space is
//       Gamma(m, r) = sqrt(  (|D_m - D_r| / DD)^2  +  (dist(m,r) / DTA)^2  )
//   and the gamma VALUE at m is the minimum of that over all reference points:
//       gamma(m) = min_r Gamma(m, r).
//   gamma(m) <= 1  =>  the point PASSES (there is an acceptably-close reference
//   point); gamma(m) > 1 => it FAILS. The clinical "gamma pass rate" is the
//   percentage of evaluated points with gamma <= 1 (TG-218 wants >= 95% at
//   3%/3mm for per-beam IMRT QA). This file computes exactly that quantity.
//
// READ THIS BEFORE: reference_cpu.cpp (loops this), kernels.cu (one thread per
// measured point calls this). See ../THEORY.md for the full derivation.
// ===========================================================================
#pragma once

// ---------------------------------------------------------------------------
// GAMMA_HD -- the host/device decorator macro.
//   When compiled by nvcc (__CUDACC__ is defined), we tag the inline functions
//   `__host__ __device__` so the SAME source compiles for both the CPU and the
//   GPU. When compiled by the plain host compiler, those keywords do not exist,
//   so the macro expands to nothing. This is the crux of CPU/GPU parity.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define GAMMA_HD __host__ __device__
#else
#define GAMMA_HD
#endif

// We deliberately use the C math functions (fabsf/sqrtf) rather than the C++
// <cmath> overloads so that host and device pick the identical single-precision
// routine. On the device nvcc maps these to the same hardware paths regardless.
#include <math.h>

// ---------------------------------------------------------------------------
// GammaParams -- the fixed geometry + tolerances of one gamma comparison.
//   Bundled in a plain struct so the identical values reach both the CPU loop
//   and the GPU kernel (passed by value into the kernel -> lands in registers /
//   constant param space). All doses are in the same arbitrary units as the
//   input planes; distances are in millimetres.
// ---------------------------------------------------------------------------
struct GammaParams {
    int   nx;             // plane width  (columns), in pixels
    int   ny;             // plane height (rows),    in pixels
    float spacing_mm;     // physical size of one pixel, mm (square pixels)
    float dd;             // dose-difference criterion, in DOSE UNITS (already
                          //   converted from a percent of the normalisation dose
                          //   by the caller -- see reference_cpu.cpp)
    float dta_mm;         // distance-to-agreement criterion, in mm
    int   search_radius;  // half-width of the neighbourhood searched, in PIXELS.
                          //   We only look within +/- search_radius pixels of the
                          //   measured point; anything farther already exceeds a
                          //   few DTA and cannot lower the minimum (THEORY §3).
    float pass_gamma;     // the pass threshold (conventionally 1.0)
};

// ---------------------------------------------------------------------------
// gamma_value_at -- compute gamma(m) for the measured point at pixel (mx, my).
//
//   This is THE hot inner computation, called once per measured pixel by BOTH
//   the CPU reference and the GPU kernel. It scans a square window of reference
//   pixels around (mx, my), forms the combined dose/distance disagreement for
//   each, and returns the square root of the minimum squared gamma. Returning
//   the true gamma (not gamma^2) keeps the caller simple; the sqrt is one op.
//
//   Parameters:
//     meas  : measured (delivered / EPID) dose plane, row-major [ny*nx]
//     ref   : reference (planned) dose plane,        row-major [ny*nx]
//     mx,my : the measured pixel this call evaluates (0-based, mx in [0,nx),
//             my in [0,ny))
//     p     : the tolerances + geometry (see GammaParams)
//
//   Returns: gamma(m) >= 0. A value <= p.pass_gamma means the point passes.
//
//   Determinism note: the loop visits reference pixels in a FIXED order and only
//   ever takes a `min`, which is order-independent and identical on host and
//   device -> the CPU and GPU results are bit-for-bit equal (verified to 0 tol).
// ---------------------------------------------------------------------------
GAMMA_HD inline float gamma_value_at(const float* meas, const float* ref,
                                     int mx, int my, GammaParams p) {
    // Dose at the measured point we are scoring.
    const float dm = meas[(size_t)my * p.nx + mx];

    // 1 / DD^2 and 1 / DTA^2 precomputed as multipliers (divide once, not per
    // reference pixel). Guard against a zero tolerance (caller sets sane values).
    const float inv_dd2  = (p.dd     > 0.0f) ? 1.0f / (p.dd     * p.dd)     : 0.0f;
    const float inv_dta2 = (p.dta_mm > 0.0f) ? 1.0f / (p.dta_mm * p.dta_mm) : 0.0f;

    // Running minimum of gamma^2. Start "very large" so the first candidate wins.
    float best = 3.4e38f;   // ~FLT_MAX; any real gamma^2 is far below this

    // Search a square window of reference pixels centred on (mx,my), clamped to
    // the plane bounds. We compare against REFERENCE points (the classic "global
    // gamma" convention: the measured point looks for a close-enough planned
    // point). Distances are measured in mm via the pixel spacing.
    const int x0 = mx - p.search_radius, x1 = mx + p.search_radius;
    const int y0 = my - p.search_radius, y1 = my + p.search_radius;

    for (int ry = y0; ry <= y1; ++ry) {
        if (ry < 0 || ry >= p.ny) continue;              // stay inside the plane
        for (int rx = x0; rx <= x1; ++rx) {
            if (rx < 0 || rx >= p.nx) continue;

            // Spatial separation (mm): pixel offset * pixel size.
            const float ddx = (float)(rx - mx) * p.spacing_mm;
            const float ddy = (float)(ry - my) * p.spacing_mm;
            const float dist2 = ddx * ddx + ddy * ddy;   // mm^2

            // Dose separation (dose units) between measured m and reference r.
            const float dr = ref[(size_t)ry * p.nx + rx];
            const float diff = dm - dr;                  // may be +/-

            // Combined gamma^2 = (dose term) + (space term), each normalised by
            // its tolerance so both are dimensionless and directly comparable.
            const float g2 = diff * diff * inv_dd2 + dist2 * inv_dta2;

            // Keep the smallest -- gamma(m) is the closest achievable agreement.
            if (g2 < best) best = g2;
        }
    }

    // gamma = sqrt(min gamma^2). fabsf/sqrtf are the single-precision C routines,
    // identical on host and device -> exact CPU/GPU parity.
    return sqrtf(best);
}
