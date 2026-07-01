// ===========================================================================
// src/demons.h  --  Shared (host + device) core of Thirion's Demons DIR
// ---------------------------------------------------------------------------
// Project 5.5 : Deformable Dose Accumulation & Adaptive Radiotherapy
//               (reduced-scope 2-D teaching version)
//
// WHERE THIS FITS IN THE PROJECT
//   Adaptive radiotherapy (ART) has three stages (catalog deep-dive):
//     (1) image the patient today (CBCT / MR-Linac),
//     (2) DEFORMABLY REGISTER today's anatomy to the planning anatomy -> a dense
//         displacement vector field (DVF) u(x),
//     (3) WARP each fraction's dose by that DVF and ACCUMULATE it in the planning
//         frame, so the total delivered dose is anatomically meaningful.
//   This header owns the per-pixel math of stage (2): Thirion's DEMONS DIR. The
//   dose warp + accumulation of stage (3) lives in dose.h. main.cu chains them.
//
//   DIR itself is exactly project 4.8's algorithm; we reuse the same didactic
//   core here (a project is self-contained -- CLAUDE.md §10 -- so the header is
//   duplicated, not shared, on purpose). What is NEW in 5.5 is dose.h: mapping a
//   physical DOSE through the DVF and summing deformed doses.
//
// WHAT DEMONS COMPUTES
//   Given a FIXED image F (the planning anatomy we register TO) and a MOVING
//   image M (today's anatomy), find a displacement field u(x)=(ux,uy) so that
//   F(x) ~= M(x + u(x)). Unlike a rigid shift, u gives every pixel its own little
//   arrow, so it models the soft-tissue deformation between two scans (a filling
//   bladder, a shrinking tumour, a breathing lung). Each iteration does three
//   things:
//     (1) WARP   : sample M at the displaced coordinates, Mw(x)=M(x+u(x)), via
//                  BILINEAR interpolation (a per-pixel gather; the 2-D sibling of
//                  the catalog's "trilinear warp").
//     (2) FORCE  : per-pixel displacement update from the intensity mismatch and
//                  the fixed-image gradient (Thirion's optical-flow force):
//                     du(x) = (F - Mw) * gradF / (|gradF|^2 + (F-Mw)^2 + eps);
//                  then u <- u + du.
//     (3) REGULARIZE : GAUSSIAN-SMOOTH the whole field (a diffusion prior) so the
//                  deformation stays spatially coherent instead of noisy.
//
//   Maps to the GPU as: warp+force are per-pixel GATHERS (one thread per pixel;
//   cf. 4.01 CT backprojection), and the Gaussian is a separable STENCIL (cf.
//   6.04 / 14.02). See THEORY §GPU mapping.
//
// WHY THE PHYSICS LIVES HERE (the __host__ __device__ idiom, PATTERNS.md §2)
//   Every per-pixel formula below is marked DM_HD = __host__ __device__ under
//   nvcc, and nothing under the plain host compiler. The CPU reference
//   (reference_cpu.cpp) and the GPU kernels (kernels.cu) therefore call the EXACT
//   SAME code, so their fields agree to floating-point rounding. Keep this header
//   free of CUDA-only constructs (no __global__, no <<<>>>) so cl.exe/g++ can
//   include it too.
//
// READ THIS AFTER: nothing (start here). Then dose.h, reference_cpu.cpp, kernels.cu.
// ===========================================================================
#pragma once

#include <cmath>   // std::floor, std::exp -- used by the inline formulas below

// DM_HD expands to "__host__ __device__" only when this header is seen by the
// CUDA compiler (nvcc defines __CUDACC__). Under the plain C++ compiler those
// keywords do not exist, so DM_HD expands to nothing and the same source stays
// valid ISO C++. This single macro is what lets one formula run on both sides.
#ifdef __CUDACC__
#define DM_HD __host__ __device__
#else
#define DM_HD
#endif

// ---------------------------------------------------------------------------
// DemonsParams -- everything that defines one registration run.
//   Kept a plain struct of scalars so it can be passed BY VALUE straight into a
//   kernel (it lands in parameter/constant memory, readable by every thread with
//   no extra copies). All fields are host-set once and never mutated on-device.
// ---------------------------------------------------------------------------
struct DemonsParams {
    int    nx, ny;        // image width / height in pixels (row-major, size nx*ny)
    int    iters;         // number of Demons iterations (the outer solver loop)
    double sigma;         // Gaussian smoothing std-dev in pixels (regularization)
    int    radius;        // Gaussian kernel half-width in pixels (>= ceil(3*sigma))
    double epsilon;       // small floor added to the force denominator (stability)
};

// ---------------------------------------------------------------------------
// dm_clampi -- clamp an integer index into [0, n-1].
//   Used at image borders so a sample that falls just outside the grid reuses
//   the edge pixel ("clamp-to-edge" boundary) instead of reading out of bounds.
//   For anatomy this is the right choice: a zero border would inject a dark ring;
//   a wrap would fold the far side of the body onto the near side (nonsense).
// ---------------------------------------------------------------------------
DM_HD inline int dm_clampi(int i, int n) {
    if (i < 0)  return 0;
    if (i >= n) return n - 1;
    return i;
}

// ---------------------------------------------------------------------------
// dm_at -- read an image value at integer pixel (ix,iy) with clamp-to-edge.
//   img is a row-major [ny*nx] array; the linear index is iy*nx + ix. Marked
//   inline + DM_HD so both the CPU loops and the GPU threads share it.
// ---------------------------------------------------------------------------
DM_HD inline double dm_at(const double* img, int ix, int iy, int nx, int ny) {
    ix = dm_clampi(ix, nx);
    iy = dm_clampi(iy, ny);
    return img[iy * nx + ix];
}

// ---------------------------------------------------------------------------
// dm_bilinear -- sample `img` at a CONTINUOUS coordinate (px,py).
//   This is the heart of the WARP step (and, in dose.h, of the DOSE warp). A
//   displacement moves a pixel to a fractional location between grid samples, so
//   we interpolate the four surrounding pixels with weights from the fractional
//   part:
//
//         (ix,iy) ---- (ix+1,iy)
//            |   fx-> .   |          value = (1-fx)(1-fy) I00
//            | fy         |                + fx    (1-fy) I10
//            v            |                + (1-fx) fy    I01
//        (ix,iy+1)---(ix+1,iy+1)              + fx    fy    I11
//
//   Bilinear interpolation is C0-continuous (no jumps) and cheap -- exactly the
//   trade-off the catalog's "custom CUDA trilinear interpolation kernel for warp"
//   points at (trilinear is the 3-D sibling of this 2-D bilinear).
//   Complexity: 4 memory reads + a handful of FMAs, O(1) per sample.
// ---------------------------------------------------------------------------
DM_HD inline double dm_bilinear(const double* img, double px, double py,
                                int nx, int ny) {
    // floor() gives the top-left integer corner; the remainder is the weight.
    const int ix = (int)floor(px);
    const int iy = (int)floor(py);
    const double fx = px - (double)ix;   // horizontal fractional part in [0,1)
    const double fy = py - (double)iy;   // vertical   fractional part in [0,1)

    // Fetch the 2x2 neighbourhood (each read is clamp-to-edge at the border).
    const double i00 = dm_at(img, ix,     iy,     nx, ny);
    const double i10 = dm_at(img, ix + 1, iy,     nx, ny);
    const double i01 = dm_at(img, ix,     iy + 1, nx, ny);
    const double i11 = dm_at(img, ix + 1, iy + 1, nx, ny);

    // Interpolate along x on both rows, then blend the two rows along y.
    const double a = i00 * (1.0 - fx) + i10 * fx;   // top    row at fractional x
    const double b = i01 * (1.0 - fx) + i11 * fx;   // bottom row at fractional x
    return a * (1.0 - fy) + b * fy;
}

// ---------------------------------------------------------------------------
// dm_grad -- CENTRAL-DIFFERENCE gradient of the fixed image at pixel (x,y).
//   grad F ~= ( [F(x+1)-F(x-1)]/2 , [F(y+1)-F(y-1)]/2 ). A 3-point STENCIL in
//   each direction (clamp-to-edge at borders). The Demons force pushes the moving
//   image "downhill" along this gradient. In this teaching version we recompute
//   it each call for clarity; a production solver precomputes it once (THEORY).
//   Writes the two components through the out-pointers gx,gy.
// ---------------------------------------------------------------------------
DM_HD inline void dm_grad(const double* F, int x, int y, int nx, int ny,
                          double* gx, double* gy) {
    *gx = 0.5 * (dm_at(F, x + 1, y, nx, ny) - dm_at(F, x - 1, y, nx, ny));
    *gy = 0.5 * (dm_at(F, x, y + 1, nx, ny) - dm_at(F, x, y - 1, nx, ny));
}

// ---------------------------------------------------------------------------
// dm_demons_force -- the per-pixel Demons update du=(du_x,du_y) at (x,y).
//   Inputs:
//     F, M  : fixed and moving images [ny*nx], row-major, device or host.
//     ux,uy : current displacement field components [ny*nx].
//     P     : run parameters (needs nx,ny,epsilon).
//   Steps (all O(1) per pixel):
//     1. warp : sample M at (x+ux, y+uy) via bilinear -> Mw.
//     2. diff = F(x) - Mw          (intensity mismatch; SIGN MATTERS, see below).
//     3. grad F                    (direction of steepest intensity change).
//     4. Thirion's force:
//          du = diff * gradF / (|gradF|^2 + diff^2 + epsilon).
//        The denominator is the classic Demons NORMALIZATION: it makes the step
//        an adaptive length -- large where the gradient is strong and the
//        mismatch is small, damped where the gradient vanishes (flat regions
//        carry no reliable direction) or the mismatch is huge (avoid overshoot).
//        epsilon prevents a 0/0 in perfectly flat, perfectly matched areas.
//
//   WHY diff = F - Mw (not Mw - F): we want to DECREASE the squared mismatch
//   (Mw - F)^2. Near a solution grad(Mw) ~= grad F, so gradient descent on that
//   energy gives du proportional to -(Mw - F)*gradF = +(F - Mw)*gradF. Using the
//   other sign makes every step climb the SSD instead of descending it -- the
//   field runs away. This is exactly Thirion's published optical-flow force.
//
//   The result is ADDED to the displacement by the caller (kernel or CPU loop),
//   then the whole field is Gaussian-smoothed (dm_gauss_* below).
// ---------------------------------------------------------------------------
DM_HD inline void dm_demons_force(const double* F, const double* M,
                                  const double* ux, const double* uy,
                                  int x, int y, const DemonsParams& P,
                                  double* du_x, double* du_y) {
    const int i = y * P.nx + x;              // this pixel's linear index
    const double u = ux[i], v = uy[i];       // its current displacement

    // (1) WARP: where does the moving image "come from" for this fixed pixel?
    const double Mw = dm_bilinear(M, (double)x + u, (double)y + v, P.nx, P.ny);

    // (2) intensity mismatch we drive to zero. F - Mw so the step DESCENDS the
    //     SSD energy (see the sign discussion in this function's header).
    const double diff = F[i] - Mw;

    // (3) fixed-image gradient (the reliable direction to move along).
    double gx, gy;
    dm_grad(F, x, y, P.nx, P.ny, &gx, &gy);

    // (4) Thirion normalization: scale = diff / (|gradF|^2 + diff^2 + eps).
    const double denom = gx * gx + gy * gy + diff * diff + P.epsilon;
    const double scale = diff / denom;
    *du_x = scale * gx;      // component of the step along x
    *du_y = scale * gy;      // component of the step along y
}

// ---------------------------------------------------------------------------
// dm_gauss_weight -- unnormalized 1-D Gaussian weight exp(-t^2 / (2 sigma^2)).
//   Used to build the separable smoothing kernel. We normalize by the running
//   weight sum inside the passes below, so an unnormalized weight is fine here.
// ---------------------------------------------------------------------------
DM_HD inline double dm_gauss_weight(int t, double sigma) {
    const double s2 = 2.0 * sigma * sigma;
    return exp(-(double)(t * t) / s2);
}

// ---------------------------------------------------------------------------
// dm_gauss_x / dm_gauss_y -- ONE pixel of a SEPARABLE Gaussian blur.
//   A 2-D Gaussian G(x,y) factorizes as G(x)*G(y), so instead of an O(r^2) box
//   we do two O(r) passes: blur along x, then along y. Each function computes the
//   smoothed value of the `src` field at pixel (x,y) for one axis, normalizing by
//   the sum of the weights it actually used (which shrinks near the border,
//   because clamped samples fold in). Both the CPU reference and the GPU kernel
//   call these, so the regularization is identical.
//
//   These are STENCILS with a (2*radius+1)-wide footprint; the GPU runs one
//   thread per output pixel and reads its neighbourhood from global memory (a
//   shared-memory tiled version is left as an exercise -- see THEORY §GPU).
// ---------------------------------------------------------------------------
DM_HD inline double dm_gauss_x(const double* src, int x, int y,
                               int nx, int ny, double sigma, int radius) {
    double acc = 0.0, wsum = 0.0;
    for (int t = -radius; t <= radius; ++t) {
        const double w = dm_gauss_weight(t, sigma);
        acc  += w * dm_at(src, x + t, y, nx, ny);   // clamp-to-edge horizontally
        wsum += w;
    }
    return acc / wsum;                               // normalize -> preserves DC
}

DM_HD inline double dm_gauss_y(const double* src, int x, int y,
                               int nx, int ny, double sigma, int radius) {
    double acc = 0.0, wsum = 0.0;
    for (int t = -radius; t <= radius; ++t) {
        const double w = dm_gauss_weight(t, sigma);
        acc  += w * dm_at(src, x, y + t, nx, ny);    // clamp-to-edge vertically
        wsum += w;
    }
    return acc / wsum;
}
