// ===========================================================================
// src/demons.h  --  Shared (host + device) core of Thirion's Demons DIR
// ---------------------------------------------------------------------------
// Project 4.8 : Deformable Image Registration (reduced-scope teaching version)
//
// WHAT THIS PROJECT COMPUTES
//   Deformable image registration (DIR) finds a dense DISPLACEMENT VECTOR FIELD
//   (DVF) u(x) that warps a MOVING image M onto a FIXED image F so that the two
//   look the same: F(x) ~= M(x + u(x)). Unlike rigid registration (one global
//   rotation+translation), the DVF gives every pixel its own little arrow, so it
//   can model soft-tissue deformation -- a breathing lung, a beating heart, a
//   brain that shifts between scans.
//
//   We implement the classic THIRION'S DEMONS algorithm in 2-D (the catalog
//   names Demons first among the key algorithms). Demons is an iterative
//   gradient-descent-like scheme. Each iteration does three things:
//
//     (1) WARP   : sample the moving image at the displaced coordinates,
//                  Mw(x) = M(x + u(x)),  via BILINEAR interpolation (a gather).
//     (2) FORCE  : compute a per-pixel update to the displacement from the
//                  intensity mismatch and the fixed-image gradient
//                  (the "optical-flow" / Demons force, derived in THEORY §math):
//                     du(x) = (Mw(x) - F(x)) * grad F(x)
//                             ---------------------------------
//                             |grad F(x)|^2 + (Mw(x) - F(x))^2
//                  then u <- u + du.
//     (3) REGULARIZE : GAUSSIAN-SMOOTH the whole displacement field. This is the
//                  regularization that keeps the deformation spatially coherent
//                  (a diffusion prior); without it, each pixel would move
//                  independently and the field would be noisy and non-physical.
//
//   These three steps map cleanly onto the GPU: warp and force are pure
//   per-pixel GATHERS (one thread per pixel; cf. project 4.01 CT backprojection),
//   and the Gaussian smoothing is a separable STENCIL (cf. 6.04 lattice-
//   Boltzmann / 14.02 reaction-diffusion). Hundreds of iterations over ~10^5
//   pixels (here) or ~10^7 voxels (a real 256^3 volume) are exactly why per-
//   iteration GPU parallelism matters -- see the catalog deep-dive.
//
// WHY THE PHYSICS LIVES HERE (the __host__ __device__ idiom, PATTERNS.md §2)
//   Every per-pixel formula below is marked DM_HD = __host__ __device__ under
//   nvcc (compiling for BOTH CPU and GPU) and nothing under the plain host
//   compiler. The CPU reference (reference_cpu.cpp) and the GPU kernels
//   (kernels.cu) therefore call the EXACT SAME code, so their results agree to
//   floating-point rounding. Keep this header free of CUDA-only constructs
//   (no __global__, no <<<>>>) so cl.exe/g++ can include it too.
//
// READ THIS AFTER: nothing (start here). Then reference_cpu.cpp, kernels.cu.
// ===========================================================================
#pragma once

// DM_HD expands to "__host__ __device__" only when this header is seen by the
// CUDA compiler (nvcc defines __CUDACC__). Under the plain C++ compiler those
// keywords do not exist, so DM_HD expands to nothing and the same source is
// valid ISO C++. This one macro is what lets a single formula run on both sides.
#ifdef __CUDACC__
#define DM_HD __host__ __device__
#else
#define DM_HD
#endif

// ---------------------------------------------------------------------------
// DemonsParams -- everything that defines one registration run.
//   Kept a plain struct of scalars so it can be passed BY VALUE straight into a
//   kernel (it lands in constant/parameter memory, readable by every thread with
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
//   Returning the edge value is the standard, artifact-free choice for image
//   warping (a zero border would inject a dark ring; a wrap would be nonsense
//   for anatomy).
// ---------------------------------------------------------------------------
DM_HD inline int dm_clampi(int i, int n) {
    if (i < 0)      return 0;
    if (i >= n)     return n - 1;
    return i;
}

// ---------------------------------------------------------------------------
// dm_at -- read image value at integer pixel (ix,iy) with clamp-to-edge.
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
//   This is the heart of the WARP step. A displacement moves a pixel to a
//   fractional location between grid samples, so we interpolate the four
//   surrounding pixels with weights from the fractional part:
//
//         (ix,iy) ---- (ix+1,iy)
//            |   fx-> .   |          value = (1-fx)(1-fy) I00
//            | fy         |                + fx    (1-fy) I10
//            v            |                + (1-fx) fy    I01
//        (ix,iy+1)---(ix+1,iy+1)              + fx    fy    I11
//
//   Bilinear interpolation is C0-continuous (no jumps) and cheap -- exactly the
//   trade-off the catalog's "custom CUDA trilinear interpolation kernel for
//   warp" points at (trilinear is the 3-D sibling of this 2-D bilinear).
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
//   grad F ~= ( [F(x+1)-F(x-1)]/2 , [F(y+1)-F(y-1)]/2 ). This is a 3-point
//   STENCIL in each direction (clamp-to-edge at borders). The Demons force
//   pushes the moving image "downhill" along this gradient, so we compute it
//   once from the (static) fixed image; in this teaching version we recompute it
//   each call for clarity -- a real solver precomputes it once (see THEORY).
//   Writes the two components through the out-pointers gx,gy.
// ---------------------------------------------------------------------------
DM_HD inline void dm_grad(const double* F, int x, int y, int nx, int ny,
                          double* gx, double* gy) {
    *gx = 0.5 * (dm_at(F, x + 1, y, nx, ny) - dm_at(F, x - 1, y, nx, ny));
    *gy = 0.5 * (dm_at(F, x, y + 1, nx, ny) - dm_at(F, x, y - 1, nx, ny));
}

// ---------------------------------------------------------------------------
// dm_demons_force -- the per-pixel Demons update du = (du_x, du_y) at (x,y).
//   Inputs:
//     F, M  : fixed and moving images [ny*nx], row-major, device or host.
//     ux,uy : current displacement field components [ny*nx].
//     P     : run parameters (needs nx,ny,epsilon).
//   Steps (all O(1) per pixel):
//     1. warp: sample M at (x+ux, y+uy) via bilinear -> Mw.
//     2. diff = F(x) - Mw          (intensity mismatch; SIGN MATTERS, see below).
//     3. grad F                    (direction of steepest intensity change).
//     4. Thirion's force:
//          du = diff * gradF / (|gradF|^2 + diff^2 + epsilon).
//        The denominator is the classic Demons NORMALIZATION: it makes the step
//        an adaptive length -- large where the gradient is strong and the
//        mismatch is small, damped where the gradient vanishes (flat regions
//        carry no reliable direction) or the mismatch is huge (avoid overshoot).
//        epsilon just prevents a 0/0 in perfectly flat, perfectly matched areas.
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
//   A 2-D Gaussian G(x,y) factorizes as G(x)*G(y), so instead of an O(r^2)
//   box we do two O(r) passes: blur along x, then along y. Each function below
//   computes the smoothed value of the `src` field at pixel (x,y) for one axis,
//   normalizing by the sum of the weights it actually used (which shrinks near
//   the border because clamped samples are folded in). Both the CPU reference
//   and the GPU kernel call these, so the regularization is identical.
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
