// ===========================================================================
// src/grid_core.h  --  The shared __host__ __device__ gridding "physics" core
// ---------------------------------------------------------------------------
// Project 4.33 : Real-Time MRI Reconstruction
//
// WHY THIS FILE EXISTS  (PATTERNS.md section 2 -- the single most useful idiom)
//   A real-time MRI reconstruction from RADIAL (non-Cartesian) k-space is a
//   GRIDDING NUFFT: every acquired sample is SPREAD onto a Cartesian grid with a
//   small convolution kernel, the grid is inverse-FFT'd, then "deapodized". The
//   spreading and the deapodization both hinge on ONE special function -- the
//   Kaiser-Bessel (KB) window and its analytic Fourier transform. If the CPU
//   reference and the GPU kernel used SEPARATE copies of that math, tiny rounding
//   differences would make the GPU-vs-CPU check meaningless.
//
//   So every per-sample / per-pixel formula lives ONCE, here, as `__host__
//   __device__` inline functions. reference_cpu.cpp includes this through the plain
//   host compiler; kernels.cu includes it through nvcc. Both therefore run
//   BYTE-FOR-BYTE the same arithmetic -- verification reduces to "does our hand
//   radix-2 FFT agree with cuFFT?" rather than "do two gridders agree?".
//
//   HARD RULE (PATTERNS.md section 2): keep this header free of CUDA-only *types*
//   and of `__global__` kernels, so the host compiler can include it. Only the HD
//   decorator macro, a tiny Cplx struct, and plain inline functions live here.
//
// READ THIS BEFORE: reference_cpu.h, kernels.cuh. The science/math behind each
// formula is in ../THEORY.md.
// ===========================================================================
#pragma once

#include <cmath>     // std::exp, std::sqrt, std::fabs, std::floor (host side)

// ---------------------------------------------------------------------------
// The HD decorator idiom.
//   When compiled by nvcc, __CUDACC__ is defined and we tag every shared function
//   `__host__ __device__` so it runs on BOTH the CPU and inside a kernel. Under the
//   plain host compiler (cl.exe / g++) those keywords do not exist, so GRID_HD
//   expands to nothing.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define GRID_HD __host__ __device__
#else
#define GRID_HD
#endif

// ---------------------------------------------------------------------------
// A minimal single-precision complex number.
//   MRI data is complex: a k-space sample and an image pixel are both complex. We
//   deliberately use our OWN tiny struct (not std::complex, not cuComplex) so the
//   SAME type name works on host and device, and the layout is trivially
//   {float re; float im;} -- bit-compatible with cuFFT's cufftComplex (float2) for
//   a plain reinterpret_cast. FP32 is what real scanners and cuFFT use (THEORY
//   "numerics").
// ---------------------------------------------------------------------------
struct Cplx {
    float re;   // real part
    float im;   // imaginary part
};

// c_make: build a complex from parts (a readable constructor usable on device).
GRID_HD inline Cplx c_make(float re, float im) { Cplx z; z.re = re; z.im = im; return z; }

// c_add: componentwise complex addition (used to accumulate onto the grid).
GRID_HD inline Cplx c_add(Cplx a, Cplx b) { return c_make(a.re + b.re, a.im + b.im); }

// c_scale: multiply a complex by a real scalar (density compensation, KB weight).
GRID_HD inline Cplx c_scale(Cplx a, float s) { return c_make(a.re * s, a.im * s); }

// c_abs: magnitude |z| = sqrt(re^2 + im^2). This is the pixel BRIGHTNESS in the
// final magnitude image a radiologist would view (phase is usually discarded).
// sqrtf has both a host and a device implementation, so this __host__ __device__
// function compiles cleanly for both and gives bit-identical results.
GRID_HD inline float c_abs(Cplx z) { return sqrtf(z.re * z.re + z.im * z.im); }

// ---------------------------------------------------------------------------
// GriddingParams: the handful of constants that define the gridding geometry.
//   Passed by value into both the CPU reference and the GPU kernel so the two
//   share exactly one description of the grid. All fields are plain scalars (no
//   pointers, no CUDA types) so this struct is trivially copyable to the device.
// ---------------------------------------------------------------------------
struct GriddingParams {
    int   n;          // Cartesian grid side length (n x n, a power of two for FFT)
    int   kb_w;       // Kaiser-Bessel kernel full width in grid cells (we use 4)
    float kb_beta;    // Kaiser-Bessel shape parameter beta (controls sidelobes)
    float kmax;       // max |k| radius in grid units (= n/2; edge of k-space)
};

// ---------------------------------------------------------------------------
// bessel_i0: the zeroth-order modified Bessel function of the first kind, I0(x).
//   The Kaiser-Bessel window is defined in terms of I0. We evaluate it with the
//   classic Abramowitz & Stegun polynomial series -- a short, deterministic sum
//   that behaves identically on host and device (no library I0 exists on device).
//   Accuracy is ~1e-7 relative, far finer than our FP32 needs. THEORY "the math"
//   derives why I0 appears.
//
//   The series (A&S 9.8.1/9.8.2) splits at |x|=3.75 for numerical conditioning:
//     small x : a power series in (x/3.75)^2
//     large x : an asymptotic-style series in (3.75/x), scaled by exp(x)/sqrt(x)
// ---------------------------------------------------------------------------
GRID_HD inline double bessel_i0(double x) {
    const double ax = x < 0.0 ? -x : x;                 // I0 is even: I0(x)=I0(|x|)
    if (ax < 3.75) {
        const double t  = x / 3.75;
        const double y  = t * t;                        // (x/3.75)^2
        // Horner form of the small-argument polynomial (A&S 9.8.1).
        return 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492
               + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))));
    }
    const double t = 3.75 / ax;                         // 3.75/|x|, in (0,1]
    // Large-argument series (A&S 9.8.2), multiplied by exp(|x|)/sqrt(|x|).
    const double poly = 0.39894228 + t * (0.01328592 + t * (0.00225319
                       + t * (-0.00157565 + t * (0.00916281 + t * (-0.02057706
                       + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377)))))));
    return (std::exp(ax) / std::sqrt(ax)) * poly;
}

// ---------------------------------------------------------------------------
// kb_weight: the Kaiser-Bessel convolution weight for a sample at grid-distance
//   `dist` cells from a target grid cell.
//     w(dist) = I0( beta * sqrt(1 - (2*dist/W)^2) ) / I0(beta)   for |dist| <= W/2
//             = 0                                                 otherwise
//   This is the finite-support interpolation kernel that SPREADS one non-Cartesian
//   sample onto the ~W nearest Cartesian cells in each axis. The KB kernel is the
//   near-optimal gridding window (minimal aliasing for a given width); THEORY "the
//   algorithm" explains why it beats a plain triangular/Gaussian kernel.
//     * dist : distance in grid cells (>= 0) between the sample and the grid cell
//     * p    : the gridding geometry (kb_w = W, kb_beta = beta)
//   Returned weight is normalized by I0(beta) so the peak weight is 1.
// ---------------------------------------------------------------------------
GRID_HD inline float kb_weight(float dist, const GriddingParams& p) {
    const float half = 0.5f * static_cast<float>(p.kb_w);   // W/2, the support radius
    if (dist > half) return 0.0f;                           // outside the kernel
    const float r = 2.0f * dist / static_cast<float>(p.kb_w);   // in [0,1]
    float arg = 1.0f - r * r;                               // 1 - (2 dist / W)^2
    if (arg < 0.0f) arg = 0.0f;                             // guard tiny negatives
    const double num = bessel_i0(static_cast<double>(p.kb_beta) * std::sqrt(static_cast<double>(arg)));
    const double den = bessel_i0(static_cast<double>(p.kb_beta));
    return static_cast<float>(num / den);
}

// ---------------------------------------------------------------------------
// kb_deapod_1d: the 1-D DEAPODIZATION factor for grid index `i` along one axis.
//   Convolving with the KB kernel in k-space MULTIPLIES the image by the kernel's
//   Fourier transform. To undo that "apodization", after the inverse FFT we DIVIDE
//   each pixel by that transform. The KB window has a closed-form FT (a sinc-like
//   function), so we compute the correction analytically instead of numerically:
//
//     a = (pi * W * x / n)^2 - beta^2 ,  x = i - n/2   (centered image coordinate)
//     FT(x) = sin(sqrt(a)) / sqrt(a)        if a > 0
//           = sinh(sqrt(-a)) / sqrt(-a)     if a < 0   (the analytic continuation)
//
//   The 2-D deapodization is the OUTER PRODUCT of two 1-D factors (the kernel is
//   separable), so the caller multiplies deapod(row) * deapod(col). THEORY "the
//   math" derives this formula. Returns the 1-D factor; the caller inverts it.
//     * i : grid index along the axis, 0..n-1
//     * p : gridding geometry (kb_w = W, kb_beta = beta, n)
// ---------------------------------------------------------------------------
GRID_HD inline float kb_deapod_1d(int i, const GriddingParams& p) {
    const double PI = 3.14159265358979323846;
    const double x  = static_cast<double>(i) - 0.5 * p.n;    // centered coordinate
    const double w  = static_cast<double>(p.kb_w);
    const double beta = static_cast<double>(p.kb_beta);
    const double tmp = PI * w * x / static_cast<double>(p.n);
    const double a = tmp * tmp - beta * beta;               // argument of the FT
    double val;
    if (a > 1e-12) {
        const double s = std::sqrt(a);
        val = std::sin(s) / s;                              // sinc-like main lobe
    } else if (a < -1e-12) {
        const double s = std::sqrt(-a);
        val = std::sinh(s) / s;                             // hyperbolic branch
    } else {
        val = 1.0;                                          // limit sin(s)/s -> 1
    }
    // The FT can dip near zero at the image edges; clamp its magnitude away from 0
    // so the reciprocal used for deapodization does not explode (a standard guard).
    if (val < 1e-3 && val > -1e-3) val = (val >= 0.0) ? 1e-3 : -1e-3;
    return static_cast<float>(val);
}

// ---------------------------------------------------------------------------
// FIXED-POINT ACCUMULATION  (PATTERNS.md section 3 -- deterministic atomics)
//   The GPU gridder has MANY threads add contributions into the SAME grid cell
//   (a scatter). Floating-point atomicAdd is NOT associative -- the sum depends on
//   the (nondeterministic) order threads arrive, so a float grid would differ run to
//   run AND differ from the ordered CPU loop. The fix: accumulate in INTEGERS.
//   Integer addition commutes, so an integer atomicAdd is order-independent and
//   therefore both DETERMINISTIC and bit-identical to the CPU. We quantize each
//   real contribution to a fixed-point integer (multiply by GRID_FIXED_SCALE and
//   round), sum the integers, then convert back to float once at the end.
//
//   Both the CPU reference AND the GPU kernel use THESE functions, so they perform
//   the exact same quantization -- making the cross-check EXACT (tolerance 0).
//   The scale is large enough that quantization is far below FP32 image noise but
//   small enough that the summed magnitudes never overflow a 64-bit integer for our
//   tiny grids. See THEORY "numerical considerations".
// ---------------------------------------------------------------------------
// GRID_FIXED_SCALE: contributions are multiplied by this and rounded to a 64-bit
//   integer before accumulation. 2^20 ~= 1e6 gives ~6 decimal digits of fixed-point
//   resolution -- finer than FP32's ~7 significant digits at these magnitudes.
#define GRID_FIXED_SCALE 1048576.0   // 2^20

// to_fixed: quantize a real contribution to a signed 64-bit fixed-point integer.
//   Uses round-half-away-from-zero (the +/-0.5 then truncate idiom) so host and
//   device round identically. long long is the same width on both.
GRID_HD inline long long to_fixed(float x) {
    const double v = static_cast<double>(x) * GRID_FIXED_SCALE;
    return (v >= 0.0) ? static_cast<long long>(v + 0.5)
                      : static_cast<long long>(v - 0.5);
}

// from_fixed: convert an accumulated fixed-point integer back to float.
GRID_HD inline float from_fixed(long long q) {
    return static_cast<float>(static_cast<double>(q) / GRID_FIXED_SCALE);
}

// ---------------------------------------------------------------------------
// radial_dcf: the density-compensation factor (DCF) for a radial readout sample at
//   integer readout offset `ro` from the spoke center (ro = j - n_ro/2).
//   WHY: radial spokes all pass through the k-space CENTER, so low frequencies are
//   massively over-sampled and high frequencies under-sampled. If we grid the raw
//   samples the center dominates and the image is a blurry blob. The DCF re-weights
//   each sample by the "area" of k-space it represents. For radial trajectories the
//   analytic DCF is proportional to |k| -- the ramp filter, exactly as in filtered
//   backprojection (project 4.01!). We use w = |ro| with a small floor at the center
//   so the DC sample is not entirely thrown away. THEORY "the algorithm" derives the
//   |k| weight from the radial-to-Cartesian Jacobian.
//     * ro : signed readout offset from the spoke center, in grid cells
//   Returns the (unnormalized) density-compensation weight, a pure function of |ro|.
// ---------------------------------------------------------------------------
GRID_HD inline float radial_dcf(float ro) {
    const float a = ro < 0.0f ? -ro : ro;   // |ro| -- the ramp |k| weight
    // Floor of 0.5 cell at the very center so the DC term keeps a small weight
    // rather than vanishing (a standard, deterministic regularization of the ramp).
    return a < 0.5f ? 0.5f : a;
}

// ---------------------------------------------------------------------------
// golden_angle_rad: the azimuthal angle (radians) of radial spoke index `s`.
//   Real-time radial MRI acquires spokes at the GOLDEN ANGLE 111.25 degrees apart:
//   consecutive spokes are maximally spread, so ANY contiguous window of spokes
//   tiles k-space near-uniformly -- exactly what lets a SLIDING WINDOW form a fresh
//   image at every time step. theta_s = s * 111.25 deg (mod 2 pi). THEORY "the
//   science" explains the golden-ratio spacing.
// ---------------------------------------------------------------------------
GRID_HD inline double golden_angle_rad(int s) {
    const double PI = 3.14159265358979323846;
    const double golden_deg = 111.2461179749811;    // 180 * (1 - 1/golden_ratio)
    return static_cast<double>(s) * golden_deg * PI / 180.0;
}
