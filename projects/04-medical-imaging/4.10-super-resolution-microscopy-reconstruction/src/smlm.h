// ===========================================================================
// src/smlm.h  --  Shared (host + device) single-molecule-localization core
// ---------------------------------------------------------------------------
// Project 4.10 : Super-Resolution Microscopy Reconstruction  (STORM / PALM SMLM)
//
// WHAT THIS PROJECT COMPUTES
//   Single-Molecule Localization Microscopy (SMLM: STORM / PALM) beats the
//   diffraction limit by a trick of time. Instead of imaging all fluorophores at
//   once (which blur together into a diffraction-limited haze ~250 nm wide), it
//   makes only a SPARSE, RANDOM subset blink "on" in each of thousands of camera
//   frames. In a sparse frame the individual blobs are separated, so each one's
//   TRUE centre can be estimated to ~10-20 nm -- far finer than the blob itself.
//   Overlay all those pin-point centres from all frames and a super-resolution
//   image emerges. This project does the compute heart of that pipeline:
//
//     1. DETECT   : find local-maxima pixels above a threshold (candidate spots).
//     2. LOCALIZE : for each candidate, fit the sub-pixel (x,y) centre of its PSF
//                   from the surrounding 7x7 patch  <-- the parallel hot loop.
//     3. RENDER   : accumulate every localization into a finer-grid histogram,
//                   the reconstructed super-resolution image.
//
// WHY A GPU
//   A real STORM run is 10^4-10^5 frames of 256x256-512x512 pixels, each with
//   hundreds of blinking emitters -> tens of millions of independent PSF fits.
//   Every fit reads only its own tiny 7x7 patch and writes only its own (x,y):
//   an EMBARRASSINGLY PARALLEL workload, one thread (or warp) per candidate spot.
//   The render is a scatter into a shared image -> atomicAdd. This is exactly the
//   flagship pattern "independent jobs + atomic reduction" (docs/PATTERNS.md §1).
//
// THE LOCALIZER WE TEACH  (deterministic, CPU==GPU byte-identical)
//   Production SMLM fits a 2D Gaussian by iterative MAXIMUM-LIKELIHOOD (MLE) or
//   least squares (Levenberg-Marquardt). Those are the gold standard but their
//   step counts and floating-point paths differ subtly between a host compiler
//   and nvcc, which muddies a teaching verification. We instead use the classic,
//   robust GAUSSIAN-WEIGHTED CENTROID REFINEMENT (a.k.a. iterative center-of-mass
//   with a Gaussian window) -- the estimator ThunderSTORM offers as its fast
//   method and the historical workhorse of particle tracking:
//
//     start  : the intensity-weighted centroid of the patch (background removed)
//     repeat : re-weight each pixel by a Gaussian centred on the CURRENT estimate,
//              recompute the weighted centroid -> a new, sharper estimate
//     after a FIXED number of iterations, report (x,y), integrated intensity, and
//     an estimated width (second moment).
//
//   Because it is a FIXED iteration count of the SAME double-precision arithmetic
//   in the SAME order on both sides, CPU and GPU agree to ~1e-9 (see THEORY §6).
//   The full MLE fit is described in THEORY §7 "Where this sits in the real world".
//
//   These helpers are __host__ __device__ (SMLM_HD) so the CPU reference and the
//   GPU kernel run byte-for-byte identical math -- the HD-macro idiom of
//   docs/PATTERNS.md §2.
//
// READ THIS AFTER: nothing (start here); then reference_cpu.h, kernels.cuh.
// ===========================================================================
#pragma once

#include <cstdint>   // fixed-width integer types for the fixed-point render

// ---------------------------------------------------------------------------
// HD-macro idiom (docs/PATTERNS.md §2): when this header is compiled by nvcc
// (__CUDACC__ defined) the per-pixel math is marked __host__ __device__ so the
// SAME functions run on the CPU reference and inside the GPU kernel. When the
// plain host compiler (cl.exe / g++) includes it, the decorators vanish. Keep
// CUDA-only constructs (no __global__, no <<<>>>) OUT of this header so the host
// compiler can include it.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define SMLM_HD __host__ __device__
#else
#define SMLM_HD
#endif

// ---------------------------------------------------------------------------
// Fixed geometry constants (compile-time so the fit patch lives in registers).
// ---------------------------------------------------------------------------

// Half-width of the square fitting patch around a candidate pixel. A 7x7 patch
// (radius 3) comfortably contains a diffraction-limited PSF whose Gaussian sigma
// is ~1-1.5 pixels -- big enough to capture the tails, small enough to stay in
// registers and avoid neighbouring emitters. (PATCH = 2*PATCH_R + 1 = 7.)
//
// These are `constexpr`, not plain `static const`: a constexpr is a compile-time
// value the compiler substitutes inline, so it is legal to read from DEVICE code
// (a __host__ __device__ function). A namespace-scope `static const double` would
// be a host-only object and nvcc rejects reading it inside a kernel.
constexpr int PATCH_R = 3;               // patch radius in pixels
constexpr int PATCH   = 2 * PATCH_R + 1; // patch side length = 7
constexpr int PATCH_N = PATCH * PATCH;   // pixels in a patch = 49

// Number of Gaussian-weighted-centroid refinement iterations. Fixed (not a
// convergence test) so the arithmetic is identical every run on CPU and GPU ->
// deterministic. ~5 iterations is plenty for this estimator to settle.
constexpr int FIT_ITERS = 5;

// The Gaussian weighting window's assumed PSF width, in pixels. Real pipelines
// estimate this from a calibration; for a fixed teaching PSF we hard-code the
// value the synthetic data is generated with (scripts/make_synthetic.py).
constexpr double FIT_SIGMA = 1.3;        // pixels

// UPSAMPLE: how many super-resolution pixels per camera pixel along each axis.
// The rendered image is (H*UPSAMPLE) x (W*UPSAMPLE); each localization lands in
// a sub-pixel bin, which is what makes the reconstruction "super"-resolved.
constexpr int UPSAMPLE = 8;

// ---------------------------------------------------------------------------
// Fixed-point render scale (determinism, docs/PATTERNS.md §3).
//   The render is a scatter-add of many localizations' intensities into shared
//   image bins. Float atomicAdd is NON-associative, so the summed pixel value
//   would depend on the (nondeterministic) thread order -> irreproducible and
//   CPU!=GPU. We instead add each localization's intensity as a FIXED-POINT
//   integer (atomicAdd on unsigned long long); integer adds commute, so the
//   image is identical regardless of order and matches the CPU exactly.
//   RENDER_SCALE = 2^16 gives ~5 decimal digits; summed intensities of a few
//   thousand emitters stay far below the 64-bit ceiling.
// ---------------------------------------------------------------------------
constexpr unsigned long long RENDER_SCALE = 1ull << 16;

// Quantize a non-negative intensity to fixed-point for atomic accumulation.
SMLM_HD inline unsigned long long smlm_to_fixed(double intensity) {
    if (intensity < 0.0) intensity = 0.0;          // clamp: emitters are emissive
    return static_cast<unsigned long long>(intensity * static_cast<double>(RENDER_SCALE));
}

// ---------------------------------------------------------------------------
// One localized emitter. The output of LOCALIZE and the input to RENDER.
//   x,y     : sub-pixel centre in CAMERA-pixel coordinates (fractional).
//   photons : integrated intensity above background (a brightness proxy).
//   sigma   : estimated PSF width in pixels (a quality/second-moment readout).
//   frame   : which raw frame this came from (for provenance / filtering).
// A plain POD struct so it copies trivially between host and device.
// ---------------------------------------------------------------------------
struct Localization {
    double x;
    double y;
    double photons;
    double sigma;
    int    frame;
};

// ---------------------------------------------------------------------------
// smlm_is_local_max: is pixel (r,c) a strict local maximum in its 3x3
//   neighbourhood AND above `threshold`?  This is the DETECT test: a candidate
//   emitter is a bright pixel that dominates its immediate neighbours (so we do
//   not fire multiple candidates on the shoulders of one PSF).
//
//   img    : the frame, row-major, H rows x W cols.
//   Border pixels (within 1 of the edge) can't be centred and return false.
//   Ties (>=) are broken by requiring STRICT '>' against earlier-scanned
//   neighbours and '>=' against later ones would be order-dependent; we keep it
//   simple and deterministic with strict '>' everywhere, accepting that an exact
//   plateau (measure-zero on real data) yields no detection.
// ---------------------------------------------------------------------------
SMLM_HD inline bool smlm_is_local_max(const float* img, int H, int W,
                                      int r, int c, double threshold) {
    if (r < 1 || c < 1 || r >= H - 1 || c >= W - 1) return false;
    const double v = static_cast<double>(img[r * W + c]);
    if (v < threshold) return false;
    // Compare against all 8 neighbours; the centre must strictly exceed each.
    for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
            if (dr == 0 && dc == 0) continue;
            const double nv = static_cast<double>(img[(r + dr) * W + (c + dc)]);
            if (v <= nv) return false;   // not a strict maximum -> reject
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// smlm_localize: fit the sub-pixel centre of the emitter whose peak pixel is
//   (r0,c0), using Gaussian-weighted centroid refinement over its 7x7 patch.
//   THIS IS THE HOT FUNCTION: called once per candidate, on CPU (loop) and GPU
//   (one thread). Shared here so both compute bit-identical results.
//
//   Parameters
//     img        : the frame (row-major, H x W), same pointer on host/device.
//     H, W       : frame dimensions in pixels.
//     r0, c0     : integer peak pixel (guaranteed interior: 3 <= r0 < H-3, etc.,
//                  because the caller only fits maxima far enough from the edge).
//     background : per-frame background level subtracted from every pixel (the
//                  camera offset + out-of-focus haze). Non-negative.
//     frame      : frame index, copied into the result for provenance.
//   Returns a fully-populated Localization.
//
//   ALGORITHM (see THEORY §3)
//     est(x,y) <- intensity-weighted centroid of the background-subtracted patch
//     for FIT_ITERS iterations:
//         w_i   = exp(-((xi-x)^2 + (yi-y)^2) / (2 sigma^2)) * I_i    (I_i >= 0)
//         x,y   <- (sum w_i xi)/(sum w_i), (sum w_i yi)/(sum w_i)
//     photons = sum of background-subtracted intensity in the patch
//     sigma   = sqrt of the intensity-weighted second moment (a width readout)
//
//   All accumulation is in `double`; the operations run in the SAME order on CPU
//   and GPU (a simple nested loop, no atomics, no reductions), so the two agree
//   to ~1e-9 (THEORY §6). Complexity: O(FIT_ITERS * PATCH_N) per emitter -- a
//   few hundred flops, entirely in registers.
// ---------------------------------------------------------------------------
SMLM_HD inline Localization smlm_localize(const float* img, int H, int W,
                                          int r0, int c0, double background,
                                          int frame) {
    (void)H;  // H is implied by the caller's bounds guarantee; kept for clarity.

    // --- Pass 0: total background-subtracted intensity + the seed centroid. ---
    // We compute the plain intensity-weighted centroid first; it is the natural
    // starting point and already an unbiased estimate for an isolated symmetric
    // PSF. Coordinates are absolute (camera-pixel) so the result is directly a
    // position in the frame.
    double sum_I  = 0.0;   // sum of I_i  (weights for the seed centroid)
    double sum_Ix = 0.0;   // sum of I_i * x_i
    double sum_Iy = 0.0;   // sum of I_i * y_i
    for (int dr = -PATCH_R; dr <= PATCH_R; ++dr) {
        for (int dc = -PATCH_R; dc <= PATCH_R; ++dc) {
            const int rr = r0 + dr;
            const int cc = c0 + dc;
            // Background-subtracted, clamped at 0 (negative excursions are noise,
            // not photons, and would bias the centroid).
            double I = static_cast<double>(img[rr * W + cc]) - background;
            if (I < 0.0) I = 0.0;
            sum_I  += I;
            sum_Ix += I * static_cast<double>(cc);   // x == column
            sum_Iy += I * static_cast<double>(rr);   // y == row
        }
    }

    Localization loc;
    loc.frame   = frame;
    loc.photons = sum_I;                 // integrated intensity above background
    // Degenerate guard: an all-background patch (sum_I == 0) cannot be localized;
    // fall back to the peak pixel centre so the result stays finite/deterministic.
    if (sum_I <= 0.0) {
        loc.x = static_cast<double>(c0);
        loc.y = static_cast<double>(r0);
        loc.sigma = FIT_SIGMA;
        return loc;
    }
    double x = sum_Ix / sum_I;           // seed x (fractional column)
    double y = sum_Iy / sum_I;           // seed y (fractional row)

    // --- Refinement: Gaussian-weighted centroid, FIT_ITERS fixed passes. ------
    // Re-weighting by a Gaussian centred on the current estimate down-weights the
    // patch corners (which may carry a neighbour's tail or noise) and pulls the
    // estimate toward the true peak. The width `two_s2` is 2*sigma^2 of the
    // weighting window.
    const double two_s2 = 2.0 * FIT_SIGMA * FIT_SIGMA;
    for (int it = 0; it < FIT_ITERS; ++it) {
        double w_sum  = 0.0;   // sum of Gaussian*intensity weights
        double w_x    = 0.0;   // sum of weight * x
        double w_y    = 0.0;   // sum of weight * y
        for (int dr = -PATCH_R; dr <= PATCH_R; ++dr) {
            for (int dc = -PATCH_R; dc <= PATCH_R; ++dc) {
                const int rr = r0 + dr;
                const int cc = c0 + dc;
                double I = static_cast<double>(img[rr * W + cc]) - background;
                if (I < 0.0) I = 0.0;
                // Squared distance from the CURRENT estimate to this pixel.
                const double ex = static_cast<double>(cc) - x;
                const double ey = static_cast<double>(rr) - y;
                // Gaussian window * intensity = the pixel's contribution weight.
                // exp() is deterministic and identical on host/device for the
                // same double argument (IEEE-754), preserving CPU==GPU parity.
                const double g = exp(-(ex * ex + ey * ey) / two_s2);
                const double w = g * I;
                w_sum += w;
                w_x   += w * static_cast<double>(cc);
                w_y   += w * static_cast<double>(rr);
            }
        }
        if (w_sum <= 0.0) break;         // no signal under the window: stop
        x = w_x / w_sum;                 // updated sub-pixel centre
        y = w_y / w_sum;
    }
    loc.x = x;
    loc.y = y;

    // --- Width readout: intensity-weighted second moment about (x,y). ---------
    // Not used for rendering, but a useful per-emitter quality metric (a real
    // pipeline would reject fits whose sigma is far from the calibrated PSF).
    double m0 = 0.0, m2 = 0.0;
    for (int dr = -PATCH_R; dr <= PATCH_R; ++dr) {
        for (int dc = -PATCH_R; dc <= PATCH_R; ++dc) {
            const int rr = r0 + dr;
            const int cc = c0 + dc;
            double I = static_cast<double>(img[rr * W + cc]) - background;
            if (I < 0.0) I = 0.0;
            const double ex = static_cast<double>(cc) - x;
            const double ey = static_cast<double>(rr) - y;
            m0 += I;
            m2 += I * (ex * ex + ey * ey);
        }
    }
    // sigma^2 estimate = (1/2) * <r^2> because <r^2> = 2 sigma^2 for a 2D
    // Gaussian; guard the sqrt against a zero denominator.
    loc.sigma = (m0 > 0.0) ? sqrt(0.5 * m2 / m0) : FIT_SIGMA;
    return loc;
}
