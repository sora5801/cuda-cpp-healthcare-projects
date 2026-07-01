// ===========================================================================
// src/glm.h  --  Shared (host + device) fMRI GLM core: HRF, design matrix, OLS
// ---------------------------------------------------------------------------
// Project 4.16 : Functional MRI Analysis
//
// WHAT THIS PROJECT COMPUTES
//   Task-based functional MRI (fMRI) records a 4-D movie of the brain: at each of
//   V voxels we have a BOLD (blood-oxygen-level-dependent) time-series of T scans.
//   The classic "activation map" question is: *which voxels responded to the
//   task?* The standard answer (SPM, FSL FEAT) is the mass-univariate GENERAL
//   LINEAR MODEL (GLM): fit the SAME linear model
//
//        y_v  =  X * beta_v  +  e_v          (one ordinary-least-squares fit)
//
//   independently at every voxel v, then test a contrast of the fitted weights
//   with a t-statistic. y_v is the voxel's length-T time-series; X is a T x K
//   DESIGN MATRIX shared by all voxels; beta_v are the K fitted weights; e_v is
//   the residual. Here K = 3 columns:
//        col 0 : the TASK regressor  = task on/off boxcar convolved with the
//                canonical hemodynamic response function (HRF)  -> "activation"
//        col 1 : a linear DRIFT term (models slow scanner signal drift)
//        col 2 : an INTERCEPT (the baseline/mean signal)
//
//   Because every voxel is an INDEPENDENT least-squares fit against the SAME X,
//   this is the "many identical small solves" pattern (docs/PATTERNS.md §1, like
//   the 9.02 SEIR ensemble): give each voxel its own GPU thread. For real data
//   (V ~ 10^5 voxels x T ~ 10^3 scans) that is a huge, embarrassingly parallel
//   workload -- exactly what a GPU is for.
//
//   THE HD-CORE IDIOM (docs/PATTERNS.md §2). Every per-voxel math routine below
//   is a __host__ __device__ inline function, so the CPU reference (reference_cpu
//   .cpp, compiled by cl.exe) and the GPU kernel (kernels.cu, compiled by nvcc)
//   run BYTE-FOR-BYTE identical arithmetic. Verification is then near-exact
//   (double precision, ~1e-9), not a hand-wave. FMRI_HD expands to
//   __host__ __device__ under nvcc and to nothing under the host compiler.
//
//   All math here is FP64 (double) on purpose: a t-statistic divides by a
//   residual standard error that can be small, so we want the head-room.
//
// READ THIS AFTER: nothing -- start here, it defines the science. Then
// reference_cpu.h (the loop that calls these), then kernels.cuh / kernels.cu.
// The full derivation is in ../THEORY.md.
// ===========================================================================
#pragma once

// FMRI_HD: mark a function callable from BOTH host and device under nvcc, while
// staying legal plain C++ when this header is seen by the host compiler alone.
#ifdef __CUDACC__
#define FMRI_HD __host__ __device__
#else
#define FMRI_HD
#endif

// K = number of design-matrix columns (regressors) = TASK, DRIFT, INTERCEPT.
// Fixed at 3 so the normal-equations matrix is a tiny 3x3 we can solve in
// registers inside a single thread (no per-voxel cuSOLVER call needed).
#define FMRI_K 3

// ---------------------------------------------------------------------------
// canonical_hrf(t): the canonical hemodynamic response function, sampled at
//   time t seconds after a brief neural event. The BOLD signal does not track
//   neural activity instantaneously; it rises, peaks ~5-6 s later, then dips
//   below baseline (the "post-stimulus undershoot") before recovering. SPM's
//   canonical HRF models this as a DIFFERENCE OF TWO GAMMA densities:
//
//        h(t) = g(t; a1,b) - c * g(t; a2,b),   g(t;a,b) = b^a t^(a-1) e^(-b t)/Gamma(a)
//
//   with the standard SPM parameters (peak ~5 s, undershoot ~15 s):
//        a1 = 6, a2 = 16, b = 1 (dispersion), c = 1/6 (undershoot ratio).
//
//   We compute the gamma density in LOG space (lgamma + exp) for numerical
//   stability -- t^(a-1) and Gamma(a) individually overflow/underflow, but
//   their logs are tame. Returns 0 for t <= 0 (no response before the event).
//
//   Units: t in seconds, return value is dimensionless (peak ~0.11 before the
//   regressor is later rescaled). lgamma/exp are in <cmath> (host) and provided
//   as device intrinsics by nvcc, so this is safe in the HD core.
// ---------------------------------------------------------------------------
#include <cmath>   // std::lgamma, std::exp, std::pow (host); device versions via nvcc

FMRI_HD inline double gamma_pdf(double t, double shape, double rate) {
    // Log of  rate^shape * t^(shape-1) * e^(-rate*t) / Gamma(shape).
    if (t <= 0.0) return 0.0;
    const double log_pdf = shape * std::log(rate)
                         + (shape - 1.0) * std::log(t)
                         - rate * t
                         - std::lgamma(shape);
    return std::exp(log_pdf);
}

FMRI_HD inline double canonical_hrf(double t) {
    if (t <= 0.0) return 0.0;
    // SPM canonical HRF constants (seconds / dimensionless).
    const double peak_shape       = 6.0;    // a1: response peaks ~6 s
    const double undershoot_shape = 16.0;   // a2: undershoot centred ~16 s
    const double dispersion       = 1.0;    // b : both gammas share this rate
    const double undershoot_ratio = 1.0 / 6.0;  // c : undershoot is 1/6 of peak
    return gamma_pdf(t, peak_shape, dispersion)
         - undershoot_ratio * gamma_pdf(t, undershoot_shape, dispersion);
}

// ---------------------------------------------------------------------------
// design_column0(...) : value of the TASK regressor at scan index `ti`.
//   The experiment alternates blocks: `block_scans` scans of task ON, then
//   `block_scans` OFF, repeating. That on/off BOXCAR is convolved with the HRF
//   to get the predicted BOLD response. We evaluate the convolution on the fly
//   (a short sum over past scans) so the CPU and GPU build the identical column
//   without storing it -- and so the whole design matrix is reproducible from
//   just (T, TR, block_scans).
//
//   ti          : scan index 0..T-1
//   TR_seconds  : repetition time = seconds between scans
//   block_scans : half-period of the boxcar, in scans
//   Returns the (unnormalised) convolved task regressor at scan ti.
//
//   Convolution: r[ti] = sum_{k=0..ti} boxcar[k] * HRF((ti-k)*TR). Since the HRF
//   decays to ~0 by ~30 s, we could cap the sum, but T is tiny here so we sum
//   the full history for simplicity and exactness.
// ---------------------------------------------------------------------------
FMRI_HD inline int boxcar_on(int scan, int block_scans) {
    // Which half of the current block are we in? (scan / block_scans) even = ON.
    return (((scan / block_scans) & 1) == 0) ? 1 : 0;
}

FMRI_HD inline double design_column0(int ti, double TR_seconds, int block_scans) {
    double acc = 0.0;
    for (int k = 0; k <= ti; ++k) {
        const int on = boxcar_on(k, block_scans);      // task on/off at scan k
        if (on) acc += canonical_hrf((ti - k) * TR_seconds);  // HRF lag (ti-k) scans
    }
    return acc;
}

// design_column1(ti,T): the linear DRIFT regressor, mapped to [-1, +1] across
//   the run so it is mean-near-zero and well-scaled (a raw 0..T ramp would be
//   huge and correlate with the intercept). ti in 0..T-1.
FMRI_HD inline double design_column1(int ti, int T) {
    return (T > 1) ? (2.0 * ti / (T - 1) - 1.0) : 0.0;
}

// design_column2(): the INTERCEPT regressor is a constant 1 (models the baseline
//   mean). A separate function purely for symmetry/readability with the others.
FMRI_HD inline double design_column2() { return 1.0; }

// design_value(col, ti, T, TR, block): dispatch to the right column builder so
//   both the CPU loop and the GPU kernel produce X identically from one place.
FMRI_HD inline double design_value(int col, int ti, int T, double TR_seconds, int block_scans) {
    if (col == 0) return design_column0(ti, TR_seconds, block_scans);
    if (col == 1) return design_column1(ti, T);
    return design_column2();
}

// ---------------------------------------------------------------------------
// GlmDesign: the run-level parameters that define X. Shared by CPU and GPU; the
//   GPU copies this tiny struct plus the precomputed 3x3 (X^T X)^-1 into the
//   kernel (both are voxel-independent -- computed once, reused for all V).
// ---------------------------------------------------------------------------
struct GlmDesign {
    int    T = 0;             // number of scans (time points)
    double TR_seconds = 0.0;  // repetition time (seconds per scan)
    int    block_scans = 0;   // task boxcar half-period, in scans
};

// ---------------------------------------------------------------------------
// solve_sym3(A, b, x): solve the 3x3 SYMMETRIC POSITIVE-DEFINITE system A x = b
//   for x, by an explicit closed-form inverse via cofactors. A is the normal-
//   equations matrix X^T X (symmetric PD when X has full column rank), b is
//   X^T y. Because K=3 is fixed and tiny, a hand-rolled inverse is faster and
//   more register-friendly than a general solver, and -- crucially -- runs the
//   SAME operations on host and device so results match bit-for-bit.
//
//   A is passed as its 6 unique entries (row-major upper triangle):
//     a00 a01 a02
//         a11 a12
//             a22
//   Returns the determinant (so callers can detect a singular/rank-deficient
//   design). Also writes the full inverse into `inv` (row-major 3x3) because the
//   t-statistic needs (X^T X)^-1_00 for the contrast variance.
// ---------------------------------------------------------------------------
FMRI_HD inline double invert_sym3(double a00, double a01, double a02,
                                  double a11, double a12, double a22,
                                  double inv[9]) {
    // Cofactors of the symmetric matrix.
    const double c00 =  (a11 * a22 - a12 * a12);
    const double c01 = -(a01 * a22 - a12 * a02);
    const double c02 =  (a01 * a12 - a11 * a02);
    const double c11 =  (a00 * a22 - a02 * a02);
    const double c12 = -(a00 * a12 - a01 * a02);
    const double c22 =  (a00 * a11 - a01 * a01);
    const double det = a00 * c00 + a01 * c01 + a02 * c02;   // expand along row 0
    if (det == 0.0) return 0.0;                             // singular -> caller guards
    const double invdet = 1.0 / det;
    // inverse = cofactor^T / det; symmetric so cofactor^T = cofactor here.
    inv[0] = c00 * invdet; inv[1] = c01 * invdet; inv[2] = c02 * invdet;
    inv[3] = c01 * invdet; inv[4] = c11 * invdet; inv[5] = c12 * invdet;
    inv[6] = c02 * invdet; inv[7] = c12 * invdet; inv[8] = c22 * invdet;
    return det;
}

// ---------------------------------------------------------------------------
// VoxelStat: what the per-voxel GLM fit reports. The headline number is `tstat`
//   -- the t-statistic for the TASK contrast c=[1,0,0], i.e. "is beta_task
//   significantly non-zero?" Larger |t| = stronger, more reliable activation.
// ---------------------------------------------------------------------------
struct VoxelStat {
    double beta_task;   // fitted weight of the task regressor (activation size)
    double tstat;       // t-statistic for H0: beta_task == 0
};

// ---------------------------------------------------------------------------
// fit_voxel(...) : the whole per-voxel GLM in one HD function.
//   Inputs:
//     y            : this voxel's time-series, length d.T  (pointer to T doubles)
//     d            : the design parameters (defines X)
//     XtX_inv      : precomputed (X^T X)^-1, row-major 3x3 (voxel-independent)
//   Steps (all standard OLS):
//     1. b = X^T y            (K-vector; only col-0 needs the on-the-fly HRF)
//     2. beta = XtX_inv * b   (the fitted weights)
//     3. residual SS = sum_t (y_t - (X beta)_t)^2
//     4. sigma^2 = RSS / (T - K)                 (unbiased noise variance)
//     5. se(beta_0) = sqrt(sigma^2 * XtX_inv_00) (contrast is c=[1,0,0])
//     6. t = beta_0 / se(beta_0)
//   Returns the VoxelStat. Recomputing the design columns per voxel trades a
//   little arithmetic for zero extra memory traffic -- on the GPU that is the
//   right trade (compute is cheap, bandwidth is precious; see THEORY §GPU).
// ---------------------------------------------------------------------------
FMRI_HD inline VoxelStat fit_voxel(const double* y, const GlmDesign& d,
                                   const double XtX_inv[9]) {
    const int T = d.T;
    // --- step 1: b = X^T y (accumulate all K dot-products in one pass over t) --
    double b0 = 0.0, b1 = 0.0, b2 = 0.0;
    for (int t = 0; t < T; ++t) {
        const double x0 = design_value(0, t, T, d.TR_seconds, d.block_scans);
        const double x1 = design_value(1, t, T, d.TR_seconds, d.block_scans);
        const double x2 = 1.0;                    // intercept
        const double yt = y[t];
        b0 += x0 * yt; b1 += x1 * yt; b2 += x2 * yt;
    }
    // --- step 2: beta = (X^T X)^-1 (X^T y) ------------------------------------
    const double beta0 = XtX_inv[0]*b0 + XtX_inv[1]*b1 + XtX_inv[2]*b2;
    const double beta1 = XtX_inv[3]*b0 + XtX_inv[4]*b1 + XtX_inv[5]*b2;
    const double beta2 = XtX_inv[6]*b0 + XtX_inv[7]*b1 + XtX_inv[8]*b2;
    // --- step 3: residual sum of squares (second pass over t) -----------------
    double rss = 0.0;
    for (int t = 0; t < T; ++t) {
        const double x0 = design_value(0, t, T, d.TR_seconds, d.block_scans);
        const double x1 = design_value(1, t, T, d.TR_seconds, d.block_scans);
        const double fitted = beta0 * x0 + beta1 * x1 + beta2 * 1.0;
        const double r = y[t] - fitted;
        rss += r * r;
    }
    // --- steps 4-6: noise variance, contrast SE, t-statistic ------------------
    const int dof = T - FMRI_K;                   // residual degrees of freedom
    const double sigma2 = (dof > 0) ? rss / dof : 0.0;
    const double var_beta0 = sigma2 * XtX_inv[0]; // Var(beta_0) = sigma^2 (X^TX)^-1_00
    VoxelStat s;
    s.beta_task = beta0;
    s.tstat = (var_beta0 > 0.0) ? beta0 / std::sqrt(var_beta0) : 0.0;
    return s;
}
