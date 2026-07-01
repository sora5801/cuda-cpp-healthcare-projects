// ===========================================================================
// src/combat.h  --  Shared (host + device) ComBat harmonization primitives
// ---------------------------------------------------------------------------
// Project 4.25 : Image Harmonization Across Scanners/Sites
//
// WHAT THIS PROJECT COMPUTES  (reduced-scope teaching version -- see THEORY.md)
//   Multi-site imaging studies pool subjects scanned on DIFFERENT machines
//   (vendors, field strengths, protocols). Each scanner stamps a systematic
//   "batch effect" onto every extracted image FEATURE (e.g. a regional cortical
//   thickness, a volume, a texture statistic). That scanner signature is
//   confounding: it looks like biology but is really hardware. ComBat is the
//   field-standard *statistical* harmonizer (NeuroComBat) that removes the
//   per-batch location (mean) and scale (variance) shift from each feature while
//   PRESERVING wanted biological covariates (age, sex, diagnosis).
//
//   (The catalog also lists image-level deep harmonizers -- CycleGAN, CALAMITI.
//    Training those is ~100 GPU-hours and out of scope for a study project; we
//    ship ComBat, the statistically-tractable core, and describe the deep
//    methods in THEORY.md "Where this sits in the real world". CLAUDE.md §13.)
//
// THE DATA LAYOUT
//   N samples (subjects/images), P features per sample, B batches (scanners),
//   and C biological covariates to preserve (an intercept column is added, so
//   the design matrix has M = C + B columns: covariates THEN batch indicators).
//   Feature matrix Y is [P x N] row-major: Y[p*N + n] is feature p of sample n.
//   We keep features in ROWS because ComBat processes each feature INDEPENDENTLY
//   -> one GPU thread owns one whole feature row (see kernels.cuh).
//
// WHY A GPU
//   Real feature sets are large: a voxel-wise or vertex-wise harmonization has
//   P ~ 10^5..10^6 features, each needing its own little regression + empirical-
//   Bayes fit. Every feature is INDEPENDENT, so this is an "ensemble of tiny
//   solves": thread p does the entire ComBat math for feature p in registers.
//   This is the same pattern as the ODE-ensemble flagships (9.02, 13.02): the
//   per-item numerics live in ONE __host__ __device__ core so the CPU reference
//   and the GPU kernel run byte-for-byte identical math (PATTERNS.md §2).
//
// DETERMINISM
//   No atomics and no cross-thread reductions in the GPU path: each feature is
//   self-contained, and its arithmetic is a fixed sequence of double-precision
//   operations executed in the SAME ORDER on host and device. The only source of
//   host/device divergence is fused-multiply-add (FMA) contraction, which we
//   discuss in THEORY.md §Numerics; it stays far below our reported precision.
//
//   These helpers are __host__ __device__ (CB_HD) so both sides share the math.
// ===========================================================================
#pragma once

#include <cstddef>
#include <cmath>     // std::sqrt -- host needs this; nvcc device code also honors it

// The HD-macro idiom (PATTERNS.md §2): under nvcc (__CUDACC__) the functions are
// compiled for BOTH the host and the device; under the plain C++ compiler the
// decorators simply vanish, so reference_cpu.cpp can include this same header.
#ifdef __CUDACC__
#define CB_HD __host__ __device__
#else
#define CB_HD
#endif

// Upper bound on the design-matrix width M = C + B (covariates + batches). ComBat
// design matrices are TINY (a handful of covariates, a handful of scanners), so a
// small fixed cap lets every thread keep its per-feature normal-equations system
// in registers/local arrays instead of touching global memory. If a real study
// needs more, raise this one constant (and note the register-pressure tradeoff).
#ifndef CB_MAX_M
#define CB_MAX_M 16
#endif

// ---------------------------------------------------------------------------
// cb_solve_normal_equations
//   Solve the small symmetric positive-definite linear system  A x = b  where
//   A is M x M (row-major, M <= CB_MAX_M) and b is length M, IN PLACE, via
//   Gauss-Jordan elimination with partial pivoting. Returns the solution in x.
//
//   WHY THIS EXISTS: ComBat's step-1 model fit is an ordinary-least-squares
//   regression  beta = (X^T X)^{-1} (X^T y)  for EACH feature. X^T X is the M x M
//   matrix A (same for a given design but we recompute per feature to keep each
//   thread independent), and X^T y is b. M is tiny (<= CB_MAX_M), so a direct
//   dense solve in registers is faster and simpler than calling a batched library
//   solver -- and it keeps the code a readable white box (THEORY.md §GPU mapping
//   explains the cuBLAS/cuSOLVER alternative for very wide designs).
//
//   Params:
//     a  [M*M] in/out : the normal-matrix A; destroyed (reduced to identity).
//     b  [M]   in/out : the right-hand side X^T y; destroyed.
//     x  [M]   out    : the solution beta.
//     M        in     : system size (1..CB_MAX_M).
//   Complexity: O(M^3), trivial for M<=16. No allocation, no recursion.
// ---------------------------------------------------------------------------
CB_HD inline void cb_solve_normal_equations(double* a, double* b, double* x, int M) {
    // Forward elimination with partial pivoting: for each column c, pick the row
    // (at or below c) with the largest |pivot| for numerical stability, swap it
    // up, then eliminate column c from every other row.
    for (int c = 0; c < M; ++c) {
        // --- find the pivot row (largest magnitude in column c) ---
        int piv = c;
        double best = a[c * M + c];
        if (best < 0.0) best = -best;
        for (int r = c + 1; r < M; ++r) {
            double v = a[r * M + c];
            if (v < 0.0) v = -v;
            if (v > best) { best = v; piv = r; }
        }
        // Swap pivot row into place (both A and b) if needed.
        if (piv != c) {
            for (int k = 0; k < M; ++k) {
                double t = a[c * M + k]; a[c * M + k] = a[piv * M + k]; a[piv * M + k] = t;
            }
            double tb = b[c]; b[c] = b[piv]; b[piv] = tb;
        }
        // Normalize the pivot row so A[c][c] becomes 1.
        double diag = a[c * M + c];
        // Guard a (near-)singular design: if the pivot is ~0 the columns are
        // collinear (e.g. a redundant batch indicator). We fall back to leaving
        // the row as-is; THEORY.md §Numerics notes the caller should use a
        // full-rank, reference-coded design so this never triggers in practice.
        if (diag == 0.0) diag = 1.0;
        double inv = 1.0 / diag;
        for (int k = 0; k < M; ++k) a[c * M + k] *= inv;
        b[c] *= inv;
        // Eliminate column c from all OTHER rows.
        for (int r = 0; r < M; ++r) {
            if (r == c) continue;
            double f = a[r * M + c];
            if (f == 0.0) continue;
            for (int k = 0; k < M; ++k) a[r * M + k] -= f * a[c * M + k];
            b[r] -= f * b[c];
        }
    }
    // After full Gauss-Jordan, A is the identity and b holds the solution.
    for (int i = 0; i < M; ++i) x[i] = b[i];
}

// ---------------------------------------------------------------------------
// cb_harmonize_feature
//   The ENTIRE ComBat computation for ONE feature (one row of Y). This is the
//   "per-item physics" shared by the CPU reference and the GPU kernel -- calling
//   it from a serial host loop or from one GPU thread yields identical output.
//
//   The algorithm (each step cross-referenced to THEORY.md §Algorithm):
//     (1) FIT a linear model  y_n = X_n . beta + eps_n  by OLS over all N
//         samples. X_n is this sample's design row (covariates + batch dummies).
//     (2) POOLED VARIANCE: sigma^2 = mean over n of (y_n - X_n.beta)^2. This is
//         the residual variance after removing covariate + batch means.
//     (3) STANDARDIZE: z_n = (y_n - covariate_fit_n) / sigma, where the
//         covariate_fit uses only the COVARIATE part of beta (we deliberately
//         keep the covariate/biological signal and strip only batch).
//     (4) BATCH L/S ESTIMATES: for each batch, gamma_hat = mean of z over that
//         batch's samples (the location shift) and delta_hat = variance of z in
//         that batch (the scale shift).
//     (5) EMPIRICAL-BAYES SHRINKAGE (the heart of ComBat): pull each batch's raw
//         gamma_hat/delta_hat toward the ACROSS-FEATURE prior so small batches
//         borrow strength. Here, because we harmonize features independently and
//         the priors are supplied per-batch by the caller, we apply the closed-
//         form parametric shrinkage using those priors (see THEORY.md §EB).
//     (6) ADJUST: z*_n = (z_n - gamma*) / sqrt(delta*)   then map back to data
//         scale: y*_n = z*_n * sigma + covariate_fit_n. This is the harmonized
//         value: the batch's mean/variance signature is removed, the biological
//         covariate signal and the grand mean are restored.
//
//   DESIGN LAYOUT & FULL RANK (important -- see THEORY.md §Numerics):
//     The design row is  [ C covariate columns | B batch-indicator columns ]  with
//     NO separate intercept column. The B batch dummies collectively span the
//     intercept (their row-sum is 1 for every sample), so ADDING an intercept
//     would make X rank-deficient and the normal-equations solve ill-posed (its
//     answer would then depend on floating-point rounding order -> host/device and
//     compiler-to-compiler disagreement). Omitting the intercept keeps X FULL RANK
//     and the solve well-conditioned and reproducible. The per-feature GRAND MEAN
//     is recovered as the batch-size-weighted average of the batch coefficients,
//     alpha = sum_b (n_b/N) * beta_batch[b]  (this is NeuroComBat's `stand_mean`).
//
//   Params (all host or device pointers; feature-local, no shared state):
//     y        [N]      in  : the raw feature values for the N samples.
//     design   [N*M]    in  : row-major design matrix X (shared across features).
//     batch_of [N]      in  : batch index (0..B-1) of each sample.
//     N,M,C,B          in  : sizes (M = C + B; C covariate cols, then B batch cols).
//     batch_n  [B]      in  : number of samples in each batch (for the L/S means).
//     gamma_bar[B]      in  : EB prior MEAN of gamma per batch (across features).
//     tau2     [B]      in  : EB prior VARIANCE of gamma per batch.
//     a_prior  [B]      in  : EB inverse-gamma shape for delta per batch.
//     b_prior  [B]      in  : EB inverse-gamma scale for delta per batch.
//     out      [N]      out : the harmonized feature values.
//   Returns: nothing; writes `out`. O(N*M + M^3) time, O(M^2) local scratch.
// ---------------------------------------------------------------------------
CB_HD inline void cb_harmonize_feature(
        const double* y, const double* design, const int* batch_of,
        int N, int M, int C, int B,
        const int* batch_n,
        const double* gamma_bar, const double* tau2,
        const double* a_prior, const double* b_prior,
        double* out) {

    // ---- (1) Build and solve the OLS normal equations  (X^T X) beta = X^T y ---
    // Local scratch sized by the compile-time cap CB_MAX_M so it lives in
    // registers/local memory -- no global allocation, so every thread is
    // independent (the ensemble pattern).
    double AtA[CB_MAX_M * CB_MAX_M];   // X^T X, symmetric M x M
    double Aty[CB_MAX_M];              // X^T y, length M
    double beta[CB_MAX_M];             // the fitted coefficients
    for (int i = 0; i < M * M; ++i) AtA[i] = 0.0;
    for (int i = 0; i < M; ++i) Aty[i] = 0.0;
    // Accumulate X^T X and X^T y in one pass over the N samples. Each sample
    // contributes the outer product of its design row plus that row times y_n.
    for (int n = 0; n < N; ++n) {
        const double* xn = design + (std::size_t)n * M;   // this sample's design row
        const double yn = y[n];
        for (int i = 0; i < M; ++i) {
            Aty[i] += xn[i] * yn;
            for (int j = 0; j < M; ++j) AtA[i * M + j] += xn[i] * xn[j];
        }
    }
    cb_solve_normal_equations(AtA, Aty, beta, M);   // beta <- (X^T X)^{-1} X^T y

    // The GRAND MEAN alpha is the batch-size-weighted average of the batch
    // coefficients (columns C..C+B-1 of beta). Because there is no intercept, the
    // batch coefficients ARE the per-batch means of the (covariate-adjusted)
    // feature; pooling them by batch size gives the overall feature mean we want
    // to preserve. This is NeuroComBat's `stand_mean`.
    double alpha = 0.0;
    for (int b = 0; b < B; ++b) alpha += (double)batch_n[b] * beta[C + b];
    alpha /= (double)N;

    // ---- (2) Pooled residual standard deviation sigma -------------------------
    // sigma^2 = (1/N) * sum_n (y_n - X_n.beta)^2. This is the spread left after
    // the full model (covariates + batch) is subtracted -- the natural scale to
    // standardize by so batches become comparable.
    double ss = 0.0;
    for (int n = 0; n < N; ++n) {
        const double* xn = design + (std::size_t)n * M;
        double fit = 0.0;
        for (int i = 0; i < M; ++i) fit += xn[i] * beta[i];
        double r = y[n] - fit;
        ss += r * r;
    }
    double sigma2 = ss / (double)N;
    if (sigma2 <= 0.0) sigma2 = 1e-12;          // guard a constant feature
    double sigma = std::sqrt(sigma2);

    // ---- (3) Standardize about (grand mean + covariate fit), NOT batch ---------
    // stand_fit_n = alpha + sum over the C COVARIATE columns of X_n * beta. We keep
    // the biology (covariate columns 0..C-1) and the grand mean, and will remove
    // only the batch location/scale. z_n = (y_n - stand_fit_n) / sigma.
    // (We store z back into `out` as temporary scratch; it is overwritten in (6).)
    for (int n = 0; n < N; ++n) {
        const double* xn = design + (std::size_t)n * M;
        double cfit = alpha;
        for (int i = 0; i < C; ++i) cfit += xn[i] * beta[i];   // grand mean + covariate fit
        out[n] = (y[n] - cfit) / sigma;                        // standardized z_n
    }

    // ---- (4) Raw per-batch location (gamma_hat) and scale (delta_hat) ---------
    // For each batch b: gamma_hat[b] = mean of z over batch b's samples; this is
    // exactly the batch's average leftover offset (its location signature). We
    // loop batches x samples; B is tiny. delta_hat[b] = mean squared deviation of
    // z from gamma_hat within the batch (its scale signature).
    double gamma_hat[CB_MAX_M];   // reuse the CB_MAX_M cap: B <= M <= CB_MAX_M
    double delta_hat[CB_MAX_M];
    for (int b = 0; b < B; ++b) {
        double sum = 0.0;
        for (int n = 0; n < N; ++n) if (batch_of[n] == b) sum += out[n];
        double cnt = (double)batch_n[b];
        gamma_hat[b] = (cnt > 0.0) ? (sum / cnt) : 0.0;
    }
    for (int b = 0; b < B; ++b) {
        double ss2 = 0.0;
        for (int n = 0; n < N; ++n) if (batch_of[n] == b) {
            double d = out[n] - gamma_hat[b];
            ss2 += d * d;
        }
        double cnt = (double)batch_n[b];
        delta_hat[b] = (cnt > 1.0) ? (ss2 / (cnt - 1.0)) : 1.0;  // unbiased-ish
        if (delta_hat[b] <= 0.0) delta_hat[b] = 1e-12;
    }

    // ---- (5) Empirical-Bayes shrinkage toward the across-feature priors -------
    // Location: the parametric EB posterior mean of gamma is a precision-weighted
    // average of the raw estimate and the prior mean gamma_bar:
    //   gamma_star = (n_b * tau2_b * gamma_hat + delta_star * gamma_bar_b)
    //               / (n_b * tau2_b + delta_star)
    // Small batches (small n_b) get pulled harder toward the prior -> they
    // "borrow strength" from the whole feature panel. This is why ComBat is more
    // robust than a plain per-batch z-score. THEORY.md §EB derives this.
    // Scale: the inverse-gamma posterior mean of delta:
    //   delta_star = (b_prior + 0.5 * sum_n_in_b (z - gamma_star)^2)
    //               / (n_b/2 + a_prior - 1)
    for (int b = 0; b < B; ++b) {
        double nb = (double)batch_n[b];
        // Location posterior (uses the CURRENT delta_hat as the likelihood scale).
        double denom = nb * tau2[b] + delta_hat[b];
        double gamma_star = (denom > 0.0)
            ? (nb * tau2[b] * gamma_hat[b] + delta_hat[b] * gamma_bar[b]) / denom
            : gamma_hat[b];
        // Scale posterior: recompute the within-batch sum of squares about the
        // shrunk location gamma_star.
        double sos = 0.0;
        for (int n = 0; n < N; ++n) if (batch_of[n] == b) {
            double d = out[n] - gamma_star;
            sos += d * d;
        }
        double ddenom = nb * 0.5 + a_prior[b] - 1.0;
        double delta_star = (ddenom > 0.0) ? (b_prior[b] + 0.5 * sos) / ddenom : delta_hat[b];
        if (delta_star <= 0.0) delta_star = 1e-12;
        // Stash the shrunk parameters back into the per-batch arrays for step (6).
        gamma_hat[b] = gamma_star;
        delta_hat[b] = delta_star;
    }

    // ---- (6) Apply the correction and map back to the data scale --------------
    // z*_n = (z_n - gamma*_b) / sqrt(delta*_b)      removes the batch L/S signature;
    // y*_n = z*_n * sigma + (alpha + covariate_fit)  restores grand mean + biology.
    for (int n = 0; n < N; ++n) {
        const double* xn = design + (std::size_t)n * M;
        int b = batch_of[n];
        double zc = (out[n] - gamma_hat[b]) / std::sqrt(delta_hat[b]);   // batch-free z
        double cfit = alpha;
        for (int i = 0; i < C; ++i) cfit += xn[i] * beta[i];        // grand mean + covariate fit
        out[n] = zc * sigma + cfit;                                 // harmonized value
    }
}
