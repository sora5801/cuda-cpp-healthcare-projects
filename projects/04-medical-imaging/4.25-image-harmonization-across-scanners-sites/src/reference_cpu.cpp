// ===========================================================================
// src/reference_cpu.cpp  --  Loader, shared helpers, serial ComBat reference
// ---------------------------------------------------------------------------
// Project 4.25 : Image Harmonization Across Scanners/Sites
//
// Compiled by the host compiler only (cl.exe / g++). The per-feature ComBat math
// lives in combat.h and is shared verbatim with the GPU kernel; here we provide
// (a) the dataset loader, (b) the design-matrix builder and (c) the empirical-
// Bayes prior estimator -- both reused by kernels.cu so CPU and GPU fit the SAME
// model with the SAME priors -- and (d) the trusted serial harmonizer.
// ===========================================================================
#include "reference_cpu.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <istream>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// Small helper: read the next whitespace-separated token as a double, SKIPPING
// any line that begins with '#'. Our synthetic sample carries "# ... synthetic"
// banner lines (data/README.md), and we want the loader to ignore them rather
// than choke. Returns false at end-of-file. We buffer the current line in a
// static-free way by peeking characters; simplicity over speed (tiny files).
// ---------------------------------------------------------------------------
static bool next_token(std::istream& in, std::string& tok) {
    while (true) {
        int c = in.peek();
        if (c == EOF) return false;
        if (c == '#') {                       // comment line: swallow to newline
            std::string discard;
            std::getline(in, discard);
            continue;
        }
        if (std::isspace(c)) { in.get(); continue; }   // skip leading whitespace
        break;
    }
    return static_cast<bool>(in >> tok);
}

// Read the next token and parse it as a double (throws on malformed input).
static bool read_double(std::istream& in, double& v, const std::string& where) {
    std::string t;
    if (!next_token(in, t)) return false;
    try { v = std::stod(t); } catch (...) { throw std::runtime_error("expected a number in " + where); }
    return true;
}
static bool read_int(std::istream& in, int& v, const std::string& where) {
    std::string t;
    if (!next_token(in, t)) return false;
    try { v = std::stoi(t); } catch (...) { throw std::runtime_error("expected an integer in " + where); }
    return true;
}

// ---------------------------------------------------------------------------
// load_dataset: parse the tiny synthetic sample (format in data/README.md).
//   Fails loudly on any malformed header/truncation so a demo never silently
//   runs on garbage.
// ---------------------------------------------------------------------------
Dataset load_dataset(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open dataset file: " + path);
    Dataset d;
    // Header: N P B C  (comment lines beginning with '#' are skipped by the readers).
    if (!read_int(in, d.N, path) || !read_int(in, d.P, path) ||
        !read_int(in, d.B, path) || !read_int(in, d.C, path) ||
        d.N <= 0 || d.P <= 0 || d.B <= 0 || d.C < 0)
        throw std::runtime_error("bad header (expected 'N P B C') in " + path);
    // The design width must fit the register-resident cap in combat.h.
    if (d.M() > CB_MAX_M)
        throw std::runtime_error("design too wide (M > CB_MAX_M); raise CB_MAX_M");

    // Batch labels: one per sample, each in [0, B).
    d.batch.resize(d.N);
    for (int n = 0; n < d.N; ++n) {
        if (!read_int(in, d.batch[n], path) || d.batch[n] < 0 || d.batch[n] >= d.B)
            throw std::runtime_error("bad/out-of-range batch label in " + path);
    }
    // Covariates: N rows of C values (skipped entirely when C == 0).
    d.cov.resize((std::size_t)d.N * d.C);
    for (std::size_t i = 0; i < d.cov.size(); ++i)
        if (!read_double(in, d.cov[i], path)) throw std::runtime_error("covariates truncated in " + path);
    // Feature matrix: P rows, each with N values (row = one feature across samples).
    d.Y.resize((std::size_t)d.P * d.N);
    for (std::size_t i = 0; i < d.Y.size(); ++i)
        if (!read_double(in, d.Y[i], path)) throw std::runtime_error("feature matrix truncated in " + path);
    return d;
}

// ---------------------------------------------------------------------------
// build_design: assemble the [N x M] row-major design matrix X.
//   [ C covariates | B batch indicators ]   (NO intercept -- see below).
//   The covariate block (first C columns) is the biology we PRESERVE; the batch
//   block (last B columns) is what ComBat removes. We use full-dummy batch coding
//   (one indicator per batch, no dropped reference) but OMIT a separate intercept
//   column, because the B batch dummies already sum to 1 for every sample and so
//   span the intercept. Including both would make X^T X singular and the OLS solve
//   ill-posed (its result would depend on rounding order, breaking CPU==GPU
//   reproducibility). The grand mean is instead recovered from the batch
//   coefficients inside cb_harmonize_feature. See THEORY.md §Numerics.
// ---------------------------------------------------------------------------
void build_design(const Dataset& d, std::vector<double>& design) {
    const int N = d.N, C = d.C, M = d.M();
    design.assign((std::size_t)N * M, 0.0);
    for (int n = 0; n < N; ++n) {
        double* row = &design[(std::size_t)n * M];
        for (int c = 0; c < C; ++c)                     // covariate columns 0..C-1
            row[c] = d.cov[(std::size_t)n * C + c];
        row[C + d.batch[n]] = 1.0;                      // this sample's batch dummy
    }
}

// ---------------------------------------------------------------------------
// Internal helper: given the full data, compute each feature's raw per-batch
// location gamma_hat and scale delta_hat (the same quantities cb_harmonize_feature
// computes internally, but here we need them across ALL features to fit the EB
// priors). We factor the standardization out so the prior fit and the per-feature
// harmonization use identical definitions.
//
// For feature p and batch b:
//   gamma_hat[p][b] = mean over batch-b samples of z, where
//   z_n = (y_n - covariate_fit_n) / sigma_p, and sigma_p is the pooled residual SD.
//   delta_hat[p][b] = within-batch variance of z.
// ---------------------------------------------------------------------------
static void raw_batch_stats(const Dataset& d, const std::vector<double>& design,
                            const std::vector<int>& batch_n,
                            std::vector<double>& gamma_hat,  // [P*B]
                            std::vector<double>& delta_hat)  // [P*B]
{
    const int N = d.N, P = d.P, B = d.B, C = d.C, M = d.M();
    gamma_hat.assign((std::size_t)P * B, 0.0);
    delta_hat.assign((std::size_t)P * B, 0.0);

    // Reusable per-feature scratch for the OLS fit (mirrors combat.h step 1-3).
    std::vector<double> AtA((std::size_t)M * M), Aty(M), beta(M);
    for (int p = 0; p < P; ++p) {
        const double* y = &d.Y[(std::size_t)p * N];
        // Fit beta = (X^T X)^{-1} X^T y.
        std::fill(AtA.begin(), AtA.end(), 0.0);
        std::fill(Aty.begin(), Aty.end(), 0.0);
        for (int n = 0; n < N; ++n) {
            const double* xn = &design[(std::size_t)n * M];
            for (int i = 0; i < M; ++i) {
                Aty[i] += xn[i] * y[n];
                for (int j = 0; j < M; ++j) AtA[(std::size_t)i * M + j] += xn[i] * xn[j];
            }
        }
        cb_solve_normal_equations(AtA.data(), Aty.data(), beta.data(), M);
        // Grand mean = batch-size-weighted average of the batch coefficients
        // (identical definition to cb_harmonize_feature; no intercept column).
        double alpha = 0.0;
        for (int b = 0; b < B; ++b) alpha += (double)batch_n[b] * beta[C + b];
        alpha /= (double)N;
        // Pooled residual SD about the FULL model.
        double ss = 0.0;
        for (int n = 0; n < N; ++n) {
            const double* xn = &design[(std::size_t)n * M];
            double fit = 0.0;
            for (int i = 0; i < M; ++i) fit += xn[i] * beta[i];
            double r = y[n] - fit; ss += r * r;
        }
        double sigma = std::sqrt(std::max(ss / (double)N, 1e-12));
        // Standardize about (grand mean + covariate-only fit), then batch means/vars.
        std::vector<double> gsum(B, 0.0);
        for (int n = 0; n < N; ++n) {
            const double* xn = &design[(std::size_t)n * M];
            double cfit = alpha;
            for (int i = 0; i < C; ++i) cfit += xn[i] * beta[i];      // covariate cols 0..C-1
            double z = (y[n] - cfit) / sigma;
            gsum[d.batch[n]] += z;
        }
        for (int b = 0; b < B; ++b) {
            double cnt = (double)batch_n[b];
            gamma_hat[(std::size_t)p * B + b] = (cnt > 0.0) ? gsum[b] / cnt : 0.0;
        }
        std::vector<double> vsum(B, 0.0);
        for (int n = 0; n < N; ++n) {
            const double* xn = &design[(std::size_t)n * M];
            double cfit = alpha;
            for (int i = 0; i < C; ++i) cfit += xn[i] * beta[i];
            double z = (y[n] - cfit) / sigma;
            double g = gamma_hat[(std::size_t)p * B + d.batch[n]];
            vsum[d.batch[n]] += (z - g) * (z - g);
        }
        for (int b = 0; b < B; ++b) {
            double cnt = (double)batch_n[b];
            double v = (cnt > 1.0) ? vsum[b] / (cnt - 1.0) : 1.0;
            delta_hat[(std::size_t)p * B + b] = (v > 0.0) ? v : 1e-12;
        }
    }
}

// ---------------------------------------------------------------------------
// estimate_priors: fit the empirical-Bayes hyperparameters from raw stats.
//   gamma_bar[b] = mean over features of gamma_hat[.,b]      (prior mean of loc.)
//   tau2[b]      = variance over features of gamma_hat[.,b]  (prior var of loc.)
//   For the scale, fit an inverse-gamma to the across-feature delta_hat[.,b] by
//   method of moments (NeuroComBat's `aprior`/`bprior`):
//     m = mean(delta_hat[.,b]), s2 = var(delta_hat[.,b])
//     a_prior = (2 s2 + m^2) / s2,   b_prior = (m s2 + m^3) / s2.
//   These give each per-feature scale a sensible shrinkage target.
// ---------------------------------------------------------------------------
void estimate_priors(const Dataset& d, const std::vector<double>& design,
                     std::vector<double>& gamma_bar, std::vector<double>& tau2,
                     std::vector<double>& a_prior,   std::vector<double>& b_prior,
                     std::vector<int>&    batch_n)
{
    const int P = d.P, B = d.B, N = d.N;
    // Count samples per batch (needed everywhere downstream).
    batch_n.assign(B, 0);
    for (int n = 0; n < N; ++n) batch_n[d.batch[n]]++;

    std::vector<double> gamma_hat, delta_hat;
    raw_batch_stats(d, design, batch_n, gamma_hat, delta_hat);

    gamma_bar.assign(B, 0.0); tau2.assign(B, 0.0);
    a_prior.assign(B, 0.0);   b_prior.assign(B, 0.0);
    for (int b = 0; b < B; ++b) {
        // Location prior: mean and variance of gamma_hat across features.
        double gm = 0.0;
        for (int p = 0; p < P; ++p) gm += gamma_hat[(std::size_t)p * B + b];
        gm /= (double)P;
        double gv = 0.0;
        for (int p = 0; p < P; ++p) {
            double dd = gamma_hat[(std::size_t)p * B + b] - gm; gv += dd * dd;
        }
        gv /= (double)P;
        gamma_bar[b] = gm;
        tau2[b] = (gv > 0.0) ? gv : 1e-6;      // guard a degenerate (single-feature) panel
        // Scale prior: method-of-moments inverse-gamma on delta_hat.
        double dm = 0.0;
        for (int p = 0; p < P; ++p) dm += delta_hat[(std::size_t)p * B + b];
        dm /= (double)P;
        double dv = 0.0;
        for (int p = 0; p < P; ++p) {
            double dd = delta_hat[(std::size_t)p * B + b] - dm; dv += dd * dd;
        }
        dv /= (double)P;
        if (dv <= 0.0) dv = 1e-6;
        a_prior[b] = (2.0 * dv + dm * dm) / dv;
        b_prior[b] = (dm * dv + dm * dm * dm) / dv;
    }
}

// ---------------------------------------------------------------------------
// combat_cpu: the trusted serial reference. Loop the shared per-feature core
// over the P feature rows. Complexity O(P * (N*M + M^3)); fully parallel across
// features -> that is exactly what the GPU kernel exploits.
// ---------------------------------------------------------------------------
void combat_cpu(const Dataset& d, const std::vector<double>& design,
                const std::vector<double>& gamma_bar, const std::vector<double>& tau2,
                const std::vector<double>& a_prior,   const std::vector<double>& b_prior,
                const std::vector<int>&    batch_n,
                std::vector<double>& out)
{
    const int N = d.N, P = d.P, M = d.M(), C = d.C, B = d.B;
    out.assign((std::size_t)P * N, 0.0);
    for (int p = 0; p < P; ++p) {
        cb_harmonize_feature(
            &d.Y[(std::size_t)p * N], design.data(), d.batch.data(),
            N, M, d.Ccols(), B,
            batch_n.data(), gamma_bar.data(), tau2.data(),
            a_prior.data(), b_prior.data(),
            &out[(std::size_t)p * N]);
    }
    (void)C;   // C is folded into Ccols(); silence unused in case of future edits
}

// ---------------------------------------------------------------------------
// max_batch_mean_gap: the headline "did harmonization work" diagnostic. For each
// feature, compute every batch's mean, then the max-minus-min spread across
// batches; return the largest such spread over all features. Large before
// harmonization (the scanner offsets), near-zero after.
// ---------------------------------------------------------------------------
double max_batch_mean_gap(const Dataset& d, const std::vector<double>& table) {
    const int N = d.N, P = d.P, B = d.B;
    std::vector<double> sum(B), cnt(B);
    double worst = 0.0;
    for (int p = 0; p < P; ++p) {
        std::fill(sum.begin(), sum.end(), 0.0);
        std::fill(cnt.begin(), cnt.end(), 0.0);
        for (int n = 0; n < N; ++n) {
            sum[d.batch[n]] += table[(std::size_t)p * N + n];
            cnt[d.batch[n]] += 1.0;
        }
        double lo = 1e300, hi = -1e300;
        for (int b = 0; b < B; ++b) if (cnt[b] > 0.0) {
            double m = sum[b] / cnt[b];
            if (m < lo) lo = m;
            if (m > hi) hi = m;
        }
        if (hi - lo > worst) worst = hi - lo;
    }
    return worst;
}
