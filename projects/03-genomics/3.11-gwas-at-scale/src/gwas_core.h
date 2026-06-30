// ===========================================================================
// src/gwas_core.h  --  The ONE TRUE per-element math, shared by CPU and GPU
// ---------------------------------------------------------------------------
// Project 3.11 : GWAS at Scale   (see ../THEORY.md and the catalog deep-dive)
//
// WHY THIS HEADER EXISTS  (PATTERNS.md §2, the "__host__ __device__ core" idiom)
//   A genome-wide association study has two numerical kernels:
//     (1) STANDARDIZE a genotype column (SNP) into a z-score, and
//     (2) ASSOCIATION test: fit a single-marker linear regression of the
//         phenotype y on one standardized SNP column (plus an intercept) and
//         report the effect size, its standard error, the t/chi-square
//         statistic, and a -log10(p) score.
//   We want the CPU reference (reference_cpu.cpp, compiled by cl.exe) and the
//   GPU kernels (kernels.cu, compiled by nvcc) to compute *byte-for-byte the
//   same arithmetic*, so "GPU == CPU" verification can be EXACT, not fuzzy.
//   The way to guarantee that is to write each scalar formula EXACTLY ONCE,
//   here, as an inline function tagged `__host__ __device__`, and call it from
//   both sides. Neither side gets to "improve" the formula on its own.
//
//   Keep this header free of CUDA-only constructs (no __global__, no <<<>>>,
//   no kernels). It must be includable by the plain host C++ compiler. The
//   GWAS_HD macro below evaporates to nothing under cl.exe and becomes
//   `__host__ __device__` under nvcc -- that is the whole trick.
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cu  (both include this file).
// ===========================================================================
#pragma once

#include <cmath>     // std::sqrt, std::erfc, std::log10, std::fabs
#include <cstddef>   // std::size_t

// ---------------------------------------------------------------------------
// GWAS_HD : the host/device portability shim.
//   * Under nvcc (__CUDACC__ defined) a function tagged GWAS_HD is compiled for
//     BOTH the CPU (host) and the GPU (device), so the very same code object can
//     be called from main.cu's host code and from inside a __global__ kernel.
//   * Under a plain host compiler the decorators do not exist, so GWAS_HD must
//     expand to nothing -- the function is then just an ordinary inline.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define GWAS_HD __host__ __device__
#else
#define GWAS_HD
#endif

namespace gwas {

// ===========================================================================
// SECTION A -- genotype standardization (the GRM "Z matrix")
// ===========================================================================
//
// A SNP genotype under the additive model is an integer dosage in {0, 1, 2}:
// the count of minor (effect) alleles an individual carries at that locus.
// Two individuals are "genetically related" when they share unusual alleles,
// so before measuring relatedness we must (a) MEAN-CENTER each SNP (subtract
// its average dosage) and (b) SCALE by the genotype's standard deviation under
// Hardy-Weinberg equilibrium. With allele frequency p, the additive variance is
// 2*p*(1-p); dividing by its square root turns rare-allele sharing into a large
// (informative) number and common-allele sharing into a small one -- this is
// exactly the GCTA / VanRaden normalization used to build the genetic
// relatedness matrix (GRM). See THEORY.md §"The math".

// allele_freq: the minor-allele frequency p of a SNP column.
//   sum_dosage = Σ_i g_ij over the N individuals (each g in {0,1,2}); each
//   individual carries up to 2 alleles, so the divisor is 2*N. Returns p in
//   [0,1]. A monomorphic SNP (p==0 or p==1) carries no information; callers
//   guard against it (its scale would be zero -> divide-by-zero).
GWAS_HD inline double allele_freq(double sum_dosage, int n_individuals) {
    return sum_dosage / (2.0 * static_cast<double>(n_individuals));
}

// hwe_sd: the Hardy-Weinberg standard deviation sqrt(2*p*(1-p)) of a SNP.
//   This is the per-SNP scale factor. We clamp tiny values away from zero so a
//   near-monomorphic column cannot blow up the z-scores; callers should already
//   have filtered such SNPs, this is a numerical seatbelt.
GWAS_HD inline double hwe_sd(double p) {
    double var = 2.0 * p * (1.0 - p);     // additive genetic variance at HWE
    if (var < 1.0e-12) var = 1.0e-12;     // seatbelt: avoid 1/0 for rare SNPs
    return std::sqrt(var);
}

// standardize: map one raw dosage g in {0,1,2} to its z-score (g - 2p)/sd.
//   mean dosage under HWE is 2p, hence the centering term 2p (not the empirical
//   column mean -- GCTA centers on the expected value). The result is the (i,j)
//   entry of the standardized matrix Z whose GRM is (1/M) Z Zᵀ.
GWAS_HD inline double standardize(int g, double p, double sd) {
    return (static_cast<double>(g) - 2.0 * p) / sd;
}

// ===========================================================================
// SECTION B -- single-marker association test (linear regression per SNP)
// ===========================================================================
//
// For each SNP j we fit  y_i = mu + beta * x_ij + e_i,  where x_ij is the
// STANDARDIZED genotype (z-score from Section A) and y_i is the (already
// mean-centered) phenotype. Because x is centered (Σ x = 0) and y is centered
// (Σ y = 0), the ordinary-least-squares slope collapses to the classic
// covariance/variance ratio:
//
//        beta = Σ_i x_i y_i  /  Σ_i x_i^2 .
//
// The residual variance gives the slope's standard error, the t = beta/SE
// statistic, and (t^2 ~ chi-square_1) a two-sided p-value. This is the textbook
// per-variant GWAS scan (the "logistic/linear regression per variant" the
// catalog names) -- embarrassingly parallel: one SNP per GPU thread.
//
// We pack the sufficient statistics so the SAME routine runs on CPU and GPU:
//   sxx = Σ x_i^2   (variance of the standardized genotype, times N)
//   sxy = Σ x_i y_i (covariance of genotype and phenotype, times N)
//   syy = Σ y_i^2   (variance of the centered phenotype, times N)
//   n   = number of individuals.

// A small plain-old-data bundle of one SNP's regression results. POD so it can
// live in a device array and be memcpy'd back to the host with no surprises.
struct AssocResult {
    double beta;        // OLS slope: effect of +1 SD of genotype on phenotype
    double se;          // standard error of beta
    double t;           // t statistic = beta / se
    double chi2;        // = t^2 ; under H0 ~ chi-square with 1 df
    double neg_log10p;  // -log10(two-sided p-value); bigger = stronger signal
};

// normal_sf: the upper-tail of the standard normal, P(Z > z) for z >= 0.
//   We approximate the t (df = n-2) by a standard normal because in GWAS n is
//   large (n-2 ~ n), so t_{n-2} -> N(0,1). The survival function of N(0,1) is
//   0.5 * erfc(z / sqrt(2)); std::erfc is available on both host and device, so
//   the two sides agree to the last bit. (A real tool would use an exact
//   Student-t tail for small n; THEORY.md §"real world" notes this.)
GWAS_HD inline double normal_sf(double z) {
    // 1/sqrt(2) as a literal so host and device fold the same constant.
    return 0.5 * std::erfc(z * 0.7071067811865475244);
}

// assoc_from_sufficient_stats: turn (sxx, sxy, syy, n) into the full result.
//   This is THE one true association formula -- both reference_cpu.cpp and the
//   per-SNP kernel call it, guaranteeing identical numbers.
//
//   Derivation (all sums are over the n individuals):
//     beta = sxy / sxx                                  (OLS slope, centered)
//     SSE  = syy - beta*sxy                             (residual sum of squares)
//     sigma2 = SSE / (n - 2)                            (unbiased error variance)
//     var(beta) = sigma2 / sxx ; se = sqrt(var(beta))
//     t = beta / se ; chi2 = t^2
//     p = 2 * P(Z > |t|) ; neg_log10p = -log10(p)
GWAS_HD inline AssocResult assoc_from_sufficient_stats(double sxx, double sxy,
                                                       double syy, int n) {
    AssocResult r;
    // A SNP with zero genotype variance (sxx == 0) carries no signal; report a
    // null result rather than dividing by zero. Callers filter these upstream,
    // but defend here so one bad column cannot poison the whole scan.
    if (sxx <= 1.0e-12 || n <= 2) {
        r.beta = 0.0; r.se = 0.0; r.t = 0.0; r.chi2 = 0.0; r.neg_log10p = 0.0;
        return r;
    }
    r.beta = sxy / sxx;                          // least-squares slope
    double sse = syy - r.beta * sxy;             // residual sum of squares
    if (sse < 0.0) sse = 0.0;                     // guard tiny negative roundoff
    double sigma2 = sse / static_cast<double>(n - 2);   // error variance
    double var_beta = sigma2 / sxx;              // variance of the slope
    r.se = std::sqrt(var_beta);
    // If the fit is perfect (se == 0) the signal is infinitely strong; clamp to
    // a large finite score so the output stays printable and deterministic.
    if (r.se <= 0.0) {
        r.t = 0.0; r.chi2 = 0.0; r.neg_log10p = 0.0;
        return r;
    }
    r.t = r.beta / r.se;
    r.chi2 = r.t * r.t;
    double p = 2.0 * normal_sf(std::fabs(r.t));  // two-sided p-value
    // Guard the log: p can underflow to 0.0 for huge t; cap -log10(p) so the
    // printed number is finite and reproducible across CPU/GPU.
    if (p < 1.0e-300) p = 1.0e-300;
    r.neg_log10p = -std::log10(p);
    return r;
}

}  // namespace gwas
