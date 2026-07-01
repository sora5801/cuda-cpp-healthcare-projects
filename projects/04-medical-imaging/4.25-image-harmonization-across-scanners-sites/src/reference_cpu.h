// ===========================================================================
// src/reference_cpu.h  --  Dataset + design-matrix builder + CPU ComBat reference
// ---------------------------------------------------------------------------
// Project 4.25 : Image Harmonization Across Scanners/Sites
//
// Pure C++ (no CUDA). The per-feature ComBat math lives in combat.h and is shared
// verbatim with the GPU kernel. This header declares:
//   * Dataset          -- the loaded multi-site feature table.
//   * load_dataset     -- parse the tiny synthetic sample (data/sample).
//   * build_design     -- assemble the [N x M] design matrix from covariates +
//                         batch indicators (shared by CPU and GPU so both fit the
//                         SAME model).
//   * estimate_priors  -- compute the empirical-Bayes priors (per-batch gamma_bar,
//                         tau2, a_prior, b_prior) from the raw data (shared).
//   * combat_cpu       -- the trusted serial reference: harmonize every feature.
//
// kernels.cu reuses Dataset + build_design + estimate_priors, then runs the GPU
// harmonization. main.cu compares the two harmonized tables.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "combat.h"   // cb_harmonize_feature (the shared per-feature core)

// ---------------------------------------------------------------------------
// Dataset: a multi-site feature table with biology to preserve.
//   N samples (subjects/images), P features per sample, B batches (scanners),
//   C biological covariates (NOT counting the intercept). The harmonizer removes
//   the per-batch signature from each feature while keeping the covariate signal.
// ---------------------------------------------------------------------------
struct Dataset {
    int N = 0;   // number of samples (subjects/images)
    int P = 0;   // number of features per sample
    int B = 0;   // number of batches (scanners/sites)
    int C = 0;   // number of biological covariates (age, sex, ...), no intercept

    std::vector<double> Y;      // [P*N] features, ROW-major (feature p, sample n)
    std::vector<int>    batch;  // [N]   batch index (0..B-1) of each sample
    std::vector<double> cov;    // [N*C] covariate values, row-major (sample, cov)

    // The design-matrix width M = C (covariates) + B (batch indicators). There is
    // deliberately NO separate intercept column: the B batch dummies already span
    // the intercept, so adding one would make X rank-deficient (see combat.h and
    // THEORY.md §Numerics). The covariate block we PRESERVE is columns [0..C-1];
    // the batch block we REMOVE is the last B columns. The grand mean is recovered
    // from the batch coefficients inside cb_harmonize_feature.
    int Ccols() const { return C; }            // covariate columns (no intercept)
    int M()     const { return C + B; }        // total design columns
};

// Parse the sample text format (documented in data/README.md):
//   line 1:  N P B C
//   line 2:  batch[0] batch[1] ... batch[N-1]
//   next N lines: C covariate values per sample (omitted entirely if C == 0)
//   next P lines: N feature values per line (one feature row per line)
Dataset load_dataset(const std::string& path);

// Build the [N x M] row-major design matrix X shared by both fits:
//   column 0            : intercept (all ones)
//   columns 1..C        : the C biological covariates (copied from d.cov)
//   columns Ccols..M-1  : B batch indicator columns (1 if sample in that batch)
// This "reference-free" full-dummy coding matches NeuroComBat's design; the
// intercept + covariates are the block we KEEP, the batch dummies are removed.
void build_design(const Dataset& d, std::vector<double>& design);

// Estimate the empirical-Bayes priors from the standardized data, per batch:
//   gamma_bar[b] : mean across features of the batch's raw location gamma_hat.
//   tau2[b]      : variance across features of gamma_hat.
//   a_prior[b], b_prior[b] : method-of-moments inverse-gamma hyperparameters
//                 fitted to the across-feature distribution of delta_hat.
// These priors are what let each feature "borrow strength" from the whole panel
// (that is the entire point of ComBat vs. a naive per-batch z-score). Shared by
// CPU and GPU so both harmonize with identical priors. THEORY.md §EB derives them.
void estimate_priors(const Dataset& d, const std::vector<double>& design,
                     std::vector<double>& gamma_bar, std::vector<double>& tau2,
                     std::vector<double>& a_prior,   std::vector<double>& b_prior,
                     std::vector<int>&    batch_n);

// CPU reference: harmonize every feature serially by looping cb_harmonize_feature
// (combat.h) over the P feature rows. Fills `out` [P*N] with the harmonized table.
// This is the trusted ground truth the GPU result is checked against.
void combat_cpu(const Dataset& d, const std::vector<double>& design,
                const std::vector<double>& gamma_bar, const std::vector<double>& tau2,
                const std::vector<double>& a_prior,   const std::vector<double>& b_prior,
                const std::vector<int>&    batch_n,
                std::vector<double>& out);

// A small diagnostic reused by main.cu for the STDOUT report: the largest
// absolute difference between per-batch feature MEANS, maximized over features
// and batches. Before harmonization this is large (the scanner offsets); after,
// it collapses toward zero. Returns that scalar for a given feature table.
double max_batch_mean_gap(const Dataset& d, const std::vector<double>& table);
