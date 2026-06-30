// ===========================================================================
// src/reference_cpu.h  --  Ensemble dataset + shared reweighting helpers +
//                          the CPU reference back-calculation & reweighting.
// ---------------------------------------------------------------------------
// Project 2.35 : Electron Paramagnetic Resonance (EPR/DEER) Constrained Modeling
//
// Pure C++ (no CUDA). The per-frame physics is in deer.h; this header declares:
//   * Ensemble       -- the loaded MD ensemble + the experimental target P(r).
//   * load_ensemble  -- parse the text sample (format in data/README.md).
//   * softmax_weights / mixed_distribution / objective -- host helpers reused by
//     BOTH the CPU reference and the GPU wrapper so the two reweighting runs are
//     identical (the GPU only accelerates the per-frame histogram back-calc; the
//     tiny M-vector reweighting math is shared host code -- see THEORY "GPU
//     mapping" for why that split is the right one).
//   * deer_backcalc_cpu  -- the trusted per-frame histogram baseline.
//   * reweight_cpu       -- the trusted max-entropy reweighting baseline.
//
// kernels.cu reuses Ensemble + these helpers, so the only thing that differs
// between the CPU and GPU paths is WHERE the per-frame histograms are computed.
//
// READ THIS AFTER: deer.h.  READ BEFORE: reference_cpu.cpp, main.cu, kernels.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "deer.h"          // Spin3, deer_member_histogram, chi2_to_target, kl_to_prior
#include "deer_params.h"   // NBINS, ROTAMERS_PER_SITE, REWEIGHT_*

// ---------------------------------------------------------------------------
// Ensemble: everything loaded from the sample file.
//   M frames, each with two spin-label rotamer clouds (siteA, siteB), plus the
//   experimental target distribution P_exp(r) we are fitting to. The rotamer
//   clouds are stored flat, row-major: frame m's site-A rotamer i lives at
//   siteA[m*ROTAMERS_PER_SITE + i]. This contiguous layout is what the GPU
//   kernel reads (coalesced per frame) and what the CPU loops over.
// ---------------------------------------------------------------------------
struct Ensemble {
    int M = 0;                       // number of ensemble members (MD frames)
    std::vector<Spin3> siteA;        // [M * ROTAMERS_PER_SITE] site-1 rotamer endpoints (nm)
    std::vector<Spin3> siteB;        // [M * ROTAMERS_PER_SITE] site-2 rotamer endpoints (nm)
    std::vector<double> target;      // [NBINS] experimental P_exp(r), normalized to sum 1
    std::vector<int> truth;          // [M] 1 if frame is a "true" match (synthetic label), else 0
};

// Load an ensemble from the text format documented in data/README.md.
//   Throws std::runtime_error on a malformed or missing file (fail loudly).
Ensemble load_ensemble(const std::string& path);

// ---------------------------------------------------------------------------
// softmax_weights: turn unconstrained log-weights g[m] into a normalized,
//   strictly positive weight vector w[m] = exp(g[m]) / sum_k exp(g[k]).
//   We optimize in log-weight space precisely so the weights stay positive and
//   sum to 1 automatically -- no constrained optimizer needed. A max-subtraction
//   ("log-sum-exp trick") keeps exp() from overflowing. Shared by CPU + GPU so
//   both reweighting runs follow the identical numerical path.
// ---------------------------------------------------------------------------
void softmax_weights(const std::vector<double>& g, std::vector<double>& w);

// mixed_distribution: the population-weighted model P(r) = sum_m w[m] * P_m(r).
//   `hist` is the [M*NBINS] matrix of per-frame histograms (from CPU or GPU
//   back-calc); `w` are the M weights; result is the length-NBINS mixture. This
//   is the quantity compared against the target. Shared so the mixture is formed
//   identically regardless of where `hist` came from.
void mixed_distribution(const std::vector<double>& hist, int M,
                        const std::vector<double>& w, std::vector<double>& mixed);

// objective: L(w) = chi2( mix(w), target ) + THETA * KL( w || uniform ).
//   The scalar the reweighting minimizes. Returned split into its two parts via
//   out-params so main can report both. Shared host helper.
double objective(const std::vector<double>& hist, int M,
                 const std::vector<double>& target, const std::vector<double>& g,
                 double* out_chi2, double* out_kl);

// ---------------------------------------------------------------------------
// deer_backcalc_cpu: the trusted baseline for the per-frame histograms.
//   Loops over all M frames and calls deer_member_histogram() (deer.h) for each,
//   filling hist[M*NBINS]. This is exactly what the GPU kernel parallelizes.
// ---------------------------------------------------------------------------
void deer_backcalc_cpu(const Ensemble& e, std::vector<double>& hist);

// ---------------------------------------------------------------------------
// reweight_cpu: the trusted baseline for max-entropy reweighting.
//   Given the per-frame histograms `hist`, runs REWEIGHT_ITERS gradient-descent
//   steps on the log-weights to minimize the objective, and returns the final
//   normalized weights `w` (length M). Fills the chi^2 before/after via
//   out-params. Deterministic (fixed iterations, fixed LR, starts from uniform).
// ---------------------------------------------------------------------------
void reweight_cpu(const std::vector<double>& hist, int M,
                  const std::vector<double>& target, std::vector<double>& w,
                  double* out_chi2_before, double* out_chi2_after);
