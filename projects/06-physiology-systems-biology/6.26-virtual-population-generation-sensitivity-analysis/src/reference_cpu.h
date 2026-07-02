// ===========================================================================
// src/reference_cpu.h  --  Population config loader + CPU reference + Sobol post
// ---------------------------------------------------------------------------
// Project 6.26 : Virtual Population Generation & Sensitivity Analysis
//
// Pure C++ (no CUDA). The per-sample model + sampling live in vpop.h and are
// shared with the GPU kernel; kernels.cu reuses VpopParams. The CPU reference
// evaluates the SAME N*(k+2) Saltelli model runs as the GPU, so the raw AUC
// arrays match to round-off and verification is exact.
//
// The Sobol post-processing (turning the raw AUC array into first-order and
// total-order sensitivity indices) is a cheap serial reduction done once on the
// host for BOTH the CPU array and the GPU array -- so we can compare not only
// the raw evaluations but the final indices too.
//
// READ THIS AFTER: vpop.h.  READ BEFORE: reference_cpu.cpp, kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "vpop.h"   // VpopParams, VPOP_K, vpop_eval, vpop_num_evals

// ---------------------------------------------------------------------------
// The finished sensitivity result. Indices are reported in the fixed parameter
// order {ka, CL, V, F} (see VPOP_K comment in vpop.h).
//   S[j]  : first-order Sobol index  -- fraction of Var(AUC) from parameter j
//           acting ALONE (main effect).
//   ST[j] : total-order Sobol index  -- j's main effect PLUS every interaction
//           it participates in. ST[j] >= S[j] always; ST[j] ~ S[j] means j has
//           no interactions; sum(ST) > 1 signals interactions in the model.
//   mean, var : the sample mean and variance of AUC over matrix A (population
//           exposure summary, and the Var that normalizes the indices).
// ---------------------------------------------------------------------------
struct SobolResult {
    double S[VPOP_K];
    double ST[VPOP_K];
    double mean;
    double var;
};

// Load VpopParams from the whitespace text format documented in data/README.md:
//   dose
//   ka_lo ka_hi
//   CL_lo CL_hi
//   V_lo  V_hi
//   F_lo  F_hi
//   t_end steps
//   N seed
// Throws std::runtime_error on a missing file or malformed / non-physical input.
VpopParams load_vpop(const std::string& path);

// CPU reference: evaluate every one of the N*(k+2) Saltelli model runs serially
// into `out` (sized to vpop_num_evals(P.N)). This is the trusted baseline the
// GPU raw-output array is checked against. Each entry is one virtual patient's
// AUC under one Saltelli sample.
void evaluate_cpu(const VpopParams& P, std::vector<double>& out);

// Sobol post-processing (Saltelli estimators). Consumes the flat AUC array
// (layout: block b occupies rows [b*N, (b+1)*N); see vpop.h) and fills a
// SobolResult. Pure arithmetic on host doubles; identical whether the array
// came from the CPU or the GPU, so the two SobolResults must match exactly.
SobolResult compute_sobol(const VpopParams& P, const std::vector<double>& f);
