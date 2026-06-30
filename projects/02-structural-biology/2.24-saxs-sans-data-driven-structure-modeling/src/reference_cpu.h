// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference for the Debye SAXS profile
// ---------------------------------------------------------------------------
// Project 2.24 : SAXS / SANS Data-Driven Structure Modeling
//
// WHAT THIS PROJECT COMPUTES
//   Given a 3D atomic model of a protein (coordinates + per-atom scattering
//   strengths) and a 1D experimental scattering curve I_exp(q), we:
//     1. FORWARD-MODEL the scattering profile I_model(q) from the coordinates
//        via the Debye formula (saxs_core.h), over a grid of q values.
//     2. Put it on the same scale as the experiment (a single least-squares
//        scale factor c) and report the chi-square goodness of fit.
//     3. Estimate the radius of gyration Rg from the low-q "Guinier" region.
//   This is the inner loop of SAXS-driven structure modeling: every candidate
//   conformation is scored by how well its forward-computed curve matches data.
//
// WHY A GPU
//   The Debye double sum is O(N_atoms^2) per q, evaluated at N_q q-values
//   -> O(N_q * N_atoms^2). For a real protein (10^3-10^4 atoms) over ~10^2 q's
//   this dominates, and every q is independent -> one GPU thread per q value.
//   See kernels.cuh and PATTERNS.md (independent jobs + per-thread reduction).
//
//   This header is PURE C++ (no CUDA): reference_cpu.cpp and main.cu both
//   include it; the GPU kernel reuses the SaxsModel struct and shares the actual
//   per-q physics through saxs_core.h.
//
// READ THIS AFTER: saxs_core.h.  Then kernels.cuh -> kernels.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// SaxsModel: one fully-loaded fitting problem.
//   The atom coordinates are stored as three parallel arrays (structure-of-
//   arrays, SoA) rather than an array of {x,y,z} structs. SoA is the GPU-friendly
//   layout: when all threads read x[j] for successive j the loads are coalesced
//   (contiguous in memory), which an array-of-structs would break up.
// ---------------------------------------------------------------------------
struct SaxsModel {
    // --- the atomic model (the candidate structure we forward-model) ---
    int n_atoms = 0;             // number of point scatterers (atoms)
    std::vector<double> x;       // [n_atoms] x coordinates, Ångström
    std::vector<double> y;       // [n_atoms] y coordinates, Ångström
    std::vector<double> z;       // [n_atoms] z coordinates, Ångström
    std::vector<double> f;       // [n_atoms] scattering strengths (electron count)

    // --- the q grid and the experimental target curve ---
    int n_q = 0;                 // number of momentum-transfer points
    std::vector<double> q;       // [n_q] momentum transfer, 1/Å (ascending)
    std::vector<double> I_exp;   // [n_q] experimental intensity (arbitrary scale)
    std::vector<double> sigma;   // [n_q] experimental 1-sigma error bars (>0)

    // --- provenance for the report (synthetic ground truth) ---
    // The sample is generated from a KNOWN structure whose true Rg we record, so
    // the demo can show that the recovered Guinier Rg matches it (a real science
    // check, not just CPU==GPU agreement). -1 if unknown.
    double true_rg = -1.0;       // Å; the Rg the synthetic model was built with
};

// Load a SaxsModel from the text sample format documented in data/README.md.
// Throws std::runtime_error if the file is missing or malformed.
SaxsModel load_model(const std::string& path);

// ---------------------------------------------------------------------------
// debye_profile_cpu: the trusted CPU reference.
//   Fills I_model[k] = Debye intensity at q[k], for every k, by calling the
//   shared per-q kernel in saxs_core.h in a plain serial loop. This is the
//   "obviously correct, no parallelism" baseline the GPU result is checked
//   against (CLAUDE.md §5 CPU reference path).
//     m        : the loaded model (atoms + q grid)
//     I_model  : output, resized to m.n_q, the forward-modeled intensities
// ---------------------------------------------------------------------------
void debye_profile_cpu(const SaxsModel& m, std::vector<double>& I_model);

// ---------------------------------------------------------------------------
// Analysis helpers shared by the CPU and GPU report paths (host-only math).
// ---------------------------------------------------------------------------

// best_scale: the single positive factor c that minimizes the chi-square between
//   c * I_model and I_exp given the error bars sigma (weighted least squares).
//   Closed form: c = sum(I_model*I_exp/sigma^2) / sum(I_model^2/sigma^2).
double best_scale(const std::vector<double>& I_model,
                  const std::vector<double>& I_exp,
                  const std::vector<double>& sigma);

// reduced_chi_square: chi^2 / n_q for the scaled model vs the experiment.
//   chi^2 = sum_k ((c*I_model[k] - I_exp[k]) / sigma[k])^2. A value near 1 means
//   the model fits the data to within the experimental noise (well-fit).
double reduced_chi_square(const std::vector<double>& I_model, double c,
                          const std::vector<double>& I_exp,
                          const std::vector<double>& sigma);

// guinier_rg: estimate the radius of gyration from the low-q Guinier law
//   ln I(q) ≈ ln I(0) - (Rg^2/3) q^2, valid for q*Rg < ~1.3. We linear-fit
//   ln I vs q^2 over the first `n_fit` points and read Rg from the slope.
//   Returns Rg in Å (or -1 if the fit is degenerate).
double guinier_rg(const std::vector<double>& q, const std::vector<double>& I,
                  int n_fit);
