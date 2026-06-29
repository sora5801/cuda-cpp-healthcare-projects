// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference for ADMET screening
// ---------------------------------------------------------------------------
// Project 1.16 : ADMET / Toxicity Prediction  (reduced-scope teaching version)
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the host C++ compiler and must not see any
//   CUDA syntax, so the shared DATA MODEL (the AdmetData container + the text
//   loader) and the CPU reference prototypes live here. The GPU side
//   (kernels.cuh) also includes this header to reuse the dimensions and the
//   AdmetData type -- nothing CUDA-specific leaks in either direction. The
//   actual per-element math is shared separately via admet_core.h.
//
// THE PROBLEM IN ONE SENTENCE  (full derivation in ../THEORY.md)
//   Given N candidate molecules (each a length-D descriptor vector) and M
//   trained toxicity-endpoint models (each a logistic-regression weight vector
//   + bias), predict the NxM matrix of toxicity probabilities, then reduce it to
//   a per-endpoint count of "flagged" molecules and pick the single worst
//   (highest total risk) molecule -- the triage a chemist would do.
//
//   Every (molecule, endpoint) prediction is INDEPENDENT -> perfect data
//   parallelism: one GPU thread per matrix cell (see kernels.cu).
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh. Read admet_core.h first
// (it defines ADMET_D, ADMET_M and the shared math).
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "admet_core.h"   // ADMET_D, ADMET_M, admet_predict, admet_flagged (pure C++ under host)

// ---------------------------------------------------------------------------
// AdmetData: one fully-loaded screening problem.
//
//   Memory layout (all row-major, flat std::vector<double> for cache-friendly,
//   trivially-copyable-to-GPU storage):
//     desc    : [n * D]  molecule i's descriptor at desc[i*D .. i*D+D-1]
//     weights : [M * D]  endpoint t's weights   at weights[t*D .. t*D+D-1]
//     bias    : [M]      endpoint t's bias
//   The names are carried only for a human-readable report; the math ignores
//   them. n is a runtime value (how many molecules to screen); D and M are the
//   compile-time ADMET_D / ADMET_M so the loader can validate the file.
// ---------------------------------------------------------------------------
struct AdmetData {
    int n = 0;                               // number of molecules to screen
    std::vector<double> desc;                // [n * ADMET_D] descriptors, row-major
    std::vector<double> weights;             // [ADMET_M * ADMET_D] model weights, row-major
    std::vector<double> bias;                // [ADMET_M] model biases
    std::vector<std::string> mol_names;      // [n]  e.g. "MOL_0007" (report only)
    std::vector<std::string> endpoint_names; // [ADMET_M]  e.g. "hERG_block" (report only)
};

// ---------------------------------------------------------------------------
// AdmetResult: the deterministic outputs main.cu prints + verifies.
//   flagged_per_endpoint : [M]  integer count of molecules with p >= threshold
//                                for each endpoint (the exact, reproducible metric)
//   total_flags          : [n]  per molecule, how many of the M endpoints it
//                                trips (its "multi-endpoint risk", an integer)
//   worst_mol            : index of the molecule with the largest total_flags,
//                          ties broken by the larger summed probability, then by
//                          the lower index (fully deterministic ordering)
//   worst_mol_score      : that molecule's summed probability over all endpoints
// ---------------------------------------------------------------------------
struct AdmetResult {
    std::vector<int> flagged_per_endpoint;   // [M]
    std::vector<int> total_flags;            // [n]
    int    worst_mol = -1;
    double worst_mol_score = 0.0;
};

// The flagging threshold: a molecule is "flagged" for an endpoint when its
// predicted probability is >= this value. 0.5 is the natural logistic decision
// boundary (logit >= 0). Documented and shared so CPU and GPU use the same cut.
constexpr double ADMET_THRESHOLD = 0.5;

// ---------------------------------------------------------------------------
// load_admet: parse the text dataset documented in data/README.md. Throws
// std::runtime_error on a missing file or a dimension mismatch (the file's D/M
// must equal the compiled ADMET_D/ADMET_M). Returns a populated AdmetData.
// ---------------------------------------------------------------------------
AdmetData load_admet(const std::string& path);

// ---------------------------------------------------------------------------
// admet_predict_cpu: the trusted serial baseline. Fills `probs` ([n*M] row-major,
// probs[i*M + t] = p_{i,t}) by calling the shared admet_predict() for every
// (molecule, endpoint) pair. This is the obviously-correct reference the GPU
// kernel is checked against (and the timing baseline that makes the speed-up
// legible). `probs` is resized to n*M.
// ---------------------------------------------------------------------------
void admet_predict_cpu(const AdmetData& data, std::vector<double>& probs);

// ---------------------------------------------------------------------------
// admet_reduce: turn the [n*M] probability matrix into the deterministic
// AdmetResult (per-endpoint flag counts, per-molecule flag totals, worst
// molecule). Shared by both paths: the CPU computes it from the CPU probs and
// the GPU's reduced counts are checked against it. Integer-only accumulation
// keeps it order-independent and reproducible.
// ---------------------------------------------------------------------------
AdmetResult admet_reduce(const AdmetData& data, const std::vector<double>& probs);
