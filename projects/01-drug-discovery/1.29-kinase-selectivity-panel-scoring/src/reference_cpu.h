// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference for kinase panel scoring
// ---------------------------------------------------------------------------
// Project 1.29 : Kinase Selectivity Panel Scoring
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the host C++ compiler and must not see any
//   CUDA syntax, so the shared DATA MODEL (the loaded panel container + the text
//   loader) and the CPU-reference prototypes live here. kernels.cuh ALSO includes
//   this header to reuse the same types -- nothing CUDA-specific leaks in either
//   direction. The per-kinase *physics* lives one level deeper in
//   selectivity_core.h (the __host__ __device__ core shared by CPU and GPU).
//
// THE PROBLEM (see ../THEORY.md for the full derivation)
//   Kinases share highly similar ATP pockets, so a drug that hits the intended
//   kinase often hits dozens of others -> toxicity. "Selectivity panel scoring"
//   profiles ONE compound against a PANEL of N kinases and asks two questions:
//       (1) which kinases does it bind, and how strongly (predicted pK per kinase)?
//       (2) how SELECTIVE is it overall (the S-score: fraction of the panel hit)?
//   Each kinase is scored INDEPENDENTLY from the same compound -> embarrassingly
//   parallel: one GPU thread per kinase (kernels.cu), the constant compound in
//   GPU constant memory (the 1.12 "score one query vs N items" pattern).
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh. (Physics: selectivity_core.h.)
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "selectivity_core.h"   // NFEAT, KinasePocket, score_kinase, pK helpers

// ---------------------------------------------------------------------------
// KinasePanel : a loaded problem instance.
//   ligand  : [NFEAT] the query compound's per-channel feature offers.
//   pockets : N kinase pockets (each a KinasePocket: req[NFEAT] + bias + id).
//   names   : [N] human-readable kinase names, parallel to `pockets` (report only).
//   n       : number of kinases in the panel (== pockets.size()).
// The pockets are stored as a flat std::vector<KinasePocket> precisely so the GPU
// path can memcpy the whole array to the device in one shot (kernels.cu).
// ---------------------------------------------------------------------------
struct KinasePanel {
    int n = 0;                          // number of kinases in the panel
    int32_t ligand[NFEAT] = {0};        // the query compound's feature offers
    std::vector<KinasePocket> pockets;  // [n] pocket requirement vectors
    std::vector<std::string> names;     // [n] kinase names (reporting only)
};

// ---------------------------------------------------------------------------
// load_panel : parse the tiny text dataset (format documented in data/README.md).
//   File layout (whitespace-separated, '#'-comment lines ignored):
//     line:  N  NFEAT
//     line:  LIGAND  f0 f1 ... f7              (the query compound's offers)
//     N lines: <name> <bias> r0 r1 ... r7      (one kinase pocket per line)
//   Throws std::runtime_error on a missing file, a NFEAT mismatch, or bad data
//   so the demo fails LOUDLY instead of silently scoring garbage.
// ---------------------------------------------------------------------------
KinasePanel load_panel(const std::string& path);

// ---------------------------------------------------------------------------
// score_panel_cpu : the trusted serial reference.
//   For each kinase i, compute the raw match score (score_kinase from the shared
//   core), convert to a predicted affinity pK in milli-units, and record whether
//   it counts as a "hit" (pK >= threshold). Fills two parallel output arrays:
//     pK_milli[i] : predicted affinity * 1000 (exact integer)
//     hit[i]      : 1 if bound above threshold, else 0
//   Returns the integer S-count = sum(hit) -- the numerator of the S-score. This
//   is the obviously-correct baseline the GPU result is checked against
//   (bit-for-bit, since both call the same __host__ __device__ score_kinase()).
// ---------------------------------------------------------------------------
int32_t score_panel_cpu(const KinasePanel& panel,
                        std::vector<int32_t>& pK_milli,
                        std::vector<int32_t>& hit);
