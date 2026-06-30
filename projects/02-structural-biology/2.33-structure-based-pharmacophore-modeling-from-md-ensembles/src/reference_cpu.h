// ===========================================================================
// src/reference_cpu.h  --  Pharmacophore screening data model + CPU reference
// ---------------------------------------------------------------------------
// Project 2.33 : Structure-Based Pharmacophore Modeling from MD Ensembles
//
// WHAT THIS PROJECT COMPUTES
//   Ensemble-pharmacophore virtual screening. We have ONE query pharmacophore
//   (a small set of typed 3-D feature points -- the consensus of an MD ensemble)
//   and a LIBRARY of N candidate molecules, each represented as its own set of
//   typed feature points. We score every library molecule against the query with
//   a ROCS-style Gaussian "color" overlap (the Tanimoto in pharmacophore.h) and
//   return the best-matching molecules -- the same shape as a 3-D similarity
//   screen against a compound library.
//
// WHY A GPU  (catalog "CUDA Libraries & GPU Pattern")
//   A real screen scores ONE pharmacophore against 10^6 - 10^9 library conformers
//   (ZINC, Enamine REAL). Every molecule's score is INDEPENDENT, so the work is
//   embarrassingly parallel: one GPU thread per library molecule, the (small,
//   read-only) query pharmacophore broadcast from CONSTANT memory. This is the
//   "score one query vs N independent items" pattern shared with project 1.12
//   (Tanimoto fingerprints) and 12.01 (spectral search); see docs/PATTERNS.md §1.
//
//   This header is PURE C++ (no CUDA): kernels.cu includes it to reuse the data
//   structures, and reference_cpu.cpp implements the serial baseline. The actual
//   per-molecule arithmetic lives in pharmacophore.h so host and GPU match.
//
// READ THIS BEFORE: pharmacophore.h (the scoring formula), kernels.cuh.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "pharmacophore.h"   // Feature, score_molecule, overlap_pair (HD core)

// ---------------------------------------------------------------------------
// ScreenData: a loaded screening problem.
//
//   The query is a small contiguous array of Feature points.
//
//   The library has VARIABLE-LENGTH feature sets (molecule k may have 4 features,
//   molecule k+1 may have 9). We store them the way GPUs like best: a single flat
//   `lib_feats` array holding every molecule's features back-to-back, plus an
//   `offset` array (length N+1) giving where each molecule's block starts. This
//   is the classic CSR / "ragged array" layout -- one coalesced buffer instead of
//   N little allocations, and molecule k's features are
//       lib_feats[ offset[k] .. offset[k+1] ).
//   The same layout feeds the CPU loop and the GPU kernel unchanged.
// ---------------------------------------------------------------------------
struct ScreenData {
    int N = 0;                          // number of library molecules
    int target = -1;                    // index of the planted near-perfect match
                                        // (synthetic ground truth; -1 if unknown)
    std::vector<Feature> query;         // [n_query] the query pharmacophore
    std::vector<Feature> lib_feats;     // flat: all molecules' features concatenated
    std::vector<int>     offset;        // [N+1] CSR offsets into lib_feats
    // Convenience: number of features on molecule k = offset[k+1] - offset[k].
};

// ---------------------------------------------------------------------------
// load_screen: parse the tiny text format documented in data/README.md.
//   Header line:   "N n_query target"
//   Then n_query query rows:        "type x y z weight"
//   Then, for each of N molecules:  a line "m" (feature count) followed by
//                                   m rows "type x y z weight".
//   Throws std::runtime_error on any malformed input so demos fail loudly.
// ---------------------------------------------------------------------------
ScreenData load_screen(const std::string& path);

// ---------------------------------------------------------------------------
// query_self_overlap: O_qq, the query pharmacophore's overlap with itself. It is
//   the SAME for every library molecule (it depends only on the query), so we
//   compute it ONCE and pass it into score_molecule() for all N molecules. Uses
//   the same overlap_pair() the per-molecule score uses, for exact consistency.
// ---------------------------------------------------------------------------
double query_self_overlap(const ScreenData& s);

// ---------------------------------------------------------------------------
// screen_cpu: the trusted serial baseline. For each library molecule k it calls
//   score_molecule(query, n_query, self_qq, &lib_feats[offset[k]], n_k) and writes
//   the Tanimoto color score into scores[k]. This is the reference the GPU kernel
//   is verified against (they call the SAME score_molecule, so they should agree
//   to ~float precision).
// ---------------------------------------------------------------------------
void screen_cpu(const ScreenData& s, double self_qq, std::vector<float>& scores);
