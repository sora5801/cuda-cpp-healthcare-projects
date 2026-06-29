// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference for route scoring
// ---------------------------------------------------------------------------
// Project 1.20 : Reaction Yield / Retrosynthesis Scoring
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the host C++ compiler and must not see any
//   CUDA syntax, so the shared DATA MODEL (the RouteSet container and the file
//   loader) and the CPU reference prototype live here. The GPU side
//   (kernels.cuh) also includes this header to reuse the RouteSet type. The
//   actual per-route arithmetic is shared separately through route_score.h
//   (the __host__ __device__ core), so the math is literally the same code on
//   both sides -- this header only owns the I/O and the container shape.
//
// THE PROBLEM (see ../THEORY.md for the full derivation)
//   We are handed a BATCH of N candidate retrosynthetic routes for one target
//   molecule. Each route is described by:
//     * up to MAX_STEPS reaction steps, each a NUM_FEATURES feature vector
//       (template prior, precedent count, condition penalty, selectivity);
//     * a building-block AVAILABILITY factor in [0,1] for its leaf reactants.
//   We score every route with route_score() (route_score.h) and rank them. Every
//   route is INDEPENDENT, so the GPU gives each route its own thread -- the same
//   "independent jobs" pattern as 1.12 Tanimoto and 12.01 spectral search.
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh. Read route_score.h first
// for the scoring formula itself.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "route_score.h"   // MAX_STEPS, NUM_FEATURES, route_score() (pure C++)

// ---------------------------------------------------------------------------
// RouteSet: a loaded batch of N candidate routes plus the shared scoring model.
//
//   feats : N * MAX_STEPS * NUM_FEATURES floats, ROW-MAJOR. Route r occupies the
//           block starting at  r * (MAX_STEPS * NUM_FEATURES); within that block,
//           step s occupies  s * NUM_FEATURES .. s*NUM_FEATURES + NUM_FEATURES-1.
//           A step row whose first element == STEP_ABSENT is padding (route
//           shorter than MAX_STEPS). This flat layout is exactly what we upload
//           to the GPU so thread r can index its own route by arithmetic.
//   avail : N floats, the building-block availability factor of each route.
//   w     : NUM_FEATURES logistic weights, SHARED by every step of every route.
//   b     : the logistic bias (intercept), shared.
//
//   Keeping the model (w,b) IN the dataset means the demo is fully specified by
//   one file: the same weights drive the CPU reference and the GPU constant-memory
//   copy, guaranteeing identical results.
// ---------------------------------------------------------------------------
struct RouteSet {
    int n = 0;                       // number of candidate routes
    std::vector<float> feats;        // [n * MAX_STEPS * NUM_FEATURES], row-major
    std::vector<float> avail;        // [n] availability factors in [0,1]
    float w[NUM_FEATURES] = {0};     // shared logistic weights
    float b = 0.0f;                  // shared logistic bias
};

// Stride (in floats) of one route's feature block -- handy shorthand reused by
// the loader, the CPU reference, and the kernel so the indexing never drifts.
constexpr int ROUTE_STRIDE = MAX_STEPS * NUM_FEATURES;

// ---------------------------------------------------------------------------
// load_routes: parse the tiny text dataset documented in data/README.md.
//   Format (whitespace-separated, '#' comment lines ignored):
//     line: "<n> <MAX_STEPS> <NUM_FEATURES>"   (the last two must match the build)
//     line: NUM_FEATURES weights, then 1 bias  (the shared logistic model)
//     then n route blocks, each:
//       line: "<num_real_steps> <availability>"
//       num_real_steps lines of NUM_FEATURES feature values
//   Padding (steps beyond num_real_steps up to MAX_STEPS) is filled with
//   STEP_ABSENT by the loader, so the kernel/CPU never see ragged routes.
//   Throws std::runtime_error on a missing file or a shape mismatch.
// ---------------------------------------------------------------------------
RouteSet load_routes(const std::string& path);

// ---------------------------------------------------------------------------
// score_routes_cpu: the trusted serial baseline. Fills out[r] with
// route_score() of route r (from route_score.h). This is the obviously-correct
// reference the GPU kernel is checked against, and the timing baseline that
// makes the speed-up legible. out is resized to rs.n.
// ---------------------------------------------------------------------------
void score_routes_cpu(const RouteSet& rs, std::vector<float>& out);
