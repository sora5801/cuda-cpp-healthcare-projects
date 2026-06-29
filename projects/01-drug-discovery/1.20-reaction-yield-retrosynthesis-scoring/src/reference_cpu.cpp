// ===========================================================================
// src/reference_cpu.cpp  --  The trusted serial route-scoring baseline + loader
// ---------------------------------------------------------------------------
// Project 1.20 : Reaction Yield / Retrosynthesis Scoring
//
// ROLE
//   (1) load_routes(): parse the tiny text dataset (data/README.md format) into
//       the flat RouteSet the GPU also consumes.
//   (2) score_routes_cpu(): the obviously-correct serial computation the GPU
//       kernel is verified against. It just loops over routes and calls the SAME
//       route_score() the kernel calls (route_score.h) -- so "CPU vs GPU" really
//       compares the parallel plumbing, not the arithmetic (which is shared).
//
//   Compiled by the host C++ compiler only (no CUDA). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h and route_score.h. Compare against kernels.cu
// (the GPU twin that calls the identical route_score()).
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// next_token: read the next whitespace-separated token from `in`, SKIPPING any
// line that begins with '#' (so the dataset can carry human-readable comments).
// We hand-roll this instead of operator>> because we must drop comment lines but
// keep numbers that may sit on the same physical line. Throws on EOF so a
// truncated file fails loudly rather than scoring garbage.
// ---------------------------------------------------------------------------
static std::string next_token(std::istream& in, const std::string& path) {
    std::string tok;
    while (in >> tok) {
        if (!tok.empty() && tok[0] == '#') {
            // A comment: swallow the rest of THIS line and keep scanning.
            std::string rest;
            std::getline(in, rest);
            continue;
        }
        return tok;
    }
    throw std::runtime_error("unexpected end of data in " + path);
}

// Convenience wrappers that parse the next token as the requested numeric type.
static float next_float(std::istream& in, const std::string& path) {
    return std::stof(next_token(in, path));
}
static int next_int(std::istream& in, const std::string& path) {
    return std::stoi(next_token(in, path));
}

RouteSet load_routes(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open route file: " + path);

    // ---- Header: "<n> <MAX_STEPS> <NUM_FEATURES>" --------------------------
    // The last two must match the COMPILED constants, because the flat layout
    // and the unrolled loops in route_score.h assume those exact dimensions.
    const int n        = next_int(in, path);
    const int maxSteps = next_int(in, path);
    const int numFeat  = next_int(in, path);
    if (maxSteps != MAX_STEPS)
        throw std::runtime_error("MAX_STEPS mismatch: file has " +
            std::to_string(maxSteps) + " but this build expects " +
            std::to_string(MAX_STEPS));
    if (numFeat != NUM_FEATURES)
        throw std::runtime_error("NUM_FEATURES mismatch: file has " +
            std::to_string(numFeat) + " but this build expects " +
            std::to_string(NUM_FEATURES));
    if (n <= 0) throw std::runtime_error("non-positive route count in " + path);

    RouteSet rs;
    rs.n = n;
    // Pre-fill the feature block with the STEP_ABSENT sentinel so any step we do
    // not explicitly read stays "padding" (yield 1 in route_score()).
    rs.feats.assign(static_cast<std::size_t>(n) * ROUTE_STRIDE, STEP_ABSENT);
    rs.avail.assign(static_cast<std::size_t>(n), 0.0f);

    // ---- Shared model: NUM_FEATURES weights, then 1 bias -------------------
    for (int f = 0; f < NUM_FEATURES; ++f) rs.w[f] = next_float(in, path);
    rs.b = next_float(in, path);

    // ---- Per-route blocks --------------------------------------------------
    for (int r = 0; r < n; ++r) {
        const int   realSteps = next_int(in, path);    // how many steps are real
        const float availability = next_float(in, path);
        if (realSteps < 0 || realSteps > MAX_STEPS)
            throw std::runtime_error("route " + std::to_string(r) +
                " has an out-of-range step count in " + path);
        if (availability < 0.0f || availability > 1.0f)
            throw std::runtime_error("route " + std::to_string(r) +
                " has availability outside [0,1] in " + path);
        rs.avail[static_cast<std::size_t>(r)] = availability;

        // Read realSteps feature rows; the remaining rows keep the STEP_ABSENT
        // sentinel from the assign() above (so they are skipped when scoring).
        float* block = &rs.feats[static_cast<std::size_t>(r) * ROUTE_STRIDE];
        for (int s = 0; s < realSteps; ++s) {
            float* row = block + s * NUM_FEATURES;
            for (int f = 0; f < NUM_FEATURES; ++f)
                row[f] = next_float(in, path);
        }
    }
    return rs;
}

// ---------------------------------------------------------------------------
// score_routes_cpu: the serial reference. One readable loop over routes, each
// scored by the shared route_score(). O(n * MAX_STEPS * NUM_FEATURES) work, O(1)
// extra space. Because it calls the very same __host__ __device__ route_score()
// the kernel uses, the CPU and GPU results agree to ~1e-8 -- the same algorithm,
// differing only by single-precision expf/FMA rounding (THEORY "verify").
// ---------------------------------------------------------------------------
void score_routes_cpu(const RouteSet& rs, std::vector<float>& out) {
    out.assign(static_cast<std::size_t>(rs.n), 0.0f);
    for (int r = 0; r < rs.n; ++r) {
        const float* block = &rs.feats[static_cast<std::size_t>(r) * ROUTE_STRIDE];
        // route_score() multiplies the per-step yields and the availability
        // bonus -- the one true formula, defined once in route_score.h.
        out[static_cast<std::size_t>(r)] =
            route_score(block, rs.avail[static_cast<std::size_t>(r)], rs.w, rs.b);
    }
}
