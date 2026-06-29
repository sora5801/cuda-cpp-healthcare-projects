// ===========================================================================
// src/main.cu  --  Entry point: load routes, score, verify, report best routes
// ---------------------------------------------------------------------------
// Project 1.20 : Reaction Yield / Retrosynthesis Scoring
//
// The 5-step shape every project in this repo follows:
//   1. Load the problem (a batch of candidate retrosynthetic routes + the shared
//      scoring model) from data/sample.
//   2. CPU reference  (reference_cpu.cpp)  -> trusted per-route scores.
//   3. GPU scoring    (kernels.cu)         -> the thing being taught.
//   4. VERIFY: GPU agrees with CPU within tolerance (here: bit-exact).
//   5. REPORT: deterministic top-K routes to stdout; timing + error to stderr.
//
// STDOUT is byte-for-byte deterministic (diffed by demo/run_demo against
// demo/expected_output.txt); run-to-run timings go to STDERR.
//
// Code tour: start here, then route_score.h (the formula), then kernels.cuh ->
// kernels.cu, then reference_cpu.* (loader + baseline). See ../THEORY.md.
// ===========================================================================
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "kernels.cuh"        // score_routes_gpu, RouteSet, route constants
#include "reference_cpu.h"    // load_routes, score_routes_cpu
#include "util/io.hpp"        // util::max_abs_err

static const char* PROJECT_ID   = "1.20";
static const char* PROJECT_NAME = "Reaction Yield / Retrosynthesis Scoring";

// Tolerance: the CPU and GPU call the IDENTICAL route_score() (route_score.h),
// so the ALGORITHM is the same on both sides. The scores still differ by a few
// times 1e-8, because the per-step yield uses expf() and the GPU contracts
// multiply-adds into FMAs differently than the host compiler does (THEORY
// "Numerical considerations"). 1e-6 is a physically-negligible tolerance that
// comfortably covers that single-precision divergence (PATTERNS.md sec.4).
static constexpr double TOLERANCE = 1.0e-6;
static constexpr int    TOP_K     = 5;       // how many best routes to report

// Return the indices of the TOP_K largest scores, ties broken by LOWER index so
// the ranking is deterministic regardless of how std::partial_sort orders ties.
// (Same idiom as 1.12's top_k -- a deterministic report is a hard requirement.)
static std::vector<int> top_k(const std::vector<float>& score, int k) {
    std::vector<int> idx(score.size());
    std::iota(idx.begin(), idx.end(), 0);                 // 0,1,2,...,n-1
    const int kk = std::min<int>(k, static_cast<int>(idx.size()));
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
        [&](int a, int b) {
            if (score[a] != score[b]) return score[a] > score[b];  // higher first
            return a < b;                                          // tie -> lower idx
        });
    idx.resize(kk);
    return idx;
}

int main(int argc, char** argv) {
    // ---- 1. Load ----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/routes_sample.txt";
    RouteSet rs;
    try {
        rs = load_routes(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<float> score_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    score_routes_cpu(rs, score_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU scoring (kernel timed inside the wrapper) -----------------
    std::vector<float> score_gpu;
    float gpu_kernel_ms = 0.0f;
    score_routes_gpu(rs, score_gpu, &gpu_kernel_ms);

    // ---- 4. Verify --------------------------------------------------------
    const double err  = util::max_abs_err(score_cpu, score_gpu);
    const bool   pass = err <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    const std::vector<int> best = top_k(score_gpu, TOP_K);
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("Scored %d candidate retrosynthetic routes (<= %d steps, %d features each)\n",
                rs.n, MAX_STEPS, NUM_FEATURES);
    std::printf("top-%d most synthesizable routes (higher score = better):\n",
                static_cast<int>(best.size()));
    for (std::size_t r = 0; r < best.size(); ++r)
        std::printf("  #%zu  route[%d]  score = %.6f\n",
                    r + 1, best[r], score_gpu[best[r]]);
    std::printf("RESULT: %s (GPU matches CPU within tol=1e-06)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (n=%d routes)\n", path.c_str(), rs.n);
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny batch is dominated by "
                         "launch/copy overhead; the GPU wins when a planner emits millions of routes.\n");
    std::fprintf(stderr, "[verify] max_abs_err = %.3e  (tolerance %.1e)\n", err, TOLERANCE);

    return pass ? 0 : 1;
}
