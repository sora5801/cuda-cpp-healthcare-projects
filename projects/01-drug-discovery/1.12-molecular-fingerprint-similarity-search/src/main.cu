// ===========================================================================
// src/main.cu  --  Entry point: load fingerprints, search, verify, report
// ---------------------------------------------------------------------------
// Project 1.12 : Molecular Fingerprint Similarity Search
//
// The 5-step shape every project in this repo follows:
//   1. Load the problem (a query + n library fingerprints from data/sample).
//   2. CPU reference  (reference_cpu.cpp)  -> trusted scores.
//   3. GPU search     (kernels.cu)         -> the thing being taught.
//   4. VERIFY: GPU agrees with CPU within tolerance.
//   5. REPORT: deterministic top-K to stdout; timing + error to stderr.
//
// STDOUT is byte-for-byte deterministic (diffed by demo/run_demo against
// demo/expected_output.txt); run-to-run timings go to STDERR.
//
// Code tour: start here, then kernels.cuh -> kernels.cu, then reference_cpu.*.
// ===========================================================================
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "kernels.cuh"        // tanimoto_gpu, FingerprintSet, FP_WORDS, FP_BITS
#include "reference_cpu.h"    // load_fingerprints, tanimoto_cpu
#include "util/io.hpp"        // util::CpuTimer, util::max_abs_err

static const char* PROJECT_ID   = "1.12";
static const char* PROJECT_NAME = "Molecular Fingerprint Similarity Search";

// Tolerance: CPU and GPU use identical exact-integer popcounts and IEEE float
// division, so they agree bit-for-bit; 1e-6 is a generous safety margin.
static constexpr double TOLERANCE = 1.0e-6;
static constexpr int    TOP_K     = 5;       // how many best hits to report

// Return the indices of the TOP_K largest scores, ties broken by lower index
// (so the ranking is deterministic). Uses partial_sort on an index vector.
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
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/fingerprints_sample.txt";
    FingerprintSet fps;
    try {
        fps = load_fingerprints(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<float> score_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    tanimoto_cpu(fps, score_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU search (kernel timed inside the wrapper) ------------------
    std::vector<float> score_gpu;
    float gpu_kernel_ms = 0.0f;
    tanimoto_gpu(fps, score_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    const double err = util::max_abs_err(score_cpu, score_gpu);
    const bool pass = err <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    const std::vector<int> best = top_k(score_gpu, TOP_K);
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("Tanimoto search: 1 query vs %d library fingerprints (%d-bit)\n", fps.n, FP_BITS);
    std::printf("top-%d most similar:\n", static_cast<int>(best.size()));
    for (std::size_t r = 0; r < best.size(); ++r)
        std::printf("  #%zu  lib[%d]  Tanimoto = %.6f\n", r + 1, best[r], score_gpu[best[r]]);
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-06)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (n=%d, %d-bit fingerprints)\n",
                 path.c_str(), fps.n, FP_BITS);
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny sample is dominated by "
                         "launch/copy overhead; the GPU wins at library scale (millions).\n");
    std::fprintf(stderr, "[verify] max_abs_err = %.3e  (tolerance %.1e)\n", err, TOLERANCE);

    return pass ? 0 : 1;
}
