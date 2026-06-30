// ===========================================================================
// src/main.cu  --  Entry point: load screen, run CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 2.33 : Structure-Based Pharmacophore Modeling from MD Ensembles
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the screening problem (query pharmacophore + library) from a sample.
//   2. CPU reference scores (reference_cpu.cpp)            -> trusted answer.
//   3. GPU scores (kernels.cu): query in constant memory, 1 thread/molecule.
//   4. VERIFY: GPU per-molecule scores match the CPU within tolerance.
//   5. REPORT: deterministic top-K hits + the rank of the planted target.
//
//   STDOUT is kept byte-for-byte deterministic so demo/run_demo can diff it
//   against demo/expected_output.txt. Anything that varies run-to-run (timings)
//   goes to STDERR, which the demo shows but does not diff (PATTERNS.md §3).
//
// Code tour: start here, then kernels.cuh -> kernels.cu, then pharmacophore.h
//            (the shared scoring formula), then reference_cpu.cpp.
// ===========================================================================
#include <algorithm>   // std::partial_sort, std::min
#include <cstdio>
#include <numeric>     // std::iota
#include <string>
#include <vector>

#include "kernels.cuh"        // screen_gpu, ScreenData, Feature
#include "reference_cpu.h"    // load_screen, query_self_overlap, screen_cpu
#include "util/io.hpp"        // util::CpuTimer, util::max_abs_err

// Identify the program. Kept in sync with demo/expected_output.txt.
static const char* PROJECT_ID   = "2.33";
static const char* PROJECT_NAME = "Structure-Based Pharmacophore Modeling from MD Ensembles";

// Correctness tolerance. The CPU and GPU call the SAME score_molecule(), so they
// differ only by the GPU's fused-multiply-add vs. the host's separate mul/add in
// the exp() argument -- a ~1e-7 relative wobble on a score in [0,1]. 1e-5 is a
// safe, honest absolute tolerance (PATTERNS.md §4: same-ops single-precision).
static constexpr double TOLERANCE = 1.0e-5;
static constexpr int    TOP_K     = 5;

// Human-readable feature-type names for the report (index = FeatureType value).
static const char* FEATURE_NAME[FEAT_NUM_TYPES] = {
    "donor", "acceptor", "hydrophobe", "aromatic", "pos-charge", "neg-charge"
};

// Indices of the TOP_K highest scores. Ties break toward the LOWER index so the
// ordering is deterministic regardless of sort implementation (stdout is diffed).
static std::vector<int> top_k(const std::vector<float>& score, int k) {
    std::vector<int> idx(score.size());
    std::iota(idx.begin(), idx.end(), 0);
    const int kk = std::min<int>(k, static_cast<int>(idx.size()));
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
        [&](int a, int b) { return score[a] != score[b] ? score[a] > score[b] : a < b; });
    idx.resize(kk);
    return idx;
}

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/pharmacophore_sample.txt";
    ScreenData s;
    try {
        s = load_screen(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int n_query = static_cast<int>(s.query.size());

    // The query self-overlap O_qq is constant across the library, so compute it
    // ONCE and reuse it for every molecule (host + device both receive it).
    const double self_qq = query_self_overlap(s);

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<float> score_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    screen_cpu(s, self_qq, score_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU screen (kernel timed) -------------------------------------
    std::vector<float> score_gpu;
    float gpu_kernel_ms = 0.0f;
    screen_gpu(s, self_qq, score_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    const double err = util::max_abs_err(score_cpu, score_gpu);
    const bool pass = err <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    const std::vector<int> best = top_k(score_gpu, TOP_K);
    // Rank of the planted target molecule (1-based) among all scores; ties toward
    // the lower index, matching top_k()'s tie rule.
    int target_rank = -1;
    if (s.target >= 0 && s.target < s.N) {
        const float ts = score_gpu[s.target];
        int better = 0;
        for (int i = 0; i < s.N; ++i)
            if (score_gpu[i] > ts || (score_gpu[i] == ts && i < s.target)) ++better;
        target_rank = better + 1;
    }

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("ensemble pharmacophore: %d features\n", n_query);
    for (int i = 0; i < n_query; ++i)
        std::printf("  feature[%d] %-10s at (%.2f, %.2f, %.2f) weight %.2f\n",
                    i, FEATURE_NAME[s.query[i].type],
                    s.query[i].x, s.query[i].y, s.query[i].z, s.query[i].weight);
    std::printf("screen: 1 pharmacophore vs %d library molecules\n", s.N);
    std::printf("top-%d hits (ROCS-style color Tanimoto):\n", static_cast<int>(best.size()));
    for (std::size_t r = 0; r < best.size(); ++r)
        std::printf("  #%zu  mol[%d]  score = %.6f\n", r + 1, best[r], score_gpu[best[r]]);
    if (s.target >= 0)
        std::printf("planted target mol[%d] score = %.6f, rank = %d of %d\n",
                    s.target, score_gpu[s.target], target_rank, s.N);
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-05)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d molecules, %d query features, %zu library features)\n",
                 path.c_str(), s.N, n_query, s.lib_feats.size());
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernel: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- the GPU's edge grows with library size; a real "
                         "screen scores ONE pharmacophore vs 10^6-10^9 conformers.\n");
    std::fprintf(stderr, "[verify] max_abs_err = %.3e  (tolerance %.1e)\n", err, TOLERANCE);

    return pass ? 0 : 1;
}
