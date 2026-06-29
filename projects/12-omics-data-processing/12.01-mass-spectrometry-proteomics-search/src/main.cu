// ===========================================================================
// src/main.cu  --  Entry point: spectral library search, verify, report
// ---------------------------------------------------------------------------
// Project 12.01 : Mass-Spectrometry Proteomics Search
//
// 5-step shape:
//   1. Load the query + library spectra (data/sample).
//   2. CPU reference cosine scores (reference_cpu.cpp).
//   3. GPU cosine scores (kernels.cu): query in constant memory, 1 thread/library.
//   4. VERIFY: GPU scores match CPU within tolerance.
//   5. REPORT: deterministic top-K matches + the rank of the true target.
//
// Code tour: start here, then kernels.cuh -> kernels.cu, then reference_cpu.cpp.
// ===========================================================================
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "kernels.cuh"        // cosine_gpu, SpectralData, MAX_BINS
#include "reference_cpu.h"    // load_spectra, compute_norms, cosine_cpu
#include "util/io.hpp"        // util::CpuTimer, util::max_abs_err

static const char* PROJECT_ID   = "12.1";
static const char* PROJECT_NAME = "Mass-Spectrometry Proteomics Search";

static constexpr double TOLERANCE = 1.0e-5;   // float cosine; double dot product
static constexpr int    TOP_K     = 5;

// Indices of the TOP_K highest scores (ties -> lower index), via partial_sort.
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
    const std::string path = (argc > 1) ? argv[1] : "data/sample/spectra_sample.txt";
    SpectralData s;
    try {
        s = load_spectra(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    double qnorm = 0.0;
    std::vector<double> libnorm;
    compute_norms(s, qnorm, libnorm);

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<float> score_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    cosine_cpu(s, qnorm, libnorm, score_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU search (kernel timed) -------------------------------------
    std::vector<float> score_gpu;
    float gpu_kernel_ms = 0.0f;
    cosine_gpu(s, qnorm, libnorm, score_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    const double err = util::max_abs_err(score_cpu, score_gpu);
    const bool pass = err <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    const std::vector<int> best = top_k(score_gpu, TOP_K);
    // Rank of the known target spectrum (1-based) among all scores.
    int target_rank = -1;
    if (s.target >= 0 && s.target < s.N) {
        const float ts = score_gpu[s.target];
        int better = 0;
        for (int i = 0; i < s.N; ++i)
            if (score_gpu[i] > ts || (score_gpu[i] == ts && i < s.target)) ++better;
        target_rank = better + 1;
    }

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("spectral search: 1 query vs %d library spectra (%d bins)\n", s.N, s.bins);
    std::printf("top-%d matches:\n", static_cast<int>(best.size()));
    for (std::size_t r = 0; r < best.size(); ++r)
        std::printf("  #%zu  lib[%d]  cosine = %.6f\n", r + 1, best[r], score_gpu[best[r]]);
    if (s.target >= 0)
        std::printf("true target lib[%d] cosine = %.6f, rank = %d of %d\n",
                    s.target, score_gpu[s.target], target_rank, s.N);
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-05)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d library spectra, %d bins)\n", path.c_str(), s.N, s.bins);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernel: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- speed-up grows with library size; real searches scan "
                         "10^6 peptides x 10^5 spectra.\n");
    std::fprintf(stderr, "[verify] max_abs_err = %.3e  (tolerance %.1e)\n", err, TOLERANCE);

    return pass ? 0 : 1;
}
