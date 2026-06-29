// ===========================================================================
// src/main.cu  --  Entry point: load panel, score, verify, report selectivity
// ---------------------------------------------------------------------------
// Project 1.29 : Kinase Selectivity Panel Scoring
//
// The 5-step shape every project in this repo follows:
//   1. Load the problem (a compound + a panel of N kinases from data/sample).
//   2. CPU reference  (reference_cpu.cpp)  -> trusted per-kinase pK + S-count.
//   3. GPU scoring    (kernels.cu)         -> the thing being taught.
//   4. VERIFY: GPU agrees with CPU EXACTLY (integer pK and hit flags match).
//   5. REPORT: deterministic selectivity summary + top-K hits to stdout;
//              timing + verification detail to stderr.
//
// STDOUT is byte-for-byte deterministic (diffed by demo/run_demo against
// demo/expected_output.txt). Run-to-run timings go to STDERR (shown, not diffed).
//
// Code tour: start here, then kernels.cuh -> kernels.cu, then selectivity_core.h
// (the shared physics) and reference_cpu.* (the baseline + loader).
// ===========================================================================
#include <algorithm>   // std::partial_sort, std::min
#include <cstdint>
#include <cstdio>
#include <numeric>     // std::iota
#include <string>
#include <vector>

#include "kernels.cuh"        // score_panel_gpu, KinasePanel, NFEAT
#include "reference_cpu.h"    // load_panel, score_panel_cpu
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "1.29";
static const char* PROJECT_NAME = "Kinase Selectivity Panel Scoring";

// How many of the most-potently-bound kinases to list in the report.
static constexpr int TOP_K = 5;

// ---------------------------------------------------------------------------
// top_k_by_pK : indices of the TOP_K kinases by predicted affinity (pK), ties
// broken by LOWER index so the ranking is fully deterministic (stdout must be
// byte-identical run to run). partial_sort on an index vector -> O(n log k).
// ---------------------------------------------------------------------------
static std::vector<int> top_k_by_pK(const std::vector<int32_t>& pK, int k) {
    std::vector<int> idx(pK.size());
    std::iota(idx.begin(), idx.end(), 0);                 // 0,1,2,...,n-1
    const int kk = std::min<int>(k, static_cast<int>(idx.size()));
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
        [&](int a, int b) {
            if (pK[a] != pK[b]) return pK[a] > pK[b];     // higher affinity first
            return a < b;                                 // tie -> lower index
        });
    idx.resize(kk);
    return idx;
}

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/kinase_panel_sample.txt";
    KinasePanel panel;
    try {
        panel = load_panel(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<int32_t> pK_cpu, hit_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    const int32_t s_count_cpu = score_panel_cpu(panel, pK_cpu, hit_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU scoring (kernel timed inside the wrapper) -----------------
    std::vector<int32_t> pK_gpu, hit_gpu;
    float gpu_kernel_ms = 0.0f;
    const int32_t s_count_gpu = score_panel_gpu(panel, pK_gpu, hit_gpu, &gpu_kernel_ms);

    // ---- 4. Verify (EXACT integer agreement) -------------------------------
    // Because CPU and GPU run the identical __host__ __device__ integer physics,
    // every predicted pK, every hit flag, and the S-count must match exactly.
    bool pass = (pK_cpu == pK_gpu) && (hit_cpu == hit_gpu) && (s_count_cpu == s_count_gpu);
    int mismatches = 0;
    for (std::size_t i = 0; i < pK_cpu.size(); ++i)
        if (pK_cpu[i] != pK_gpu[i] || hit_cpu[i] != hit_gpu[i]) ++mismatches;

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    // S-score = (# kinases bound at pK >= 6.000) / (panel size). Smaller = more
    // selective. We print it as an integer ratio plus a fixed-3-decimal fraction
    // computed by INTEGER arithmetic (s_count*1000 / n) so it never varies.
    const int   n        = panel.n;
    const int   s_count  = static_cast<int>(s_count_gpu);
    const int   s_milli  = (n > 0) ? (s_count * 1000) / n : 0;   // S-score * 1000, integer
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("panel: 1 compound vs %d kinases (%d-feature interaction fingerprint)\n", n, NFEAT);
    std::printf("S-score(pK>=6.000) = %d/%d = %d.%03d  (lower = more selective)\n",
                s_count, n, s_milli / 1000, s_milli % 1000);
    std::printf("top-%d most potently bound kinases:\n", TOP_K);
    const std::vector<int> best = top_k_by_pK(pK_gpu, TOP_K);
    for (std::size_t r = 0; r < best.size(); ++r) {
        const int i = best[r];
        const int32_t pk = pK_gpu[static_cast<std::size_t>(i)];
        // Render the fixed-point milli-pK as a normal pK with 3 decimals.
        std::printf("  #%zu  %-10s  pK = %d.%03d  %s\n",
                    r + 1, panel.names[static_cast<std::size_t>(i)].c_str(),
                    pk / 1000, pk % 1000, hit_gpu[static_cast<std::size_t>(i)] ? "[HIT]" : "");
    }
    std::printf("RESULT: %s (GPU matches CPU exactly: per-kinase pK, hit flags, S-count)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (n=%d kinases, NFEAT=%d)\n", path.c_str(), n, NFEAT);
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny panel is dominated by "
                         "launch/copy overhead; the GPU wins at kinome scale (500+ kinases x "
                         "many compounds).\n");
    std::fprintf(stderr, "[verify] exact-match mismatches = %d / %d kinases  (tolerance = 0, "
                         "integer physics)\n", mismatches, n);

    return pass ? 0 : 1;
}
