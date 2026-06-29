// ===========================================================================
// src/main.cu  --  Entry point: load KG, score, verify, rank, report
// ---------------------------------------------------------------------------
// Project 1.19 : Network / Polypharmacology Modeling
//
// The 5-step shape every project in this repo follows:
//   1. Load the problem (a query drug + relation + N protein tails, from
//      data/sample as TransE embeddings).
//   2. CPU reference  (reference_cpu.cpp)  -> trusted plausibility scores.
//   3. GPU scoring    (kernels.cu)         -> the thing being taught.
//   4. VERIFY: GPU agrees with CPU EXACTLY (the per-tail math is shared, so the
//      float ops are identical -> tolerance 0).
//   5. REPORT: deterministic top-K predicted targets + a recovery metric to
//      stdout; timing to stderr.
//
// STDOUT is byte-for-byte deterministic (diffed by demo/run_demo against
// demo/expected_output.txt); run-to-run timings go to STDERR.
//
// Code tour: start here, then transe.h (the scoring math), kernels.cuh ->
// kernels.cu, then reference_cpu.* (loader + CPU baseline). See ../THEORY.md.
// ===========================================================================
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "kernels.cuh"        // transe_score_gpu, KnowledgeGraph
#include "reference_cpu.h"    // load_knowledge_graph, transe_score_cpu
#include "util/io.hpp"        // util::CpuTimer, util::max_abs_err

static const char* PROJECT_ID   = "1.19";
static const char* PROJECT_NAME = "Network / Polypharmacology Modeling";

// Tolerance: CPU and GPU call the SAME transe_score() (transe.h) over the same
// data in the same loop order. They would be bit-identical EXCEPT for one real,
// teachable effect: nvcc contracts the device-side  acc + diff*diff  into a
// single fused multiply-add (FMA) by default, while the host compiler emits a
// separate multiply then add. FMA keeps more intermediate precision, so the two
// sums differ by ~1e-7 per accumulation -- a genuine GPU-vs-host divergence, not
// a bug (PATTERNS.md sec 4; see THEORY.md "Numerical considerations"). We verify
// to a small, physically-negligible tolerance and say so honestly. (To force
// bit-identical results you would disable FMA contraction with nvcc --fmad=false
// -- discussed in THEORY -- but the default-FMA behavior is what real GPU code
// ships, so we teach it rather than hide it.)
static constexpr double TOLERANCE = 1.0e-5;
static constexpr int    TOP_K     = 5;       // how many top predicted targets to show

// Return the indices of the TOP_K largest scores, ties broken by lower index
// (so the ranking is deterministic). Uses partial_sort on an index vector --
// O(n log k), and stable under ties by the explicit index comparison.
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
    const std::string path = (argc > 1) ? argv[1]
                                        : "data/sample/kg_embeddings_sample.txt";
    KnowledgeGraph kg;
    try {
        kg = load_knowledge_graph(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<float> score_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    transe_score_cpu(kg, score_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU scoring (kernel timed inside the wrapper) -----------------
    std::vector<float> score_gpu;
    float gpu_kernel_ms = 0.0f;
    transe_score_gpu(kg, score_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    const double err = util::max_abs_err(score_cpu, score_gpu);
    const bool pass = err <= TOLERANCE;

    // ---- Ranking + recovery metric (computed from the GPU scores) ---------
    // The top-K highest-scoring tails are the predicted (off-)targets. We then
    // measure RECOVERY: how many of the synthetic ground-truth targets landed in
    // the top-K -- a self-check that the method found the answer we embedded.
    const std::vector<int> best = top_k(score_gpu, TOP_K);
    int recovered = 0;
    for (int gt : kg.true_targets) {
        for (int b : best) if (b == gt) { ++recovered; break; }
    }
    const int n_true = static_cast<int>(kg.true_targets.size());

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("TransE link prediction: 1 query drug vs %d protein tails (dim=%d)\n",
                kg.n, kg.dim);
    std::printf("top-%d predicted targets (protein index : TransE score):\n",
                static_cast<int>(best.size()));
    for (std::size_t rk = 0; rk < best.size(); ++rk)
        std::printf("  #%zu  protein[%d]  score = %.6f\n",
                    rk + 1, best[rk], score_gpu[best[rk]]);
    std::printf("recovery: %d / %d ground-truth targets in top-%d\n",
                recovered, n_true, static_cast<int>(best.size()));
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-05)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (n=%d tails, dim=%d; SYNTHETIC)\n",
                 path.c_str(), kg.n, kg.dim);
    std::fprintf(stderr, "[data]   ground-truth targets:");
    for (int gt : kg.true_targets) std::fprintf(stderr, " %d", gt);
    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny KG is dominated by "
                         "launch/copy overhead; the GPU wins at graph scale (10^4-10^6 entities).\n");
    std::fprintf(stderr, "[verify] max_abs_err = %.3e  (tolerance %.1e)\n", err, TOLERANCE);

    return pass ? 0 : 1;
}
