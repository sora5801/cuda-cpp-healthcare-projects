// ===========================================================================
// src/main.cu  --  Entry point: load antibodies, screen, verify, report
// ---------------------------------------------------------------------------
// Project 2.15 : Antibody Structure Prediction  (reduced-scope: CDR screening)
//
// The 5-step shape every project in this repo follows:
//   1. Load the problem (a query antibody + n library antibodies' CDR loops).
//   2. CPU reference  (reference_cpu.cpp)  -> trusted integer scores.
//   3. GPU screen     (kernels.cu)         -> the thing being taught.
//   4. VERIFY: GPU agrees with CPU exactly (integer math, tolerance 0).
//   5. REPORT: deterministic top-K hits to stdout; timing + detail to stderr.
//
// STDOUT is byte-for-byte deterministic (diffed by demo/run_demo against
// demo/expected_output.txt); run-to-run timings go to STDERR.
//
// SCOPE HONESTY: this is NOT structure prediction (no 3-D coordinates). It is the
// library-SCREENING step that sits under high-throughput antibody work: rank
// library antibodies by CDR similarity to a query, weighting CDR-H3 most. See
// README "Limitations & honesty" and THEORY "Where this sits in the real world".
//
// Code tour: start here, then antibody.h -> kernels.cuh -> kernels.cu, then
// reference_cpu.* for the baseline.
// ===========================================================================
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "antibody.h"         // AB_NUM_CDRS, ab_cdr_score (for the per-CDR breakdown)
#include "kernels.cuh"        // score_gpu, AntibodyLibrary
#include "reference_cpu.h"    // load_library, score_cpu
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "2.15";
static const char* PROJECT_NAME = "Antibody Structure Prediction (reduced: CDR screening)";

// Tolerance: CPU and GPU both call the shared integer scoring core ab_cdr_score,
// so they agree EXACTLY. We require zero difference -- any nonzero gap is a bug.
static constexpr int32_t TOLERANCE = 0;
static constexpr int     TOP_K     = 5;       // how many best hits to report

// max_abs_err_int: largest |a[i] - b[i]| over two equal-length int32 vectors.
// Returns -1 (an impossible distance) on a length mismatch so a shape bug can
// never masquerade as agreement.
static long long max_abs_err_int(const std::vector<int32_t>& a,
                                 const std::vector<int32_t>& b) {
    if (a.size() != b.size()) return -1;
    long long worst = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        long long d = std::llabs(static_cast<long long>(a[i]) - static_cast<long long>(b[i]));
        if (d > worst) worst = d;
    }
    return worst;
}

// top_k: indices of the TOP_K largest scores, ties broken by LOWER index so the
// ranking is fully deterministic. Uses partial_sort on an index vector.
static std::vector<int> top_k(const std::vector<int32_t>& score, int k) {
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
    const std::string path = (argc > 1) ? argv[1] : "data/sample/antibodies_sample.txt";
    AntibodyLibrary ab;
    int truncated = 0;
    try {
        ab = load_library(path, &truncated);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<int32_t> score_ref;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    score_cpu(ab, score_ref);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU screen (kernel timed inside the wrapper) ------------------
    std::vector<int32_t> score_dev;
    float gpu_kernel_ms = 0.0f;
    score_gpu(ab, score_dev, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    const long long err = max_abs_err_int(score_ref, score_dev);
    const bool pass = (err == TOLERANCE);

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    const std::vector<int32_t>& s = score_dev;            // report the GPU scores
    const std::vector<int> best = top_k(s, TOP_K);
    std::printf("%s\n", PROJECT_NAME);
    std::printf("catalog ID %s -- reduced-scope teaching version\n", PROJECT_ID);
    std::printf("query antibody: %s\n", ab.query_name.c_str());
    std::printf("screened %d library antibodies by CDR-weighted BLOSUM62 similarity\n", ab.n);
    std::printf("(CDR-H3 weighted x%d; higher score = more similar CDRs)\n", ab_cdr_weight(2));
    std::printf("top-%d hits:\n", static_cast<int>(best.size()));
    for (std::size_t r = 0; r < best.size(); ++r) {
        const int j = best[r];
        // Break the score down per-CDR so the learner SEES that CDR-H3 dominates.
        const uint8_t* lib_j = &ab.lib[static_cast<std::size_t>(j) * AB_RECORD_LEN];
        const int h3 = ab_cdr_weight(2) *
            // recompute just the H3 contribution for display (index 2)
            [&]{ int x=0; const int off=2*AB_CDR_LEN;
                 for (int p=0;p<AB_CDR_LEN;++p) x+=ab_blosum62(ab.query[off+p], lib_j[off+p]);
                 return x; }();
        std::printf("  #%zu  %-10s  score = %5d  (CDR-H3 contributes %4d)\n",
                    r + 1, ab.names[j].c_str(), s[j], h3);
    }
    std::printf("RESULT: %s (GPU matches CPU exactly, integer scores)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (query + %d library antibodies, %d CDRs each)\n",
                 path.c_str(), ab.n, AB_NUM_CDRS);
    if (truncated > 0)
        std::fprintf(stderr, "[data]   warning: %d CDR token(s) exceeded %d residues and were truncated\n",
                     truncated, AB_CDR_LEN);
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny sample is dominated by "
                         "launch/copy overhead; the GPU wins at library scale (millions of antibodies).\n");
    std::fprintf(stderr, "[verify] max_abs_err = %lld  (tolerance %d, exact integer match)\n",
                 err, TOLERANCE);

    return pass ? 0 : 1;
}
