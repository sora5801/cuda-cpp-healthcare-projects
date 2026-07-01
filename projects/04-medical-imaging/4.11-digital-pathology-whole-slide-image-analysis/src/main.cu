// ===========================================================================
// src/main.cu  --  Entry point: attention-MIL forward pass on a WSI slide bag
// ---------------------------------------------------------------------------
// Project 4.11 : Digital Pathology / Whole-Slide Image Analysis
//
// 5-step shape (the shape EVERY project in this repo follows):
//   1. LOAD  the slide bag (data/sample): N tile features x FEAT_DIM.
//   2. CPU   reference attention-MIL forward pass (reference_cpu.cpp).
//   3. GPU   attention-MIL forward pass (kernels.cu): per-tile logits kernel +
//            softmax + fixed-point atomic pooling kernel + classifier.
//   4. VERIFY: GPU attention weights, pooled embedding, and slide probability
//            agree with the CPU within a documented tolerance.
//   5. REPORT: deterministic result -> stdout; timing/detail -> stderr.
//
//   STDOUT is byte-for-byte deterministic (fixed-precision prints) so
//   demo/run_demo can diff it against demo/expected_output.txt. Run-varying
//   numbers (timings, raw error magnitudes) go to STDERR, which the demo shows
//   but does not diff.
//
// Code tour: start here, then wsi.h (the per-tile math + fixed-point pooling),
// kernels.cuh -> kernels.cu (the GPU path), reference_cpu.cpp (the baseline).
// See ../THEORY.md for the "why".
// ===========================================================================
#include <algorithm>   // std::max
#include <cmath>       // std::fabs
#include <cstdio>
#include <string>
#include <utility>     // std::pair (for the attention ranking)
#include <vector>

#include "kernels.cuh"        // mil_forward_gpu
#include "reference_cpu.h"    // SlideBag, MilResult, load_slide, mil_forward_cpu, default_params
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "4.11";
static const char* PROJECT_NAME = "Digital Pathology / Whole-Slide Image Analysis";

// Verification tolerance. The attention logits use tanh()/exp(), whose DEVICE
// implementations differ from the HOST libm by ~1 ULP (~1e-16). That tiny
// difference propagates through the softmax and the fixed-point pooling, so the
// GPU and CPU agree to ~1e-9, not bit-for-bit. We verify to 1e-9 -- far tighter
// than anything that changes the clinical-style readout -- and say so honestly
// (see PATTERNS.md section 4 and THEORY.md "Numerical considerations").
static constexpr double TOLERANCE = 1.0e-9;

// Largest absolute difference between two equal-length double vectors.
static double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double worst = 0.0;
    const std::size_t n = (a.size() < b.size()) ? a.size() : b.size();
    for (std::size_t i = 0; i < n; ++i)
        worst = std::max(worst, std::fabs(a[i] - b[i]));
    if (a.size() != b.size()) worst = 1.0e30;   // shape mismatch -> force FAIL
    return worst;
}

int main(int argc, char** argv) {
    // ---- 1. Load ------------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/slide_sample.txt";
    SlideBag bag;
    try {
        bag = load_slide(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const AttnParams params = default_params();   // the frozen attention head

    // ---- 2. CPU reference (timed) ------------------------------------------
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    const MilResult cpu = mil_forward_cpu(bag, params);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU forward pass (kernels timed inside the wrapper) ------------
    float gpu_kernel_ms = 0.0f;
    const MilResult gpu = mil_forward_gpu(bag, params, &gpu_kernel_ms);

    // ---- 4. Verify ----------------------------------------------------------
    const double attn_diff  = max_abs_diff(cpu.attn, gpu.attn);
    const double embed_diff = max_abs_diff(cpu.embedding, gpu.embedding);
    const double prob_diff  = std::fabs(cpu.probability - gpu.probability);
    const bool   top_match  = (cpu.top_tile == gpu.top_tile);
    const bool   pass = (attn_diff <= TOLERANCE) && (embed_diff <= TOLERANCE)
                        && (prob_diff <= TOLERANCE) && top_match;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    // We print the GPU result (verified equal to the CPU to 1e-9) at FIXED
    // precision so the bytes are identical every run.
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("attention-MIL over a slide bag: %d tiles x %d features -> 1 slide score\n",
                bag.N, FEAT_DIM);

    // The pooled slide embedding z (the attention-weighted average tile feature).
    std::printf("slide embedding z =");
    for (int d = 0; d < FEAT_DIM; ++d) std::printf(" %.6f", gpu.embedding[d]);
    std::printf("\n");

    // The headline outputs: slide logit, tumor probability, and the top tile.
    std::printf("slide logit = %.6f   tumor probability = %.6f\n",
                gpu.slide_logit, gpu.probability);
    std::printf("top attention tile = %d   (weight %.6f)\n",
                gpu.top_tile, gpu.attn[gpu.top_tile]);

    // The five most-attended tiles (a tiny "attention heat map" ranking). We copy
    // (weight, index) pairs and partial-sort so the print is deterministic.
    std::vector<std::pair<double,int>> ranked(bag.N);
    for (int i = 0; i < bag.N; ++i) ranked[i] = {gpu.attn[i], i};
    const int show = (bag.N < 5) ? bag.N : 5;
    std::partial_sort(ranked.begin(), ranked.begin() + show, ranked.end(),
                      [](const std::pair<double,int>& x, const std::pair<double,int>& y) {
                          // Higher weight first; ties -> lower tile index (stable, deterministic).
                          if (x.first != y.first) return x.first > y.first;
                          return x.second < y.second;
                      });
    std::printf("top-%d attention tiles:", show);
    for (int k = 0; k < show; ++k)
        std::printf(" #%d=%.6f", ranked[k].second, ranked[k].first);
    std::printf("\n");

    // The slide-level call at the standard 0.5 decision threshold.
    const int predicted = (gpu.probability >= 0.5) ? 1 : 0;
    std::printf("slide call @0.5 = %s\n", predicted ? "TUMOR" : "benign");
    std::printf("RESULT: %s (GPU attention+embedding+probability match CPU within tol=1.0e-09)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%d tiles, %d features/tile", path.c_str(), bag.N, FEAT_DIM);
    if (bag.has_true_label)
        std::fprintf(stderr, ", ground-truth label = %d [%s]", bag.true_label,
                     bag.true_label ? "tumor" : "benign");
    std::fprintf(stderr, ")\n");
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernels: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- a real slide has 10^4-10^5 tiles and a\n"
                         "         cohort has millions; the GPU's edge grows with the tile count.\n");
    std::fprintf(stderr, "[verify] max attn diff = %.3e, max embed diff = %.3e, prob diff = %.3e, "
                         "top-tile match = %s (tol %.1e)\n",
                 attn_diff, embed_diff, prob_diff, top_match ? "yes" : "no", TOLERANCE);

    return pass ? 0 : 1;
}
