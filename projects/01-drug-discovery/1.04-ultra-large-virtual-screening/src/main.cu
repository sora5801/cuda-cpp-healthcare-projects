// ===========================================================================
// src/main.cu  --  Entry point: load library, screen, verify, report
// ---------------------------------------------------------------------------
// Project 1.4 : Ultra-Large Virtual Screening
//
// The 5-step shape every project in this repo follows:
//   1. Load the problem (a target + N library ligands from data/sample).
//   2. CPU reference  (reference_cpu.cpp)  -> trusted per-ligand scores.
//   3. GPU screen     (kernels.cu)         -> the thing being taught.
//   4. VERIFY: GPU agrees with CPU EXACTLY (integer scores -> tolerance 0).
//   5. REPORT: deterministic survivor count + top-K hits to stdout;
//      timing + the run-varying detail to stderr.
//
// STDOUT is byte-for-byte deterministic (diffed by demo/run_demo against
// demo/expected_output.txt); run-to-run timings go to STDERR.
//
// Code tour: start here, then screen_core.h (the shared math), kernels.cuh ->
// kernels.cu (GPU), then reference_cpu.* (CPU baseline). See ../THEORY.md.
// ===========================================================================
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "kernels.cuh"        // screen_gpu, LigandLibrary
#include "reference_cpu.h"    // load_library, screen_cpu
#include "screen_core.h"      // REJECTED sentinel
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "1.4";
static const char* PROJECT_NAME = "Ultra-Large Virtual Screening";

// Verification tolerance. Every score is an INTEGER produced by the identical
// shared score_ligand() on both sides (filter cascade + integer surrogate dock),
// so the CPU and GPU agree EXACTLY -- the strongest possible check, tolerance 0
// (PATTERNS.md sec 4). We compare the score vectors element-for-element.
static constexpr int TOP_K = 5;        // how many best hits to report

// ---------------------------------------------------------------------------
// count_mismatches: how many ligands got a different score on CPU vs GPU.
//   This is our headline correctness number. It MUST be 0: the two paths run the
//   same integer math, so any nonzero value signals a real bug (a layout error,
//   a divergent code path), not floating-point noise. Returns the count.
// ---------------------------------------------------------------------------
static int count_mismatches(const std::vector<int>& a, const std::vector<int>& b) {
    if (a.size() != b.size()) return -1;   // shape bug -> distinct, loud sentinel
    int diff = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) ++diff;
    return diff;
}

// ---------------------------------------------------------------------------
// top_k_hits: indices of the TOP_K highest-scoring ligands that PASSED the
// cascade (score != REJECTED), ties broken by LOWER index so the ranking is
// fully deterministic. We build an index list of survivors and partial_sort it.
//   score : per-ligand scores (REJECTED == failed the filter cascade)
//   k     : how many hits to return (clamped to the survivor count)
// ---------------------------------------------------------------------------
static std::vector<int> top_k_hits(const std::vector<int>& score, int k) {
    // Collect only the survivors (rejected ligands are never "hits").
    std::vector<int> idx;
    idx.reserve(score.size());
    for (int i = 0; i < static_cast<int>(score.size()); ++i)
        if (score[i] != REJECTED) idx.push_back(i);

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
                                        : "data/sample/ligands_sample.txt";
    LigandLibrary lib;
    try {
        lib = load_library(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<int> score_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    screen_cpu(lib, score_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU screen (kernel timed inside the wrapper) ------------------
    std::vector<int> score_gpu;
    float gpu_kernel_ms = 0.0f;
    screen_gpu(lib, score_gpu, &gpu_kernel_ms);

    // ---- 4. Verify (exact integer agreement) ------------------------------
    const int mism = count_mismatches(score_cpu, score_gpu);
    const bool pass = (mism == 0);

    // Count survivors (passed the cascade) using the verified GPU scores.
    int survivors = 0;
    for (int s : score_gpu) if (s != REJECTED) ++survivors;

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    const std::vector<int> best = top_k_hits(score_gpu, TOP_K);
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("screened %d ligands against 1 target (filter cascade + surrogate dock)\n",
                lib.n());
    std::printf("passed drug-likeness cascade: %d / %d\n", survivors, lib.n());
    std::printf("top-%d hits (by surrogate score):\n", static_cast<int>(best.size()));
    for (std::size_t r = 0; r < best.size(); ++r) {
        const Ligand& L = lib.ligands[best[r]];
        // Print the rank, the ligand index, its score, and the key descriptors so
        // the learner can SEE why it scored well (good feature overlap, on-target
        // size/logP). All integer -> byte-identical across runs.
        std::printf("  #%zu  ligand[%d]  score=%d  MW=%d  logP=%.2f  feat=0x%08X\n",
                    r + 1, best[r], score_gpu[best[r]],
                    L.mw, L.logp_x100 / 100.0, L.feat);
    }
    std::printf("RESULT: %s (GPU matches CPU exactly)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (n=%d ligands)\n", path.c_str(), lib.n());
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny sample is dominated by "
                         "launch/copy overhead; the GPU wins at campaign scale (billions).\n");
    std::fprintf(stderr, "[verify] mismatches = %d  (must be 0; integer scores -> exact)\n", mism);

    return pass ? 0 : 1;
}
