// ===========================================================================
// src/main.cu  --  Entry point: load backbone, design, verify, report
// ---------------------------------------------------------------------------
// Project 2.10 : Protein Design / Inverse Folding Inference
//
// The 5-step shape every project in this repo follows:
//   1. Load the problem (a protein backbone + native sequence from data/sample).
//   2. CPU reference  (reference_cpu.cpp)  -> trusted design.
//   3. GPU design     (kernels.cu)         -> the thing being taught.
//   4. VERIFY: GPU agrees with CPU EXACTLY (integer math, shared score core).
//   5. REPORT: deterministic design + recovery to stdout; timing to stderr.
//
// STDOUT is byte-for-byte deterministic (diffed by demo/run_demo against
// demo/expected_output.txt); run-to-run timings go to STDERR.
//
// Code tour: start here, then inverse_folding.h (the shared score core),
// reference_cpu.h/.cpp (the baseline), then kernels.cuh -> kernels.cu (the GPU).
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // design_gpu, Backbone, DesignResult
#include "reference_cpu.h"    // load_backbone, design_cpu, recovery_percent, ...
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "2.10";
static const char* PROJECT_NAME = "Protein Design / Inverse Folding Inference";

// agrees_exactly: do two DesignResults match element-for-element? Every field is
// an exact integer (neighbor counts, amino-acid indices, integer scores) and the
// GPU uses the identical shared scoring core as the CPU, so we demand EXACT
// equality -- no floating tolerance (PATTERNS.md sec 4, the "==" case). A
// mismatch would signal a real bug (a race, a bad index, a divergent formula).
static bool agrees_exactly(const DesignResult& a, const DesignResult& b) {
    if (a.neighbors.size() != b.neighbors.size()) return false;
    for (std::size_t i = 0; i < a.neighbors.size(); ++i) {
        if (a.neighbors[i] != b.neighbors[i]) return false;
        if (a.designed[i]  != b.designed[i])  return false;
        if (a.score[i]     != b.score[i])     return false;
    }
    return true;
}

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1]
                                        : "data/sample/backbone_sample.txt";
    Backbone bb;
    try {
        bb = load_backbone(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int L = bb.size();

    // ---- 2. CPU reference (timed) -----------------------------------------
    DesignResult cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    design_cpu(bb, cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU design (kernels timed inside the wrapper) -----------------
    DesignResult gpu;
    float gpu_kernel_ms = 0.0f;
    design_gpu(bb, gpu, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    const bool pass = agrees_exactly(cpu, gpu);

    // Derived, deterministic reporting quantities (computed from the GPU design).
    const std::string designed_seq = sequence_string(gpu.designed);
    const std::string native_seq   = sequence_string(bb.native);
    const int recovery = recovery_percent(bb, gpu);   // % designed == native

    // Count buried vs exposed positions for a one-line structural summary.
    int buried = 0;
    for (int i = 0; i < L; ++i)
        if (gpu.neighbors[i] >= BURIAL_THRESHOLD) ++buried;

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("inverse folding (reduced-scope teaching model): design a sequence for a fixed backbone\n");
    std::printf("residues L = %d   buried (>=%d contacts) = %d   exposed = %d\n",
                L, BURIAL_THRESHOLD, buried, L - buried);
    std::printf("native   : %s\n", native_seq.c_str());
    std::printf("designed : %s\n", designed_seq.c_str());
    std::printf("native sequence recovery: %d%%\n", recovery);
    std::printf("RESULT: %s (GPU design matches CPU reference exactly)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (L=%d residues)\n", path.c_str(), L);
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernels: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny backbone is dominated by "
                         "launch/copy overhead; the O(L^2) burial step is where the GPU wins at "
                         "real protein/library scale.\n");
    std::fprintf(stderr, "[verify] exact integer match across neighbors/designed/score: %s\n",
                 pass ? "yes" : "NO");

    return pass ? 0 : 1;
}
