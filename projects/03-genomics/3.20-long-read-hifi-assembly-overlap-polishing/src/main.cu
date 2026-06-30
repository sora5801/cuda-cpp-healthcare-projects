// ===========================================================================
// src/main.cu  --  Entry point: load sketches, overlap (CPU+GPU), verify, report
// ---------------------------------------------------------------------------
// Project 3.20 : Long-Read HiFi Assembly Overlap & Polishing
//
// The 5-step shape every project in this repo follows:
//   1. Load the problem (N reads' minimiser sketches from data/sample).
//   2. CPU reference  (reference_cpu.cpp)  -> trusted overlap scores.
//   3. GPU all-vs-all (kernels.cu)         -> the thing being taught.
//   4. VERIFY: GPU agrees with CPU EXACTLY (integer scores; tolerance == 0).
//   5. REPORT: deterministic top-K overlapping read pairs to stdout; timing to stderr.
//
// STDOUT is byte-for-byte deterministic (diffed by demo/run_demo against
// demo/expected_output.txt); run-to-run timings go to STDERR.
//
// Code tour: start here, then overlap_core.h -> reference_cpu.h ->
// reference_cpu.cpp -> kernels.cuh -> kernels.cu.
// ===========================================================================
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "kernels.cuh"        // overlap_gpu, ReadSet, OverlapResult
#include "reference_cpu.h"    // load_reads, overlap_cpu
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "3.20";
static const char* PROJECT_NAME = "Long-Read HiFi Assembly Overlap & Polishing";

// Tolerance: chain scores and anchor counts are INTEGERS produced by the SAME
// link function on both processors, so CPU and GPU must agree EXACTLY. We verify
// element-for-element equality -- the strongest possible check (PATTERNS.md sec 4).
static constexpr int  TOP_K = 5;     // how many strongest overlaps to report

// Return indices of the TOP_K results by (score desc, then read_i asc, read_j asc)
// so the ranking is fully deterministic even when scores tie.
static std::vector<int> top_k_overlaps(const std::vector<OverlapResult>& ov, int k) {
    std::vector<int> idx(ov.size());
    std::iota(idx.begin(), idx.end(), 0);
    const int kk = std::min<int>(k, static_cast<int>(idx.size()));
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
        [&](int a, int b) {
            if (ov[a].score != ov[b].score) return ov[a].score > ov[b].score;
            if (ov[a].read_i != ov[b].read_i) return ov[a].read_i < ov[b].read_i;
            return ov[a].read_j < ov[b].read_j;
        });
    idx.resize(kk);
    return idx;
}

// Element-for-element equality of the two result arrays. Returns the number of
// mismatching pairs (0 == perfect agreement). Also returns the first mismatch
// for a helpful diagnostic on stderr.
static int count_mismatches(const std::vector<OverlapResult>& a,
                            const std::vector<OverlapResult>& b, int* first) {
    if (a.size() != b.size()) { if (first) *first = 0; return -1; }
    int bad = 0, first_bad = -1;
    for (std::size_t k = 0; k < a.size(); ++k) {
        if (a[k].score != b[k].score || a[k].n_anchors != b[k].n_anchors ||
            a[k].read_i != b[k].read_i || a[k].read_j != b[k].read_j) {
            if (first_bad < 0) first_bad = static_cast<int>(k);
            ++bad;
        }
    }
    if (first) *first = first_bad;
    return bad;
}

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/reads_sample.txt";
    ReadSet rs;
    try {
        rs = load_reads(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<OverlapResult> cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    overlap_cpu(rs, cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU all-vs-all overlap (kernel timed inside the wrapper) ------
    std::vector<OverlapResult> gpu;
    float gpu_kernel_ms = 0.0f;
    overlap_gpu(rs, gpu, &gpu_kernel_ms);

    // ---- 4. Verify (exact) -------------------------------------------------
    int first_bad = -1;
    const int mism = count_mismatches(cpu, gpu, &first_bad);
    const bool pass = (mism == 0);

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    // Count how many pairs cleared a "candidate overlap" bar (>= 3 chained
    // anchors of integer score) -- a deterministic summary of the overlap graph.
    int candidate_pairs = 0;
    for (const auto& r : gpu) if (r.score >= 3) ++candidate_pairs;

    const std::vector<int> best = top_k_overlaps(gpu, TOP_K);
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("all-vs-all overlap: %d reads -> %lld ordered pairs scored\n",
                rs.n_reads, rs.num_pairs());
    std::printf("candidate overlaps (chain score >= 3): %d\n", candidate_pairs);
    std::printf("top-%d overlaps (by chain score):\n", static_cast<int>(best.size()));
    for (std::size_t r = 0; r < best.size(); ++r) {
        const OverlapResult& o = gpu[best[r]];
        std::printf("  #%zu  read %d <-> read %d   score=%d  anchors=%d\n",
                    r + 1, o.read_i, o.read_j, o.score, o.n_anchors);
    }
    std::printf("RESULT: %s (GPU matches CPU exactly: %lld/%lld pairs identical)\n",
                pass ? "PASS" : "FAIL",
                rs.num_pairs() - (mism < 0 ? rs.num_pairs() : mism), rs.num_pairs());

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (N=%d reads, %zu minimisers total)\n",
                 path.c_str(), rs.n_reads, rs.mins.size());
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny sample is dominated by "
                         "launch/copy overhead; the GPU's O(N^2)-pairs edge grows with read count.\n");
    if (mism > 0)
        std::fprintf(stderr, "[verify] %d mismatching pair(s); first at slot %d\n", mism, first_bad);
    else
        std::fprintf(stderr, "[verify] all %lld pairs identical (exact integer agreement)\n",
                     rs.num_pairs());

    return pass ? 0 : 1;
}
