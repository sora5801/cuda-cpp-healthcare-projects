// ===========================================================================
// src/main.cu  --  Entry point: load data, run CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 2.21 : Protein-Nucleic Acid Docking & Co-Folding (reduced-scope).
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the docking problem (protein + nucleic-acid ligand + pose grid +
//      scoring params) from data/sample (or the path given on argv[1]).
//   2. Compute the CPU reference (reference_cpu.cpp: dock_cpu)  -> trusted scores.
//   3. Compute the GPU result    (kernels.cu: dock_gpu)         -> the thing taught.
//   4. VERIFY: assert GPU == CPU EXACTLY (integer scores, tolerance 0).
//   5. REPORT: the best pose + a ranked shortlist to stdout; timing to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Anything that varies run-to-run (timings) goes to
//   STDERR, which the demo shows but does not diff (PATTERNS.md sec 3).
//
// READ THIS FIRST in the code tour, then docking_core.h (the physics),
// reference_cpu.* (the baseline), kernels.* (the GPU twin). The "why" is in
// ../THEORY.md; the data format is in ../data/README.md.
// ===========================================================================
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "kernels.cuh"        // dock_gpu (GPU path), DockingProblem, decode_pose
#include "reference_cpu.h"    // load_problem, dock_cpu (CPU baseline)
#include "util/io.hpp"        // util::CpuTimer

// These two tokens identify the program. They MUST stay in sync with
// demo/expected_output.txt (which is captured from a real run).
static const char* PROJECT_ID   = "2.21";
static const char* PROJECT_NAME = "Protein-Nucleic Acid Docking & Co-Folding";

// How many top poses to list in the deterministic shortlist.
static constexpr int TOP_K = 5;

// ---------------------------------------------------------------------------
// count_mismatch: the number of poses whose GPU score differs from the CPU
//   score. Because both backends run the SAME integer score_pose(), this MUST
//   be 0 -- it is our exact correctness gate (PATTERNS.md sec 4, "exact"). We
//   report the count (not a float error) so a single wrong pose is visible.
//   Returns the count and, via out-params, the first differing pose for a
//   helpful diagnostic.
// ---------------------------------------------------------------------------
static long long count_mismatch(const std::vector<int64_t>& a,
                                const std::vector<int64_t>& b,
                                long long& first_bad) {
    first_bad = -1;
    if (a.size() != b.size()) return (long long)a.size() + (long long)b.size();
    long long bad = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            if (first_bad < 0) first_bad = (long long)i;
            ++bad;
        }
    }
    return bad;
}

// ---------------------------------------------------------------------------
// top_k: indices of the TOP_K highest scores, ties broken by LOWER pose index
//   so the ranking is fully deterministic (PATTERNS.md sec 3). partial_sort on
//   an index vector -> O(n log k).
// ---------------------------------------------------------------------------
static std::vector<long long> top_k(const std::vector<int64_t>& score, int k) {
    std::vector<long long> idx(score.size());
    std::iota(idx.begin(), idx.end(), 0LL);                 // 0,1,2,...,n-1
    const int kk = std::min<int>(k, (int)idx.size());
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
        [&](long long i, long long j) {
            if (score[(std::size_t)i] != score[(std::size_t)j])
                return score[(std::size_t)i] > score[(std::size_t)j];  // higher first
            return i < j;                                              // tie -> lower idx
        });
    idx.resize(kk);
    return idx;
}

int main(int argc, char** argv) {
    // ---- 1. Load the problem ------------------------------------------------
    const std::string path = (argc > 1) ? argv[1]
                                        : "data/sample/complex_sample.txt";
    DockingProblem prob;
    try {
        prob = load_problem(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const long long N = prob.n_poses();

    // ---- 2. CPU reference (timed) ------------------------------------------
    std::vector<int64_t> score_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    dock_cpu(prob, score_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU result (kernel timed inside the wrapper) -------------------
    std::vector<int64_t> score_gpu;
    float gpu_kernel_ms = 0.0f;
    dock_gpu(prob, score_gpu, &gpu_kernel_ms);

    // ---- 4. Verify (EXACT integer equality) --------------------------------
    long long first_bad = -1;
    const long long bad = count_mismatch(score_cpu, score_gpu, first_bad);
    const bool pass = (bad == 0);

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    // Rank by the GPU scores (identical to CPU when pass). Decode the winner so
    // the learner sees the actual rigid transform, not just a number.
    const std::vector<long long> best = top_k(score_gpu, TOP_K);

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("rigid-body docking: protein (%d atoms) vs nucleic-acid ligand (%d atoms)\n",
                prob.Np(), prob.Nl());
    std::printf("pose space: %d orientations x %dx%dx%d translations = %lld poses\n",
                prob.n_rot(), prob.grid.nx, prob.grid.ny, prob.grid.nz, N);
    std::printf("top-%d poses by interface score:\n", (int)best.size());
    for (std::size_t r = 0; r < best.size(); ++r) {
        const long long p = best[r];
        int32_t tx, ty, tz;
        const int rot = decode_pose(p, prob.grid, prob.n_rot(), tx, ty, tz);
        // Print translations in Angstrom (divide the fixed-point units out) as
        // exact decimals -- the values are exact multiples of the grid step.
        std::printf("  #%zu  pose %lld  score = %lld  rot = %d  "
                    "t = (%.3f, %.3f, %.3f) A\n",
                    r + 1, p, (long long)score_gpu[(std::size_t)p], rot,
                    (double)tx / COORD_SCALE, (double)ty / COORD_SCALE,
                    (double)tz / COORD_SCALE);
    }
    std::printf("RESULT: %s (GPU matches CPU exactly: %lld/%lld poses agree)\n",
                pass ? "PASS" : "FAIL", N - bad, N);

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (Np=%d, Nl=%d, poses=%lld)\n",
                 path.c_str(), prob.Np(), prob.Nl(), N);
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny sample is "
                         "dominated by launch/copy overhead; the GPU's edge grows "
                         "with the pose space and atom counts.\n");
    if (!pass) {
        std::fprintf(stderr, "[verify] FIRST mismatch at pose %lld: cpu=%lld gpu=%lld\n",
                     first_bad,
                     (long long)score_cpu[(std::size_t)first_bad],
                     (long long)score_gpu[(std::size_t)first_bad]);
    } else {
        std::fprintf(stderr, "[verify] all %lld pose scores identical "
                             "(integer arithmetic -> exact, tolerance 0)\n", N);
    }

    // Exit code feeds the demo's pass/fail gate.
    return pass ? 0 : 1;
}
