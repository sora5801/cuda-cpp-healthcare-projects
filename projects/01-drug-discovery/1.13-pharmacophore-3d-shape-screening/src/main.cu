// ===========================================================================
// src/main.cu  --  Entry point: load conformers, screen, verify, report
// ---------------------------------------------------------------------------
// Project 1.13 : Pharmacophore & 3D Shape Screening
//
// The 5-step shape every project in this repo follows:
//   1. Load the problem (a query + n library conformers from data/sample).
//   2. CPU reference  (reference_cpu.cpp)  -> trusted Shape Tanimoto scores.
//   3. GPU screen     (kernels.cu)         -> the thing being taught.
//   4. VERIFY: GPU agrees with CPU within tolerance.
//   5. REPORT: deterministic top-K ranking to stdout; timing + error to stderr.
//
// STDOUT is byte-for-byte deterministic (diffed by demo/run_demo against
// demo/expected_output.txt); run-to-run timings go to STDERR.
//
// Code tour: start here, then shape_overlap.h (the physics), kernels.cuh ->
// kernels.cu (the GPU side), then reference_cpu.* (the baseline + loader).
// ===========================================================================
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "kernels.cuh"        // shape_screen_gpu, ConformerSet, Molecule
#include "reference_cpu.h"    // load_conformers, shape_tanimoto_cpu
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "1.13";
static const char* PROJECT_NAME = "Pharmacophore & 3D Shape Screening";

// Tolerance: CPU and GPU run the IDENTICAL double-precision physics in the
// IDENTICAL loop order (shape_overlap.h). The only possible difference is FMA
// contraction (the device may fuse a*b+c where the host does not), which is a
// ~1e-12 relative effect on a short sum. 1e-9 (absolute, on scores in [0,1]) is
// a generous, honest floor that this difference never reaches. See THEORY sec 6.
static constexpr double TOLERANCE = 1.0e-9;
static constexpr int    TOP_K     = 5;       // how many best hits to report

// max_abs_err for DOUBLE arrays (util::max_abs_err is float-only). Returns
// +infinity on a length mismatch so a shape bug cannot masquerade as agreement.
static double max_abs_err_d(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double d = std::fabs(a[i] - b[i]);
        if (d > worst) worst = d;
    }
    return worst;
}

// Return the indices of the TOP_K largest scores, ties broken by lower index
// (so the ranking is deterministic). Uses partial_sort on an index vector.
static std::vector<int> top_k(const std::vector<double>& score, int k) {
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
    const std::string path = (argc > 1) ? argv[1] : "data/sample/conformers_sample.txt";
    ConformerSet set;
    try {
        set = load_conformers(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<double> score_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    shape_tanimoto_cpu(set, score_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU screen (kernel timed inside the wrapper) ------------------
    std::vector<double> score_gpu;
    float gpu_kernel_ms = 0.0f;
    shape_screen_gpu(set, score_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    const double err  = max_abs_err_d(score_cpu, score_gpu);
    const bool   pass = err <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    // We print the GPU scores (verified equal to the CPU's) at FIXED 6-decimal
    // precision so the output is byte-identical run to run.
    const std::vector<int> best = top_k(score_gpu, TOP_K);
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("Gaussian shape screen: query (%d atoms) vs %d library conformers\n",
                set.query.n_atoms, set.n);
    std::printf("top-%d by Shape Tanimoto:\n", static_cast<int>(best.size()));
    for (std::size_t r = 0; r < best.size(); ++r) {
        const int k = best[r];
        std::printf("  #%zu  %-10s  ShapeTanimoto = %.6f\n",
                    r + 1, set.name[static_cast<std::size_t>(k)].c_str(), score_gpu[k]);
    }
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-09)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (query=%d atoms, n=%d conformers)\n",
                 path.c_str(), set.query.n_atoms, set.n);
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny sample is dominated by "
                         "launch/copy overhead; the GPU wins at library scale (millions of conformers).\n");
    std::fprintf(stderr, "[verify] max_abs_err = %.3e  (tolerance %.1e)\n", err, TOLERANCE);

    return pass ? 0 : 1;
}
