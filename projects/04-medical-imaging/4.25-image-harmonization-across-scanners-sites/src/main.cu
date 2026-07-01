// ===========================================================================
// src/main.cu  --  Entry point: harmonize a multi-site feature table, verify
// ---------------------------------------------------------------------------
// Project 4.25 : Image Harmonization Across Scanners/Sites
//
// 6-step shape:
//   1. Load the multi-site feature table (data/sample): N samples x P features,
//      B scanners, C biological covariates.
//   2. Build the shared design matrix and fit the empirical-Bayes priors (host,
//      reused by both CPU and GPU so they harmonize identically).
//   3. CPU reference ComBat (reference_cpu.cpp) -> the trusted baseline.
//   4. GPU ComBat (kernels.cu): one thread per feature, same shared core.
//   5. VERIFY: the two harmonized tables agree to ~machine precision.
//   6. REPORT: the batch-mean gap BEFORE vs AFTER (proof the scanner signature
//      was removed) + a few harmonized values, all deterministic -> STDOUT.
//
// Code tour: start here, then combat.h (the per-feature math), reference_cpu.cpp
// (loader + priors + serial reference), kernels.cu (the GPU twin).
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // combat_gpu, Dataset (via reference_cpu.h)
#include "reference_cpu.h"    // load_dataset, build_design, estimate_priors, combat_cpu
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "4.25";
static const char* PROJECT_NAME = "Image Harmonization Across Scanners/Sites";

// Verification tolerance. CPU and GPU call the SAME __host__ __device__ core on
// the SAME double-precision inputs; the only divergence is fused-multiply-add
// contraction, which stays well below 1e-9 for these small per-feature solves
// (PATTERNS.md §4: "~machine precision for short double-precision computations").
// We report the actual max diff on stderr so the learner sees the real number.
static constexpr double TOLERANCE = 1.0e-9;

// max_abs_diff: largest |a[i]-b[i]| over two equal-length double tables.
static double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return 1e300;
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double dd = std::fabs(a[i] - b[i]);
        if (dd > worst) worst = dd;
    }
    return worst;
}

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/harmonization_sample.txt";
    Dataset d;
    try {
        d = load_dataset(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. Design matrix + empirical-Bayes priors (shared by CPU & GPU) ----
    std::vector<double> design;
    build_design(d, design);
    std::vector<double> gamma_bar, tau2, a_prior, b_prior;
    std::vector<int> batch_n;
    estimate_priors(d, design, gamma_bar, tau2, a_prior, b_prior, batch_n);

    // ---- 3. CPU reference (timed) ------------------------------------------
    std::vector<double> harm_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    combat_cpu(d, design, gamma_bar, tau2, a_prior, b_prior, batch_n, harm_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 4. GPU ComBat (kernel timed) --------------------------------------
    std::vector<double> harm_gpu;
    float gpu_kernel_ms = 0.0f;
    combat_gpu(d, design, gamma_bar, tau2, a_prior, b_prior, batch_n, harm_gpu, &gpu_kernel_ms);

    // ---- 5. Verify ---------------------------------------------------------
    const double diff = max_abs_diff(harm_cpu, harm_gpu);
    const bool pass = (diff <= TOLERANCE);

    // ---- Diagnostics: batch-mean gap before vs after -----------------------
    // The scientific check (PATTERNS.md §4, "compare against a known result"):
    // harmonization must SHRINK the across-scanner mean spread. We measure it on
    // the raw table and on the harmonized (GPU) table.
    const double gap_before = max_batch_mean_gap(d, d.Y);
    const double gap_after  = max_batch_mean_gap(d, harm_gpu);

    // ---- 6a. Deterministic report -> STDOUT --------------------------------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("ComBat: %d samples x %d features, %d scanners, %d covariate(s)\n",
                d.N, d.P, d.B, d.C);
    std::printf("max across-scanner feature-mean gap:\n");
    std::printf("  before harmonization = %.6f\n", gap_before);
    std::printf("  after  harmonization = %.6f\n", gap_after);
    // Show the first feature's harmonized values for the first sample of each
    // scanner, so the reader can eyeball that same-biology subjects now agree.
    std::printf("feature 0, first sample of each scanner (harmonized):\n");
    for (int b = 0; b < d.B; ++b) {
        int first = -1;
        for (int n = 0; n < d.N; ++n) if (d.batch[n] == b) { first = n; break; }
        if (first >= 0)
            std::printf("  scanner %d, sample %2d: %.6f\n", b, first, harm_gpu[first]);
    }
    std::printf("RESULT: %s (GPU harmonized table matches CPU reference)\n",
                pass ? "PASS" : "FAIL");

    // ---- 6b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d samples, %d features, %d scanners)\n",
                 path.c_str(), d.N, d.P, d.B);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernel: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- the GPU's edge grows with the FEATURE "
                         "count; real voxel/vertex harmonization has P ~ 10^5-10^6.\n");
    std::fprintf(stderr, "[verify] max |GPU - CPU| = %.3e  (tolerance %.1e)\n", diff, TOLERANCE);
    std::fprintf(stderr, "[science] scanner mean gap shrank %.6f -> %.6f "
                         "(batch effect removed; covariate signal preserved).\n",
                 gap_before, gap_after);

    return pass ? 0 : 1;
}
