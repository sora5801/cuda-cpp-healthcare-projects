// ===========================================================================
// src/main.cu  --  Entry point: load fMRI, fit GLM (CPU+GPU), verify, report
// ---------------------------------------------------------------------------
// Project 4.16 : Functional MRI Analysis
//
// The 5-step shape every project in this repo follows:
//   1. Load the problem (V voxels x T scans + design params from data/sample).
//   2. Precompute the voxel-independent (X^T X)^-1 once.
//   3. CPU reference  (reference_cpu.cpp) -> trusted per-voxel t-stats.
//   4. GPU kernel     (kernels.cu)        -> the thing being taught.
//   5. VERIFY GPU vs CPU within tolerance, then REPORT the deterministic
//      activation map (top voxels by |t|) to stdout; timing/error to stderr.
//
// STDOUT is byte-for-byte deterministic (diffed by demo/run_demo against
// demo/expected_output.txt); run-to-run timings go to STDERR.
//
// Code tour: start here, then glm.h (the science) -> kernels.cuh -> kernels.cu,
// then reference_cpu.*. See ../THEORY.md for the "why".
// ===========================================================================
#include <algorithm>   // std::partial_sort, std::min
#include <cmath>       // std::fabs, std::sqrt
#include <cstdio>
#include <limits>      // std::numeric_limits (infinity sentinel)
#include <numeric>     // std::iota
#include <string>
#include <vector>

#include "kernels.cuh"        // glm_gpu, glm_kernel
#include "reference_cpu.h"    // load_fmri, compute_XtX_inv, glm_cpu
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "4.16";
static const char* PROJECT_NAME = "Functional MRI Analysis";

// Verification tolerance on the per-voxel t-statistic. CPU and GPU call the SAME
// fit_voxel() (glm.h) in double precision, so they agree to round-off; the only
// possible divergence is the GPU's fused-multiply-add reassociating a few FP64
// products. 1e-9 is comfortably above that and far below any t-value we report
// to 4 decimals. (See docs/PATTERNS.md §4 and THEORY §"How we verify".)
static constexpr double TOLERANCE = 1.0e-9;

// How many top-activated voxels to list in the deterministic report.
static constexpr int TOP_K = 6;

// max_abs_err over two double vectors (returns +inf on a length mismatch so a
// shape bug can never masquerade as agreement). Mirrors util::max_abs_err but
// for doubles (our t-stats are FP64).
static double max_abs_err_d(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return std::numeric_limits<double>::infinity();  // shape bug
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs(a[i] - b[i]);
        if (d > worst) worst = d;
    }
    return worst;
}

// Indices of the TOP_K largest t-statistics, ties broken by LOWER index so the
// ranking is fully deterministic. partial_sort on an index vector.
static std::vector<int> top_k(const std::vector<double>& t, int k) {
    std::vector<int> idx(t.size());
    std::iota(idx.begin(), idx.end(), 0);
    const int kk = std::min<int>(k, static_cast<int>(idx.size()));
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
        [&](int a, int b) {
            if (t[a] != t[b]) return t[a] > t[b];   // larger t first
            return a < b;                           // tie -> lower index
        });
    idx.resize(kk);
    return idx;
}

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1]
                                        : "data/sample/fmri_sample.txt";
    FmriDataset ds;
    try {
        ds = load_fmri(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. Precompute the voxel-independent (X^T X)^-1 --------------------
    double XtX_inv[9];
    const double det = compute_XtX_inv(ds.design, XtX_inv);
    if (det == 0.0) {
        std::fprintf(stderr, "[error] design matrix is rank-deficient (det X^TX = 0)\n");
        return 2;
    }

    // ---- 3. CPU reference (timed) -----------------------------------------
    std::vector<double> t_cpu, beta_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    glm_cpu(ds, XtX_inv, t_cpu, beta_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 4. GPU kernel (timed inside the wrapper) -------------------------
    std::vector<double> t_gpu, beta_gpu;
    float gpu_kernel_ms = 0.0f;
    glm_gpu(ds, XtX_inv, t_gpu, beta_gpu, &gpu_kernel_ms);

    // ---- 5. Verify --------------------------------------------------------
    const double err = max_abs_err_d(t_cpu, t_gpu);
    const bool pass = err <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    // Rank voxels by the GPU t-statistic (== CPU within tol). We also count how
    // many of the top-K are TRULY active (embedded ground truth from the
    // synthetic generator) -- this recovers the planted answer, proving the
    // pipeline works end to end (docs/PATTERNS.md §6).
    const std::vector<int> best = top_k(t_gpu, TOP_K);
    int recovered = 0;
    for (int v : best) if (ds.true_active[v]) ++recovered;
    // Total planted-active voxels, for context in the summary line.
    int n_active = 0;
    for (int v = 0; v < ds.V; ++v) if (ds.true_active[v]) ++n_active;

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("mass-univariate GLM: %d voxels x %d scans (TR=%.1fs, block=%d scans)\n",
                ds.V, ds.design.T, ds.design.TR_seconds, ds.design.block_scans);
    std::printf("design: [task(HRF), linear-drift, intercept]   contrast = task\n");
    std::printf("top-%d voxels by task t-statistic (tie -> lower index):\n",
                static_cast<int>(best.size()));
    for (std::size_t r = 0; r < best.size(); ++r) {
        const int v = best[r];
        std::printf("  #%zu  voxel[%3d]  t = %8.4f   beta_task = %8.4f   %s\n",
                    r + 1, v, t_gpu[v], beta_gpu[v],
                    ds.true_active[v] ? "[active]" : "[  -   ]");
    }
    std::printf("recovered %d/%d top voxels that are truly task-active "
                "(of %d active total)\n", recovered,
                static_cast<int>(best.size()), n_active);
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-09)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (V=%d voxels, T=%d scans)\n",
                 path.c_str(), ds.V, ds.design.T);
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny sample is dominated by "
                         "launch/copy overhead; the GPU wins at whole-brain scale "
                         "(V ~ 10^5 voxels).\n");
    std::fprintf(stderr, "[verify] max_abs_err(t) = %.3e  (tolerance %.1e)\n", err, TOLERANCE);

    return pass ? 0 : 1;
}
