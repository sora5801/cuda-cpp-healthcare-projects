// ===========================================================================
// src/main.cu  --  Entry point: build an MSM on CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 1.17 : Markov State Models from MD
//
// 5-step shape (every project in this repo follows it):
//   1. Load the featurized trajectory (data/sample): N frames x D features,
//      K microstates, lag tau.
//   2. CPU reference MSM (reference_cpu.cpp): the trusted baseline.
//   3. GPU MSM (kernels.cu): parallel assign + atomic-integer accumulate/count.
//   4. VERIFY: labels, centroids, the integer count matrix, and the transition
//      matrix all match the CPU (integer/fixed-point atomics commute -> exact).
//   5. REPORT: deterministic MSM summary -> stdout; timing -> stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt; run-varying numbers (timings) go to STDERR.
//
// Code tour: start here, then msm.h (distance + fixed-point), reference_cpu.cpp
//   (the pipeline), kernels.cu (the GPU twin).
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // msm_gpu, Dataset, MsmResult
#include "reference_cpu.h"    // load_dataset, msm_cpu
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "1.17";
static const char* PROJECT_NAME = "Markov State Models from MD";

// Fixed number of Lloyd iterations -> fully deterministic (no convergence test
// that could differ between CPU and GPU). 25 is ample for the separated basins
// in the synthetic sample.
static constexpr int ITERS = 25;

// Verification tolerances. Labels and the integer count matrix must match
// EXACTLY (integer atomics commute). Centroids match to fixed-point precision;
// the transition matrix and pi are derived from identical integer counts, so a
// tiny slack only covers double-rounding in the host helpers.
static constexpr double CENTROID_TOL = 1.0e-4;
static constexpr double MATRIX_TOL   = 1.0e-12;

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/trajectory_sample.txt";
    Dataset d;
    try {
        d = load_dataset(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    const MsmResult cpu = msm_cpu(d, ITERS);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU MSM (kernels timed inside the wrapper) --------------------
    float gpu_kernel_ms = 0.0f;
    const MsmResult gpu = msm_gpu(d, ITERS, &gpu_kernel_ms);

    // ---- 4. Verify (labels, centroids, counts, T) -------------------------
    int label_mismatch = 0;
    for (int i = 0; i < d.N; ++i) if (cpu.labels[i] != gpu.labels[i]) ++label_mismatch;

    double cent_diff = 0.0;
    for (std::size_t i = 0; i < cpu.centroids.size(); ++i)
        cent_diff = std::fmax(cent_diff, std::fabs((double)cpu.centroids[i] - (double)gpu.centroids[i]));

    long long count_mismatch = 0;   // integer count matrix must be exactly equal
    for (std::size_t i = 0; i < cpu.counts.size(); ++i)
        if (cpu.counts[i] != gpu.counts[i]) ++count_mismatch;

    double matrix_diff = 0.0;
    for (std::size_t i = 0; i < cpu.T.size(); ++i)
        matrix_diff = std::fmax(matrix_diff, std::fabs(cpu.T[i] - gpu.T[i]));

    const bool pass = (label_mismatch == 0) && (count_mismatch == 0)
                      && (cent_diff <= CENTROID_TOL) && (matrix_diff <= MATRIX_TOL);

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) ----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("MSM: %d frames x %d features -> %d microstates, lag=%d, %d k-means iters\n",
                d.N, d.D, d.K, d.lag, ITERS);

    // Microstate equilibrium populations (the stationary distribution pi).
    std::printf("microstate populations (pi):\n");
    for (int k = 0; k < d.K; ++k)
        std::printf("  state %d: n=%5u  pi=%.4f\n", k, gpu.sizes[k], gpu.pi[k]);

    // The transition probability matrix T (rows = "from", cols = "to").
    std::printf("transition matrix T (row=from, col=to):\n");
    for (int i = 0; i < d.K; ++i) {
        std::printf("  ");
        for (int j = 0; j < d.K; ++j) std::printf(" %.4f", gpu.T[(std::size_t)i * d.K + j]);
        std::printf("\n");
    }

    // The slowest kinetic process recovered from the second eigenvalue.
    std::printf("slowest implied timescale t2 = %.2f frames (lambda2 = %.4f)\n",
                gpu.timescale, gpu.lambda2);
    std::printf("RESULT: %s (GPU labels+counts+centroids+T match CPU)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) -----------------
    std::fprintf(stderr, "[data]   source: %s  (%d frames, %d features, lag %d)\n",
                 path.c_str(), d.N, d.D, d.lag);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU loop: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- on this tiny synthetic set the GPU is "
                         "launch-bound; the edge grows with millions of MD frames.\n");
    std::fprintf(stderr, "[verify] label mismatches=%d  count mismatches=%lld  "
                         "max|dCentroid|=%.3e  max|dT|=%.3e\n",
                 label_mismatch, count_mismatch, cent_diff, matrix_diff);
    std::fprintf(stderr, "[verify] inertia(cpu/gpu) = %.4f / %.4f\n",
                 compute_inertia(d, cpu.centroids, cpu.labels),
                 compute_inertia(d, gpu.centroids, gpu.labels));

    return pass ? 0 : 1;
}
