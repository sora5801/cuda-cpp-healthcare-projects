// ===========================================================================
// src/main.cu  --  Entry point: cluster cytometry events, verify, report
// ---------------------------------------------------------------------------
// Project 11.09 : Flow Cytometry & High-Content Screening Analysis
//
// 5-step shape:
//   1. Load the events (data/sample): N cells x D markers, K clusters.
//   2. CPU reference k-means (reference_cpu.cpp).
//   3. GPU k-means (kernels.cu): parallel assign + atomic fixed-point accumulate.
//   4. VERIFY: labels + centroids match exactly (fixed-point atomics commute).
//   5. REPORT: deterministic cluster sizes, centroids, and inertia.
//
// Code tour: start here, then kmeans.h (distance + fixed-point), kernels.cu,
// reference_cpu.cpp.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // kmeans_gpu, Dataset
#include "reference_cpu.h"    // load_dataset, kmeans_cpu
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "11.9";
static const char* PROJECT_NAME = "Flow Cytometry & High-Content Screening Analysis";

static constexpr int    ITERS     = 20;       // fixed Lloyd iterations (deterministic)
static constexpr double TOLERANCE = 1.0e-4;   // centroids agree exactly; this is slack

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/cytometry_sample.txt";
    Dataset d;
    try {
        d = load_dataset(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<float> cent_cpu;
    std::vector<int> lab_cpu;
    std::vector<unsigned int> sz_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    const double inertia_cpu = kmeans_cpu(d, ITERS, cent_cpu, lab_cpu, sz_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU k-means (loop timed) --------------------------------------
    std::vector<float> cent_gpu;
    std::vector<int> lab_gpu;
    std::vector<unsigned int> sz_gpu;
    float gpu_kernel_ms = 0.0f;
    const double inertia_gpu = kmeans_gpu(d, ITERS, cent_gpu, lab_gpu, sz_gpu, &gpu_kernel_ms);

    // ---- 4. Verify (labels + centroids) -----------------------------------
    int label_mismatch = 0;
    for (int i = 0; i < d.N; ++i) if (lab_cpu[i] != lab_gpu[i]) ++label_mismatch;
    double cent_diff = 0.0;
    for (std::size_t i = 0; i < cent_cpu.size(); ++i)
        cent_diff = std::fmax(cent_diff, std::fabs((double)cent_cpu[i] - (double)cent_gpu[i]));
    const bool pass = (label_mismatch == 0) && (cent_diff <= TOLERANCE);

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("k-means: %d events x %d markers -> %d clusters, %d iterations\n", d.N, d.D, d.K, ITERS);
    for (int k = 0; k < d.K; ++k) {
        std::printf("  cluster %d (n=%5u): centroid =", k, sz_gpu[k]);
        for (int j = 0; j < d.D; ++j) std::printf(" %.4f", cent_gpu[(std::size_t)k * d.D + j]);
        std::printf("\n");
    }
    std::printf("inertia = %.4f\n", inertia_gpu);
    std::printf("RESULT: %s (GPU labels+centroids match CPU)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d events, %d markers)\n", path.c_str(), d.N, d.D);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU loop: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- speed-up grows with event count; real runs are "
                         "10^6-10^7 cells.\n");
    std::fprintf(stderr, "[verify] label mismatches = %d, max centroid diff = %.3e, "
                         "inertia(cpu/gpu) = %.4f / %.4f\n",
                 label_mismatch, cent_diff, inertia_cpu, inertia_gpu);

    return pass ? 0 : 1;
}
