// ===========================================================================
// src/main.cu  --  Entry point: load waveform, filter, verify, report
// ---------------------------------------------------------------------------
// Project 7.10 : Physiological Signal & Waveform Analysis
//
// 5-step shape:
//   1. Load the noisy waveform (data/sample).
//   2. CPU reference 1-D convolution (reference_cpu.cpp).
//   3. GPU tiled 1-D convolution (kernels.cu).
//   4. VERIFY: GPU filtered signal matches CPU within tolerance.
//   5. REPORT: deterministic filtered samples to stdout; timing to stderr.
//
// Code tour: start here, then kernels.cuh -> kernels.cu (the tiling), then
// reference_cpu.cpp. The science/GPU-mapping is in ../THEORY.md.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // conv1d_gpu, Signal
#include "reference_cpu.h"    // load_signal, make_gaussian_filter, conv1d_cpu
#include "util/io.hpp"        // util::CpuTimer, util::max_abs_err

static const char* PROJECT_ID   = "7.10";
static const char* PROJECT_NAME = "Physiological Signal & Waveform Analysis";

// Low-pass Gaussian FIR: 31 taps, sigma 5 samples (smooths high-frequency noise
// while preserving the slow ECG morphology). Tunable; see Exercises.
static constexpr int    FILTER_K     = 31;
static constexpr double FILTER_SIGMA = 5.0;
static constexpr double TOLERANCE    = 1.0e-4;   // float conv: CPU/GPU FMA differences

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/ecg_sample.txt";
    Signal s;
    try {
        s = load_signal(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const std::vector<float> h = make_gaussian_filter(FILTER_K, FILTER_SIGMA);

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<float> y_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    conv1d_cpu(s, h, y_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU tiled convolution (kernel timed) --------------------------
    std::vector<float> y_gpu;
    float gpu_kernel_ms = 0.0f;
    conv1d_gpu(s, h, y_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    const double err = util::max_abs_err(y_cpu, y_gpu);
    const bool pass = err <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    // Noise-reduction sanity: RMS of the high-frequency residual (x - filtered).
    double resid = 0.0;
    float ymax = y_gpu[0]; int imax = 0;
    for (int i = 0; i < s.n; ++i) {
        const double d = s.x[i] - y_gpu[i];
        resid += d * d;
        if (y_gpu[i] > ymax) { ymax = y_gpu[i]; imax = i; }
    }
    resid = std::sqrt(resid / s.n);

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("1-D FIR low-pass: n=%d samples, K=%d taps, sigma=%.1f\n", s.n, FILTER_K, FILTER_SIGMA);
    std::printf("filtered peak = %.6f at sample %d\n", ymax, imax);
    std::printf("removed (RMS of x - filtered) = %.6f\n", resid);
    std::printf("filtered samples (8 evenly spaced):");
    for (int s8 = 0; s8 < 8; ++s8) {
        const int i = (s8 * (s.n - 1)) / 7;
        std::printf(" %.6f", y_gpu[i]);
    }
    std::printf("\n");
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-04)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d samples)\n", path.c_str(), s.n);
    std::fprintf(stderr, "[timing] CPU conv: %.3f ms   GPU tiled conv: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- the GPU's edge grows with signal length and with "
                         "batching thousands of multi-hour recordings.\n");
    std::fprintf(stderr, "[verify] max_abs_err = %.3e  (tolerance %.1e)\n", err, TOLERANCE);

    return pass ? 0 : 1;
}
