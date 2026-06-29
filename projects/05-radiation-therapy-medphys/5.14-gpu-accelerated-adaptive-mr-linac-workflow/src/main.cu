// ===========================================================================
// src/main.cu  --  Entry point: load data, run CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 5.14 -- GPU-Accelerated Adaptive MR-Linac Workflow   (template skeleton)
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the problem (from data/sample, or a built-in synthetic fallback).
//   2. Compute the CPU reference (reference_cpu.cpp)         -> trusted answer.
//   3. Compute the GPU result    (kernels.cu)                -> the thing taught.
//   4. VERIFY: assert GPU agrees with CPU within a tolerance -> correctness.
//   5. REPORT: deterministic result to stdout; timing to stderr.
//
//   STDOUT is kept byte-for-byte deterministic so demo/run_demo can diff it
//   against demo/expected_output.txt. Anything that varies run-to-run (timings)
//   goes to STDERR, which the demo shows but does not diff.
//
//   TODO(impl): swap the SAXPY placeholder for this project's real problem,
//   data loading, and verification. Keep the 5-step shape and the stdout/stderr
//   split so the demo harness keeps working.
//
// READ THIS FIRST in the code tour, then kernels.cuh -> kernels.cu, and
// reference_cpu.cpp for the baseline. See ../THEORY.md for the "why".
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // saxpy_gpu (GPU path)
#include "reference_cpu.h"    // saxpy_cpu (CPU baseline)
#include "util/io.hpp"        // util::CpuTimer, util::max_abs_err, read_floats

// These two tokens are filled in by tools/scaffold.py so the program identifies
// itself. They MUST stay in sync with demo/expected_output.txt (also stamped).
static const char* PROJECT_ID   = "5.14";
static const char* PROJECT_NAME = "GPU-Accelerated Adaptive MR-Linac Workflow";

// Correctness tolerance: the GPU result must match the CPU within this.
static constexpr double TOLERANCE = 1.0e-5;

// Build the built-in synthetic problem used when no data file is supplied.
//   n=8, a=2, x[i]=i, y[i]=10*i  =>  out[i] = 2*i + 10*i = 12*i (exact ints).
// These EXACT values are what demo/expected_output.txt encodes.
static void make_synthetic(int& n, float& a, std::vector<float>& x, std::vector<float>& y) {
    n = 8;
    a = 2.0f;
    x.resize(n);
    y.resize(n);
    for (int i = 0; i < n; ++i) {
        x[i] = static_cast<float>(i);
        y[i] = static_cast<float>(10 * i);
    }
}

// Parse a sample file laid out as:  n  a  x0 x1 ... x{n-1}  y0 y1 ... y{n-1}
// Returns false if the file is missing/short so the caller can fall back.
static bool load_sample(const std::string& path, int& n, float& a,
                        std::vector<float>& x, std::vector<float>& y) {
    std::vector<float> v;
    try {
        v = util::read_floats(path);
    } catch (const std::exception&) {
        return false;  // file not found -> caller uses synthetic data
    }
    if (v.size() < 2) return false;
    n = static_cast<int>(v[0]);
    a = v[1];
    if (n <= 0 || v.size() < static_cast<std::size_t>(2 + 2 * n)) return false;
    x.assign(v.begin() + 2, v.begin() + 2 + n);
    y.assign(v.begin() + 2 + n, v.begin() + 2 + 2 * n);
    return true;
}

int main(int argc, char** argv) {
    // ---- 1. Load the problem ------------------------------------------------
    int n = 0;
    float a = 0.0f;
    std::vector<float> x, y;
    const char* source = "synthetic (built-in)";
    if (argc > 1 && load_sample(argv[1], n, a, x, y)) {
        source = argv[1];
    } else {
        make_synthetic(n, a, x, y);
    }

    // ---- 2. CPU reference (timed) ------------------------------------------
    std::vector<float> out_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    saxpy_cpu(n, a, x, y, out_cpu);
    double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU result (kernel timed inside the wrapper) -------------------
    std::vector<float> out_gpu;
    float gpu_kernel_ms = 0.0f;
    saxpy_gpu(n, a, x, y, out_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ----------------------------------------------------------
    double err = util::max_abs_err(out_cpu, out_gpu);
    bool pass = err <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("[template placeholder kernel: SAXPY  out = a*x + y]\n");
    std::printf("n = %d  a = %g\n", n, a);
    int show = n < 16 ? n : 8;                 // print all if small, else first 8
    std::printf("out[0:%d] =", show);
    for (int i = 0; i < show; ++i) std::printf(" %.6f", out_gpu[i]);
    std::printf("\n");
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-05)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s\n", source);
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- tiny n is dominated "
                         "by launch/copy overhead, not compute.\n");
    std::fprintf(stderr, "[verify] max_abs_err = %.6e  (tolerance %.1e)\n", err, TOLERANCE);

    // Exit code feeds the demo's pass/fail gate.
    return pass ? 0 : 1;
}
