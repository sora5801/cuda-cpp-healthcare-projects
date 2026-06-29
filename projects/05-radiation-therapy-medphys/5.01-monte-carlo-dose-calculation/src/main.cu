// ===========================================================================
// src/main.cu  --  Entry point: run MC on CPU + GPU, verify, report dose
// ---------------------------------------------------------------------------
// Project 5.01 : Monte Carlo Dose Calculation (simplified slab)
//
// 5-step shape:
//   1. Load the simulation parameters (data/sample).
//   2. CPU reference Monte Carlo (reference_cpu.cpp).
//   3. GPU Monte Carlo (kernels.cu) -- IDENTICAL histories (shared RNG).
//   4. VERIFY: integer dose tallies match exactly (atomics commute on ints).
//   5. REPORT: deterministic depth-dose histogram to stdout; timing to stderr.
//
// Code tour: start here, then mc_physics.h (RNG + transport), kernels.cu,
// reference_cpu.cpp. The science/GPU-mapping is in ../THEORY.md.
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // dose_gpu, DoseProblem
#include "reference_cpu.h"    // load_dose_problem, dose_cpu
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "5.1";
static const char* PROJECT_NAME = "Monte Carlo Dose Calculation (simplified slab)";

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/mc_params.txt";
    DoseProblem prob;
    try {
        prob = load_dose_problem(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<unsigned long long> dose_c;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    dose_cpu(prob, dose_c);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU Monte Carlo (kernel timed) --------------------------------
    std::vector<unsigned long long> dose_g;
    float gpu_kernel_ms = 0.0f;
    dose_gpu(prob, dose_g, &gpu_kernel_ms);

    // ---- 4. Verify (exact integer match) ----------------------------------
    int mismatches = 0;
    unsigned long long total = 0;
    int peak_bin = 0;
    for (int b = 0; b < prob.sp.n_bins; ++b) {
        if (dose_c[b] != dose_g[b]) ++mismatches;
        total += dose_g[b];
        if (dose_g[b] > dose_g[peak_bin]) peak_bin = b;
    }
    const bool pass = (mismatches == 0);

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    const unsigned long long emitted = prob.n_photons * prob.sp.E0;
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("slab L=%.1f cm, %d depth bins, mu=%.3f /cm, p_abs=%.2f\n",
                prob.sp.L, prob.sp.n_bins, prob.sp.mu, prob.sp.p_abs);
    std::printf("histories = %llu, E0 = %llu quanta/photon\n", prob.n_photons, prob.sp.E0);
    std::printf("deposited = %llu of %llu quanta (%.1f%%), peak depth bin = %d\n",
                total, emitted, 100.0 * total / emitted, peak_bin);
    std::printf("depth-dose (quanta per bin):\n ");
    for (int b = 0; b < prob.sp.n_bins; ++b) std::printf(" %llu", dose_g[b]);
    std::printf("\n");
    std::printf("RESULT: %s (GPU dose tally matches CPU exactly)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s\n", path.c_str());
    std::fprintf(stderr, "[timing] CPU MC: %.3f ms   GPU MC: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- speed-up grows with history count; "
                         "clinical plans run 1e9-1e10 histories.\n");
    std::fprintf(stderr, "[verify] bin mismatches = %d (integer dose => atomics commute)\n", mismatches);

    return pass ? 0 : 1;
}
