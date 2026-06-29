// ===========================================================================
// src/main.cu  --  Entry point: integrate SEIR ensemble, verify, report
// ---------------------------------------------------------------------------
// Project 9.02 : Large-Scale Compartmental & Metapopulation Models
//
// 5-step shape:
//   1. Load the ensemble config (a beta x gamma parameter sweep).
//   2. CPU reference: integrate every member serially (reference_cpu.cpp).
//   3. GPU: one thread per member, full RK4 loop each (kernels.cu).
//   4. VERIFY: per-member results match (same RK4 -> same numbers).
//   5. REPORT: deterministic sample members + ensemble summary.
//
// Code tour: start here, then seir.h (the ODE + RK4), kernels.cu, reference_cpu.cpp.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // integrate_gpu, EnsembleConfig, MemberResult
#include "reference_cpu.h"    // load_ensemble, integrate_cpu, member_params
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "9.2";
static const char* PROJECT_NAME = "Large-Scale Compartmental & Metapopulation Models";

// Double-precision RK4: CPU and GPU do the same ops, agreeing to ~round-off.
static constexpr double TOLERANCE = 1.0e-6;

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/ensemble_params.txt";
    EnsembleConfig c;
    try {
        c = load_ensemble(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int M = ensemble_size(c);

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<MemberResult> res_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    integrate_cpu(c, res_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU ensemble (kernel timed) -----------------------------------
    std::vector<MemberResult> res_gpu;
    float gpu_kernel_ms = 0.0f;
    integrate_gpu(c, res_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    double worst = 0.0;
    for (int i = 0; i < M; ++i) {
        worst = std::fmax(worst, std::fabs(res_cpu[i].peak_I_frac - res_gpu[i].peak_I_frac));
        worst = std::fmax(worst, std::fabs(res_cpu[i].attack_rate - res_gpu[i].attack_rate));
    }
    const bool pass = worst <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    // Ensemble summary: how many members have an epidemic (R0>1), the mean peak
    // among them, and the largest peak.
    int epidemics = 0; double sum_peak = 0.0, max_peak = 0.0;
    for (int i = 0; i < M; ++i) {
        double beta, gamma; member_params(c, i, beta, gamma);
        if (beta / gamma > 1.0) { ++epidemics; sum_peak += res_gpu[i].peak_I_frac; }
        max_peak = std::fmax(max_peak, res_gpu[i].peak_I_frac);
    }

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("SEIR ensemble: %d members (%d beta x %d gamma), N=%.0f, %d days @ dt=%.2f\n",
                M, c.nb, c.ng, c.N, static_cast<int>(c.steps * c.dt), c.dt);
    std::printf("sample members (beta gamma R0 -> peakI%% peakDay attack%%):\n");
    const int picks[5] = {0, M / 4, M / 2, (3 * M) / 4, M - 1};
    for (int s = 0; s < 5; ++s) {
        const int i = picks[s];
        double beta, gamma; member_params(c, i, beta, gamma);
        std::printf("  m%-5d: %.3f %.3f %.2f -> %6.3f %5.1f %6.2f\n",
                    i, beta, gamma, beta / gamma,
                    100.0 * res_gpu[i].peak_I_frac, res_gpu[i].peak_step * c.dt,
                    100.0 * res_gpu[i].attack_rate);
    }
    std::printf("ensemble: %d/%d members with R0>1; mean peak I = %.4f; max peak I = %.4f\n",
                epidemics, M, epidemics ? sum_peak / epidemics : 0.0, max_peak);
    std::printf("RESULT: %s (GPU ensemble matches CPU within tol=1.0e-06)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d members)\n", path.c_str(), M);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- speed-up grows with ensemble size; real UQ runs "
                         "10^4-10^6 members.\n");
    std::fprintf(stderr, "[verify] worst per-member diff = %.3e  (tolerance %.1e)\n", worst, TOLERANCE);

    return pass ? 0 : 1;
}
