// ===========================================================================
// src/main.cu  --  Entry point: virtual population + Sobol sensitivity, verified
// ---------------------------------------------------------------------------
// Project 6.26 : Virtual Population Generation & Sensitivity Analysis
//
// 5-step shape (the shape EVERY project in this repo follows):
//   1. Load the population/sensitivity config (parameter ranges + N).
//   2. CPU reference: evaluate all N*(k+2) Saltelli model runs (reference_cpu.cpp).
//   3. GPU: one thread per model run -> the same AUC array (kernels.cu).
//   4. VERIFY: (a) raw AUC arrays agree to round-off, and (b) the Sobol indices
//      computed from each agree exactly (same host reduction on both arrays).
//   5. REPORT: deterministic Sobol table + population summary -> stdout;
//      timings + the analytic cross-check verdict -> stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Timings (run-to-run varying) go to STDERR.
//
// Code tour: start here, then vpop.h (model + Saltelli sampling), kernels.cu
// (GPU twin), reference_cpu.cpp (CPU reference + Sobol reduction).
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // evaluate_gpu, VpopParams
#include "reference_cpu.h"    // load_vpop, evaluate_cpu, compute_sobol, SobolResult
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "6.26";
static const char* PROJECT_NAME = "Virtual Population Generation & Sensitivity Analysis";

// The model + Saltelli reduction are all double precision and run the identical
// operations on both sides, so CPU and GPU agree to ~round-off. 1e-9 is a strict
// floating-point tolerance (PATTERNS.md section 4, "same exact operations").
static constexpr double TOLERANCE = 1.0e-9;

// Fixed parameter labels, in the VPOP_K order used everywhere (vpop.h).
static const char* PARAM_NAME[VPOP_K] = {"ka", "CL", "V", "F"};

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/vpop_config.txt";
    VpopParams P;
    try {
        P = load_vpop(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const long total = vpop_num_evals(P.N);   // N*(k+2) model evaluations

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<double> f_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    evaluate_cpu(P, f_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU evaluation (kernel timed) ---------------------------------
    std::vector<double> f_gpu;
    float gpu_kernel_ms = 0.0f;
    evaluate_gpu(P, f_gpu, &gpu_kernel_ms);

    // ---- 4a. Verify the raw AUC arrays -------------------------------------
    double worst_raw = 0.0;
    for (long g = 0; g < total; ++g)
        worst_raw = std::fmax(worst_raw,
                              std::fabs(f_cpu[(std::size_t)g] - f_gpu[(std::size_t)g]));

    // ---- 4b. Sobol reduction on BOTH arrays, then verify the indices -------
    const SobolResult sob_cpu = compute_sobol(P, f_cpu);
    const SobolResult sob_gpu = compute_sobol(P, f_gpu);
    double worst_idx = std::fabs(sob_cpu.var - sob_gpu.var);
    for (int j = 0; j < VPOP_K; ++j) {
        worst_idx = std::fmax(worst_idx, std::fabs(sob_cpu.S[j]  - sob_gpu.S[j]));
        worst_idx = std::fmax(worst_idx, std::fabs(sob_cpu.ST[j] - sob_gpu.ST[j]));
    }
    const bool pass = (worst_raw <= TOLERANCE) && (worst_idx <= TOLERANCE);

    // Independent SCIENCE check (PATTERNS.md section 4): AUC = F*Dose/CL depends
    // ONLY on F and CL, so the first-order indices S[CL] + S[F] must dominate and
    // S[ka] + S[V] must be ~0. This validates the algorithm, not just CPU==GPU.
    // (Reported on stderr; it does not affect the deterministic stdout.)
    const double s_relevant   = sob_gpu.S[1] + sob_gpu.S[3];                       // CL + F
    const double s_irrelevant = std::fabs(sob_gpu.S[0]) + std::fabs(sob_gpu.S[2]); // ka + V
    const bool science_ok = (s_relevant > 0.90) && (s_irrelevant < 0.05);

    // argmax over the first-order indices -> the single most influential input.
    int dominant = 0;
    for (int j = 1; j < VPOP_K; ++j)
        if (sob_gpu.S[j] > sob_gpu.S[dominant]) dominant = j;

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("virtual population: %d Saltelli base samples, k=%d params, "
                "%ld model evals\n", P.N, VPOP_K, total);
    std::printf("PK model: 1-compartment oral, dose=%.0f mg, AUC over %.0f h "
                "(%d trapezoid steps)\n", P.dose, P.t_end, P.steps);
    std::printf("parameter ranges (uniform priors):\n");
    for (int j = 0; j < VPOP_K; ++j)
        std::printf("  %-3s in [%.3f, %.3f]\n", PARAM_NAME[j], P.lo[j], P.hi[j]);
    std::printf("population AUC: mean=%.4f  variance=%.4f  (mg.h/L)\n",
                sob_gpu.mean, sob_gpu.var);
    std::printf("Sobol sensitivity indices (fraction of AUC variance):\n");
    std::printf("  param   S1(first-order)   ST(total-order)\n");
    for (int j = 0; j < VPOP_K; ++j)
        std::printf("  %-4s    %12.4f     %12.4f\n",
                    PARAM_NAME[j], sob_gpu.S[j], sob_gpu.ST[j]);
    std::printf("dominant parameter (largest S1): %s\n", PARAM_NAME[dominant]);
    std::printf("RESULT: %s (GPU matches CPU; raw AUC and Sobol indices within "
                "tol=1.0e-09)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail + science cross-check -> STDERR ----------------
    std::fprintf(stderr, "[data]   source: %s  (N=%d, %ld evaluations)\n",
                 path.c_str(), P.N, total);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- Sobol needs N*(k+2) "
                 "evaluations; the GPU's edge grows with N, k, and model cost.\n");
    std::fprintf(stderr, "[verify] worst raw AUC diff = %.3e ; worst index diff "
                 "= %.3e ; tol = %.1e\n", worst_raw, worst_idx, TOLERANCE);
    std::fprintf(stderr, "[science] analytic AUC = F*Dose/CL depends only on "
                 "F,CL -> expect S(CL)+S(F) ~ 1, S(ka)+S(V) ~ 0.\n");
    std::fprintf(stderr, "[science] S(CL)+S(F)=%.4f  |S(ka)|+|S(V)|=%.4f  -> %s\n",
                 s_relevant, s_irrelevant, science_ok ? "CONSISTENT" : "UNEXPECTED");

    return pass ? 0 : 1;
}
