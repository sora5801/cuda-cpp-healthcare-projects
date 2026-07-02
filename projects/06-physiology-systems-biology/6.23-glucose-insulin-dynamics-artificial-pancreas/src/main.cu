// ===========================================================================
// src/main.cu  --  Entry point: simulate a virtual-patient cohort, verify, report
// ---------------------------------------------------------------------------
// Project 6.23 : Glucose-Insulin Dynamics & Artificial Pancreas
//
// WHAT THIS FILE DOES  (the 5-step shape EVERY project in this repo follows)
//   1. Load the cohort configuration (data/sample, else a built-in fallback).
//   2. CPU reference: simulate every patient serially (reference_cpu.cpp).
//   3. GPU: one thread per patient, full closed-loop RK4+PID loop (kernels.cu).
//   4. VERIFY: per-patient metrics match (same math -> same numbers).
//   5. REPORT: deterministic sample patients + cohort summary to STDOUT;
//              timing / run-varying detail to STDERR.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Timings go to STDERR (shown, not diffed).
//
// Code tour: start here, then bergman.h (the model + RK4 + PID), kernels.cuh ->
// kernels.cu, and reference_cpu.cpp for the baseline. See ../THEORY.md.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // simulate_cohort_gpu, CohortConfig, PatientResult
#include "reference_cpu.h"    // load_cohort, simulate_cohort_cpu, patient_params
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "6.23";
static const char* PROJECT_NAME = "Glucose-Insulin Dynamics & Artificial Pancreas";

// Correctness tolerance. CPU and GPU run the SAME double-precision RK4 + PID
// (bergman.h), so they agree to ~machine precision. Over ~thousands of steps the
// GPU's fused multiply-add (FMA) can diverge from the host compiler by ~1e-9 in
// double precision (PATTERNS.md §4), so we verify to a physically-negligible
// 1e-4 mg/dL on glucose metrics -- far below any clinical meaning.
static constexpr double TOLERANCE = 1.0e-4;

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path =
        (argc > 1) ? argv[1] : "data/sample/cohort_params.txt";
    CohortConfig c;
    try {
        c = load_cohort(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int M = cohort_size(c);

    // ---- 2. CPU reference (timed) ------------------------------------------
    std::vector<PatientResult> res_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    simulate_cohort_cpu(c, res_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU cohort (kernel timed inside the wrapper) -------------------
    std::vector<PatientResult> res_gpu;
    float gpu_kernel_ms = 0.0f;
    simulate_cohort_gpu(c, res_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    // Compare the headline glucose metrics (min/max/mean G and TIR) per patient.
    double worst = 0.0;
    for (int i = 0; i < M; ++i) {
        worst = std::fmax(worst, std::fabs(res_cpu[i].min_G   - res_gpu[i].min_G));
        worst = std::fmax(worst, std::fabs(res_cpu[i].max_G   - res_gpu[i].max_G));
        worst = std::fmax(worst, std::fabs(res_cpu[i].mean_G  - res_gpu[i].mean_G));
        worst = std::fmax(worst, std::fabs(res_cpu[i].tir_frac - res_gpu[i].tir_frac));
    }
    const bool pass = worst <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    // Cohort summary: how many patients stay safe (no hypoglycemia) and their
    // average time-in-range -- the primary outcomes of an artificial-pancreas
    // in-silico trial.
    int safe = 0;              // patients with zero time below 70 mg/dL
    double sum_tir = 0.0, min_tir = 1.0;
    for (int i = 0; i < M; ++i) {
        if (res_gpu[i].hypo_frac <= 0.0) ++safe;
        sum_tir += res_gpu[i].tir_frac;
        min_tir = std::fmin(min_tir, res_gpu[i].tir_frac);
    }

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("Closed-loop cohort: %d virtual patients (%d SI x %d SG), "
                "%.0f min run @ dt=%.2f min, control every %.0f min\n",
                M, c.nSI, c.nSG, c.steps * c.dt, c.dt, c.control_dt);
    std::printf("meal: %.0f g carbs at t=%.0f min; target glucose %.0f mg/dL (PID)\n",
                c.meal_D / 1000.0, c.meal_t, c.G_target);
    std::printf("sample patients (SI=p3/p2  SG=p1 -> minG maxG meanG TIR%% hypo%% ins):\n");
    // Deterministic picks spanning the cohort (corners + centre).
    const int picks[5] = {0, M / 4, M / 2, (3 * M) / 4, M - 1};
    for (int s = 0; s < 5; ++s) {
        const int i = picks[s];
        const PatientParams pp = patient_params(c, i);
        const double SI = pp.p3 / pp.p2;   // insulin sensitivity
        std::printf("  p%-4d: SI=%.4f SG=%.4f -> %6.1f %6.1f %6.1f %5.1f %5.1f %6.1f\n",
                    i, SI, pp.p1,
                    res_gpu[i].min_G, res_gpu[i].max_G, res_gpu[i].mean_G,
                    100.0 * res_gpu[i].tir_frac, 100.0 * res_gpu[i].hypo_frac,
                    res_gpu[i].insulin_total);
    }
    std::printf("cohort: %d/%d patients avoided hypoglycemia; "
                "mean TIR = %.2f%%; worst TIR = %.2f%%\n",
                safe, M, 100.0 * sum_tir / M, 100.0 * min_tir);
    std::printf("RESULT: %s (GPU cohort matches CPU within tol=%.0e mg/dL)\n",
                pass ? "PASS" : "FAIL", TOLERANCE);

    // ---- 5b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d patients)\n", path.c_str(), M);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- speed-up grows with cohort size; "
                         "RL / UQ trials run 10^3-10^6 patients x many episodes.\n");
    std::fprintf(stderr, "[verify] worst per-patient metric diff = %.3e  (tolerance %.1e)\n",
                 worst, TOLERANCE);

    return pass ? 0 : 1;
}
