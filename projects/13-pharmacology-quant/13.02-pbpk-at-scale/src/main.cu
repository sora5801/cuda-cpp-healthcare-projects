// ===========================================================================
// src/main.cu  --  Entry point: integrate a virtual population, verify, report
// ---------------------------------------------------------------------------
// Project 13.02 : PBPK at Scale
//
// 5-step shape:
//   1. Load the population config (model medians + variability).
//   2. CPU reference: integrate every virtual patient (reference_cpu.cpp).
//   3. GPU: one thread per patient, full RK4 loop each (kernels.cu).
//   4. VERIFY: per-patient exposure metrics match (shared RNG + RK4).
//   5. REPORT: deterministic sample patients + population exposure summary.
//
// Code tour: start here, then pbpk.h (model + RK4 + sampling), kernels.cu,
// reference_cpu.cpp.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // integrate_gpu, PbpkParams, PatientResult
#include "reference_cpu.h"    // load_pbpk, integrate_cpu
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "13.2";
static const char* PROJECT_NAME = "PBPK at Scale";

// Double-precision RK4: CPU and GPU run identical ops -> agree to ~round-off.
static constexpr double TOLERANCE = 1.0e-6;

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/pbpk_params.txt";
    PbpkParams P;
    try {
        P = load_pbpk(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int M = P.n_patients;

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<PatientResult> res_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    integrate_cpu(P, res_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU population (kernel timed) ---------------------------------
    std::vector<PatientResult> res_gpu;
    float gpu_kernel_ms = 0.0f;
    integrate_gpu(P, res_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    double worst = 0.0;
    for (int i = 0; i < M; ++i) {
        worst = std::fmax(worst, std::fabs(res_cpu[i].Cmax - res_gpu[i].Cmax));
        worst = std::fmax(worst, std::fabs(res_cpu[i].AUC  - res_gpu[i].AUC));
        worst = std::fmax(worst, std::fabs(res_cpu[i].Tmax - res_gpu[i].Tmax));
    }
    const bool pass = worst <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    double sum_cmax = 0.0, sum_auc = 0.0, min_auc = res_gpu[0].AUC, max_auc = res_gpu[0].AUC;
    for (int i = 0; i < M; ++i) {
        sum_cmax += res_gpu[i].Cmax;
        sum_auc  += res_gpu[i].AUC;
        min_auc = std::fmin(min_auc, res_gpu[i].AUC);
        max_auc = std::fmax(max_auc, res_gpu[i].AUC);
    }
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("PBPK population: %d virtual patients, 3 compartments, %d h @ dt=%.2f h\n",
                M, static_cast<int>(P.steps * P.dt), P.dt);
    std::printf("dose=%.0f mg, median ka=%.2f CL=%.2f Vc=%.1f Vp=%.1f Q=%.2f, CV=%.0f%%\n",
                P.dose, P.ka, P.CL, P.Vc, P.Vp, P.Q, 100.0 * P.cv);
    std::printf("sample patients (Cmax mg/L, Tmax h, AUC mg.h/L):\n");
    const int picks[5] = {0, M / 4, M / 2, (3 * M) / 4, M - 1};
    for (int s = 0; s < 5; ++s) {
        const int i = picks[s];
        std::printf("  patient %-5d: Cmax=%.4f  Tmax=%.2f  AUC=%.4f\n",
                    i, res_gpu[i].Cmax, res_gpu[i].Tmax, res_gpu[i].AUC);
    }
    std::printf("population: mean Cmax=%.4f, mean AUC=%.4f, AUC range [%.4f, %.4f]\n",
                sum_cmax / M, sum_auc / M, min_auc, max_auc);
    std::printf("RESULT: %s (GPU population matches CPU within tol=1.0e-06)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d patients)\n", path.c_str(), M);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- speed-up grows with population x compound count; "
                         "QSP/PBPK studies run 10^4-10^6 ODE solves.\n");
    std::fprintf(stderr, "[verify] worst per-patient diff = %.3e  (tolerance %.1e)\n", worst, TOLERANCE);

    return pass ? 0 : 1;
}
