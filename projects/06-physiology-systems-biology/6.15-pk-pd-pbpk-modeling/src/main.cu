// ===========================================================================
// src/main.cu  --  Entry point: integrate a PK/PD virtual population, verify, report
// ---------------------------------------------------------------------------
// Project 6.15 : PK/PD & PBPK Modeling
//
// WHAT THIS FILE DOES  (the 5-step shape EVERY project in this repo follows)
//   1. Load the population config (PK/PD medians + between-subject variability).
//   2. CPU reference: integrate every virtual patient serially (reference_cpu.cpp).
//   3. GPU: one thread per patient, full coupled-PK/PD RK4 loop each (kernels.cu).
//   4. VERIFY: per-patient PK exposure + PD effect metrics match (shared RNG+RK4).
//   5. REPORT: deterministic sample patients + population summary -> STDOUT;
//              run-varying timings -> STDERR (shown, not diffed by the demo).
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt (PATTERNS.md §3). We use %.4f fixed formatting and
//   the same double-precision math on both sides.
//
// CODE TOUR: start here, then pkpd.h (the coupled PK/PD model + RK4 + sampling),
// kernels.cuh -> kernels.cu (the GPU population), reference_cpu.cpp (the baseline).
// See ../THEORY.md for the science/math/GPU-mapping.
// ===========================================================================
#include <cmath>     // std::fabs, std::fmax, std::fmin
#include <cstdio>    // std::printf, std::fprintf
#include <string>
#include <vector>

#include "kernels.cuh"        // integrate_gpu, PkPdParams, PatientResult
#include "reference_cpu.h"    // load_pkpd, integrate_cpu
#include "util/io.hpp"        // util::CpuTimer

// Program identity (kept in sync with demo/expected_output.txt's first line).
static const char* PROJECT_ID   = "6.15";
static const char* PROJECT_NAME = "PK/PD & PBPK Modeling";

// Correctness tolerance. CPU and GPU run the IDENTICAL double-precision RK4 on
// the IDENTICAL sampled patients (shared pkpd.h), so per-patient metrics agree to
// ~machine precision. We verify to 1e-6 -- comfortably above the ~1e-12 we
// actually observe, but tight enough to catch any real divergence. The GPU's
// fused multiply-add can differ from the host in the last bits over ~10^3 steps;
// 1e-6 is a small, honest, physically-negligible tolerance (PATTERNS.md §4).
static constexpr double TOLERANCE = 1.0e-6;

int main(int argc, char** argv) {
    // ---- 1. Load the population config -------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/pkpd_params.txt";
    PkPdParams P;
    try {
        P = load_pkpd(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;   // config error: distinct from a verification FAIL (exit 1)
    }
    const int M = P.n_patients;

    // ---- 2. CPU reference (timed) ------------------------------------------
    std::vector<PatientResult> res_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    integrate_cpu(P, res_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU population (kernel timed inside the wrapper) ---------------
    std::vector<PatientResult> res_gpu;
    float gpu_kernel_ms = 0.0f;
    integrate_gpu(P, res_gpu, &gpu_kernel_ms);

    // ---- 4. Verify: worst per-patient difference over ALL metrics ----------
    //   Comparing every PK (Cmax/Tmax/AUC) and PD (Rmax/Tresp/effect) metric of
    //   every patient is the strongest CPU==GPU check; the single worst absolute
    //   difference is the headline correctness number.
    double worst = 0.0;
    for (int i = 0; i < M; ++i) {
        worst = std::fmax(worst, std::fabs(res_cpu[i].Cmax   - res_gpu[i].Cmax));
        worst = std::fmax(worst, std::fabs(res_cpu[i].AUC    - res_gpu[i].AUC));
        worst = std::fmax(worst, std::fabs(res_cpu[i].Tmax   - res_gpu[i].Tmax));
        worst = std::fmax(worst, std::fabs(res_cpu[i].Rmax   - res_gpu[i].Rmax));
        worst = std::fmax(worst, std::fabs(res_cpu[i].Tresp  - res_gpu[i].Tresp));
        worst = std::fmax(worst, std::fabs(res_cpu[i].effect - res_gpu[i].effect));
    }
    const bool pass = worst <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    //   Population summary statistics computed on the (deterministic) GPU results.
    //   R0 = kin/kout is the shared biomarker baseline every patient starts from.
    const double R0 = P.kin / P.kout;
    double sum_cmax = 0.0, sum_auc = 0.0, sum_eff = 0.0;
    double min_auc = res_gpu[0].AUC, max_auc = res_gpu[0].AUC;
    for (int i = 0; i < M; ++i) {
        sum_cmax += res_gpu[i].Cmax;
        sum_auc  += res_gpu[i].AUC;
        sum_eff  += res_gpu[i].effect;
        min_auc = std::fmin(min_auc, res_gpu[i].AUC);
        max_auc = std::fmax(max_auc, res_gpu[i].AUC);
    }

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("coupled 1-cpt oral PK + indirect-response PD; %d virtual patients, "
                "%d h @ dt=%.2f h\n", M, static_cast<int>(P.steps * P.dt), P.dt);
    std::printf("dose=%.0f mg, median ka=%.2f/h CL=%.2f L/h Vc=%.1f L; "
                "PD kin=%.1f kout=%.2f/h Imax=%.2f IC50=%.2f mg/L (baseline R0=%.2f), CV=%.0f%%\n",
                P.dose, P.ka, P.CL, P.Vc, P.kin, P.kout, P.Imax, P.IC50, R0, 100.0 * P.cv);
    std::printf("sample patients (Cmax mg/L, Tmax h, AUC mg.h/L | Rmax, Tresp h, effect):\n");
    // Five evenly-spaced patients: deterministic, index-based picks.
    const int picks[5] = {0, M / 4, M / 2, (3 * M) / 4, M - 1};
    for (int s = 0; s < 5; ++s) {
        const int i = picks[s];
        std::printf("  patient %-5d: Cmax=%.4f Tmax=%.2f AUC=%.4f | Rmax=%.4f Tresp=%.2f effect=%.4f\n",
                    i, res_gpu[i].Cmax, res_gpu[i].Tmax, res_gpu[i].AUC,
                    res_gpu[i].Rmax, res_gpu[i].Tresp, res_gpu[i].effect);
    }
    std::printf("population: mean Cmax=%.4f mg/L, mean AUC=%.4f mg.h/L, AUC range [%.4f, %.4f]\n",
                sum_cmax / M, sum_auc / M, min_auc, max_auc);
    std::printf("population: mean PD effect (peak fractional rise above baseline)=%.4f\n",
                sum_eff / M);
    std::printf("RESULT: %s (GPU population matches CPU within tol=1.0e-06)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%d patients)\n", path.c_str(), M);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- population PK/PD studies "
                         "run 10^4-10^6 ODE solves where the GPU's edge grows.\n");
    std::fprintf(stderr, "[verify] worst per-patient diff = %.3e  (tolerance %.1e)\n",
                 worst, TOLERANCE);

    return pass ? 0 : 1;   // exit code feeds the demo's pass/fail gate
}
