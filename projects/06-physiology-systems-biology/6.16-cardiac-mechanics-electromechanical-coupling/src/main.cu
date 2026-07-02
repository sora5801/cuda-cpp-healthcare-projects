// ===========================================================================
// src/main.cu  --  Entry point: integrate the heart ensemble, verify, report
// ---------------------------------------------------------------------------
// Project 6.16 : Cardiac Mechanics & Electromechanical Coupling
//
// 5-step shape (every project in this repo follows it):
//   1. Load the ensemble config (a contractility x afterload sweep).
//   2. CPU reference: integrate every virtual heart serially (reference_cpu.cpp).
//   3. GPU: one thread per heart, full multi-beat RK4 loop each (kernels.cu).
//   4. VERIFY: per-heart PV-loop summaries match (same RK4 -> same numbers).
//   5. REPORT: deterministic sample hearts + ensemble summary to STDOUT;
//              timing + run-varying detail to STDERR (so the demo can diff
//              only the deterministic part).
//
// Code tour: start here, then cardiac.h (the physics + RK4), reference_cpu.h/.cpp
// (config + CPU baseline), kernels.cuh -> kernels.cu (the GPU twin). ../THEORY.md
// has the "why".
//
//   NOT FOR CLINICAL USE. Synthetic, illustrative model (see README/THEORY).
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // integrate_gpu, EnsembleConfig, CycleResult
#include "reference_cpu.h"    // load_ensemble, integrate_cpu, member_params
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "6.16";
static const char* PROJECT_NAME = "Cardiac Mechanics & Electromechanical Coupling";

// Verification tolerance -- an HONEST, documented physical tolerance (PATTERNS.md
// section 4). CPU and GPU run the SAME double-precision RK4 on the SAME shared
// cardiac.h code, but this is a LONG iterative solver: ~80,000 RK4 steps per
// heart. The GPU's fused-multiply-add (FMA) contracts a*b+c into one rounding
// while the host compiler uses two, so per-step the two trajectories drift by
// ~1e-15 and, compounded over 80k steps AND through non-smooth valve switches,
// reach ~1e-3..1e-2 in the recorded scalars. The effect is largest for P_peak,
// which is a MAX over discrete samples: a sub-microvolt trajectory shift can
// move which timestep is the peak, showing up as a ~1e-2 mmHg jump. That is a
// REAL lesson, not a bug. We verify the PV-loop outputs (mL, mmHg, %) to a
// physically-negligible 0.1 (relative ~1e-3 on ~80-mmHg / ~100-mL / ~50-%
// quantities) -- and say so plainly rather than pretend the results are
// bit-identical. THEORY.md "Numerical considerations" derives this.
static constexpr double TOLERANCE = 1.0e-1;

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1]
                                         : "data/sample/heart_ensemble.txt";
    EnsembleConfig c;
    try {
        c = load_ensemble(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int M = ensemble_size(c);

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<CycleResult> res_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    integrate_cpu(c, res_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU ensemble (kernel timed) -----------------------------------
    std::vector<CycleResult> res_gpu;
    float gpu_kernel_ms = 0.0f;
    integrate_gpu(c, res_gpu, &gpu_kernel_ms);

    // ---- 4. Verify (worst per-heart absolute diff over EF, SV, P_peak) -----
    double worst = 0.0;
    double wEF = 0.0, wSV = 0.0, wPP = 0.0;
    for (int i = 0; i < M; ++i) {
        wEF = std::fmax(wEF, std::fabs(res_cpu[i].EF_percent  - res_gpu[i].EF_percent));
        wSV = std::fmax(wSV, std::fabs(res_cpu[i].SV_mL       - res_gpu[i].SV_mL));
        wPP = std::fmax(wPP, std::fabs(res_cpu[i].P_peak_mmHg - res_gpu[i].P_peak_mmHg));
    }
    worst = std::fmax(wEF, std::fmax(wSV, wPP));
    const bool pass = worst <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    // Ensemble summary: mean ejection fraction, and the highest/lowest-EF hearts
    // (which fall out of the contractility/afterload sweep -- high contractility
    // + low afterload gives the best EF, and vice-versa).
    double sum_ef = 0.0;
    int best = 0, worst_ef_idx = 0;
    for (int i = 0; i < M; ++i) {
        sum_ef += res_gpu[i].EF_percent;
        if (res_gpu[i].EF_percent > res_gpu[best].EF_percent)         best = i;
        if (res_gpu[i].EF_percent < res_gpu[worst_ef_idx].EF_percent) worst_ef_idx = i;
    }
    const double mean_ef = (M > 0) ? sum_ef / M : 0.0;

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("[reduced-scope teaching model: 0-D electromechanics + Windkessel; "
                "NOT FOR CLINICAL USE]\n");
    std::printf("ensemble: %d hearts (%d contractility x %d afterload), "
                "%d beats @ dt=%.2f ms, %d steps/beat\n",
                M, c.nT, c.nR, c.n_beats, c.dt_ms, c.steps_per_beat);
    std::printf("sample hearts (Tref[mmHg/mL] Rsys[mmHg.ms/mL] -> EDV ESV SV[mL] EF%% Ppeak[mmHg]):\n");

    // Deterministic picks spanning the sweep corners + center.
    const int picks[5] = {0, M / 4, M / 2, (3 * M) / 4, M - 1};
    for (int s = 0; s < 5; ++s) {
        const int i = picks[s];
        const HeartParams p = member_params(c, i);
        const CycleResult& r = res_gpu[i];
        std::printf("  h%-5d: %6.2f %8.4f -> %6.2f %6.2f %6.2f %6.2f %8.2f\n",
                    i, p.Tref, p.R_sys,
                    r.EDV_mL, r.ESV_mL, r.SV_mL, r.EF_percent, r.P_peak_mmHg);
    }

    // Highlight the best/worst ejection fraction across the sweep.
    {
        const HeartParams pb = member_params(c, best);
        const HeartParams pw = member_params(c, worst_ef_idx);
        std::printf("best EF : h%-5d (Tref=%.2f R=%.4f) EF=%6.2f%% SV=%6.2f mL\n",
                    best, pb.Tref, pb.R_sys, res_gpu[best].EF_percent, res_gpu[best].SV_mL);
        std::printf("worst EF: h%-5d (Tref=%.2f R=%.4f) EF=%6.2f%% SV=%6.2f mL\n",
                    worst_ef_idx, pw.Tref, pw.R_sys,
                    res_gpu[worst_ef_idx].EF_percent, res_gpu[worst_ef_idx].SV_mL);
    }
    std::printf("ensemble: mean EF = %.2f%%\n", mean_ef);
    std::printf("RESULT: %s (GPU ensemble matches CPU within tol=1.0e-01)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d hearts)\n", path.c_str(), M);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernel: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- speed-up grows with ensemble size; "
                         "a real solver batches an ODE over millions of Gauss points.\n");
    std::fprintf(stderr, "[verify] worst per-heart diff = %.3e  (tolerance %.1e)\n", worst, TOLERANCE);

    return pass ? 0 : 1;
}
