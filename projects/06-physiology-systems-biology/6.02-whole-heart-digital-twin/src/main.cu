// ===========================================================================
// src/main.cu  --  Entry point: load ensemble, run CPU + GPU, verify, twin-fit
// ---------------------------------------------------------------------------
// Project 6.2 : Whole-Heart Digital Twin   (REDUCED-SCOPE TEACHING VERSION)
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the ensemble config (a contractility sweep + a clinical target SV).
//   2. CPU reference: simulate every virtual heart serially (reference_cpu.cpp).
//   3. GPU: one thread per heart, the full multi-beat RK4 forward solve (kernels.cu).
//   4. VERIFY: per-member summaries match (same shared physics -> same numbers).
//   5. REPORT: deterministic ensemble table + the fitted twin -> STDOUT;
//              timings + run-varying detail -> STDERR (shown, not diffed).
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Timings (which vary run to run) go to STDERR.
//
//   NOT FOR CLINICAL USE. This is a spatially-lumped (0-D) TEACHING model on
//   SYNTHETIC parameters; it does not represent any real patient.
//
// READ THIS FIRST in the code tour, then heart.h (the physics), kernels.cuh ->
// kernels.cu (the GPU ensemble), and reference_cpu.cpp (the CPU baseline).
// See ../THEORY.md for the science and the GPU mapping.
// ===========================================================================
#include <cmath>      // std::fabs, std::fmax
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // integrate_gpu, EnsembleConfig, TwinResult
#include "reference_cpu.h"    // load_ensemble, integrate_cpu, member_params
#include "util/io.hpp"        // util::CpuTimer

// Program identity (kept in sync with demo/expected_output.txt).
static const char* PROJECT_ID   = "6.2";
static const char* PROJECT_NAME = "Whole-Heart Digital Twin (reduced-scope teaching model)";

// Correctness tolerance. CPU and GPU run the SAME double-precision RK4 over the
// same fixed number of steps via the shared heart.h core, so they should agree
// to a few ULPs. We allow a hair more (1e-9 mL / mmHg) to absorb the GPU's
// fused-multiply-add reassociation, which can differ from the host compiler's
// by ~1e-12 per step (PATTERNS.md section 4). This is far below any physiological
// significance, so the demo's PASS is meaningful, not a rubber stamp.
static constexpr double TOLERANCE = 1.0e-9;

int main(int argc, char** argv) {
    // ---- 1. Load ------------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/heart_ensemble.txt";
    EnsembleConfig c;
    try {
        c = load_ensemble(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int n = ensemble_size(c);

    // ---- 2. CPU reference (timed) ------------------------------------------
    std::vector<TwinResult> res_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    integrate_cpu(c, res_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU ensemble (kernel timed with CUDA events) -------------------
    std::vector<TwinResult> res_gpu;
    float gpu_kernel_ms = 0.0f;
    integrate_gpu(c, res_gpu, &gpu_kernel_ms);

    // ---- 4. Verify GPU == CPU per member -----------------------------------
    // Compare every clinically-meaningful output of every heart; the single
    // worst absolute difference is our headline correctness number.
    double worst = 0.0;
    for (int i = 0; i < n; ++i) {
        worst = std::fmax(worst, std::fabs(res_cpu[i].stroke_vol  - res_gpu[i].stroke_vol));
        worst = std::fmax(worst, std::fabs(res_cpu[i].ejection_fr - res_gpu[i].ejection_fr));
        worst = std::fmax(worst, std::fabs(res_cpu[i].peak_plv    - res_gpu[i].peak_plv));
        worst = std::fmax(worst, std::fabs(res_cpu[i].peak_pao    - res_gpu[i].peak_pao));
    }
    const bool pass = worst <= TOLERANCE;

    // ---- 4b. The "twin fit": pick the member closest to the target SV ------
    // This is the inference step in miniature -- scan the ensemble for the
    // contractility whose forward-simulated stroke volume best matches the
    // clinical target. Deterministic tie-break: the lowest index wins.
    int best = 0;
    double best_err = 1.0e300;
    for (int i = 0; i < n; ++i) {
        const double e = std::fabs(res_gpu[i].stroke_vol - c.target_sv);
        if (e < best_err) { best_err = e; best = i; }
    }
    const HeartParams bp = member_params(c, best);

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("closed-loop 0-D twin: FitzHugh-Nagumo EP + elastance mechanics + 3-element Windkessel\n");
    std::printf("ensemble: %d virtual hearts, contractility E_max %.2f..%.2f mmHg/mL, %d beats @ dt=%.2f ms\n",
                n, c.emax_lo, c.emax_hi, c.beats, c.dt_ms);
    std::printf("SYNTHETIC parameters -- not a real patient. Not for clinical use.\n");
    std::printf("member  Emax(mmHg/mL)  EDV(mL)  ESV(mL)   SV(mL)   EF(%%)  Ppk_lv  Ppk_ao\n");
    for (int i = 0; i < n; ++i) {
        const HeartParams p = member_params(c, i);
        const TwinResult& t = res_gpu[i];
        std::printf("  m%-4d  %11.3f  %7.2f  %7.2f  %7.3f  %5.1f  %6.2f  %6.2f\n",
                    i, p.E_max, t.edv, t.esv, t.stroke_vol,
                    100.0 * t.ejection_fr, t.peak_plv, t.peak_pao);
    }
    std::printf("twin-fit: target SV = %.3f mL -> best member m%d "
                "(Emax=%.3f mmHg/mL, SV=%.3f mL, EF=%.1f%%)\n",
                c.target_sv, best, bp.E_max, res_gpu[best].stroke_vol,
                100.0 * res_gpu[best].ejection_fr);
    std::printf("RESULT: %s (GPU ensemble matches CPU within tol=1.0e-09)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d members)\n", path.c_str(), n);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernel: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- speed-up grows with ensemble size; real twin "
                         "inference runs 10^3-10^6 forward solves.\n");
    std::fprintf(stderr, "[fit]    best-member SV error = %.3e mL\n", best_err);
    std::fprintf(stderr, "[verify] worst per-member diff = %.3e  (tolerance %.1e)\n", worst, TOLERANCE);

    return pass ? 0 : 1;
}
