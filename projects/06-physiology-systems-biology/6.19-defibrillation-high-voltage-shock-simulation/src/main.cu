// ===========================================================================
// src/main.cu  --  Entry point: load sweep, run CPU + GPU, verify, report DFT
// ---------------------------------------------------------------------------
// Project 6.19 : Defibrillation & High-Voltage Shock Simulation
//                (REDUCED-SCOPE teaching version -- see ../THEORY.md)
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the defibrillation-threshold (DFT) sweep from data/sample.
//   2. CPU reference sweep  (reference_cpu.cpp)             -> trusted answer.
//   3. GPU sweep            (kernels.cu, one thread/shock)  -> the thing taught.
//   4. VERIFY: GPU residual-activity per amplitude matches CPU within tolerance.
//   5. REPORT: deterministic table of (amplitude -> residual, success?) and the
//      recovered DFT to STDOUT; timings to STDERR.
//
//   The science payoff: as shock amplitude increases, residual activity should
//   DROP -- weak shocks fail (tissue stays active), strong shocks defibrillate
//   (tissue quiesces). The smallest successful amplitude is the DEFIBRILLATION
//   THRESHOLD, the central quantity in defibrillator/ICD design.
//
//   STDOUT is byte-for-byte deterministic (fixed-precision doubles, no timings)
//   so demo/run_demo can diff it against demo/expected_output.txt. Timings and
//   run-varying detail go to STDERR (shown, not diffed) -- PATTERNS.md section 3.
//
// READ THIS FIRST in the code tour, then defib.h (the physics), kernels.cuh ->
// kernels.cu (the GPU), reference_cpu.cpp (the baseline). See ../THEORY.md.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // sweep_gpu (GPU path)
#include "reference_cpu.h"    // load_sweep, sweep_cpu, find_dft (CPU baseline)
#include "util/io.hpp"        // util::CpuTimer

// Program self-identification. Kept in sync with demo/expected_output.txt.
static const char* PROJECT_ID   = "6.19";
static const char* PROJECT_NAME = "Defibrillation & High-Voltage Shock Simulation";

// Correctness tolerance. CPU and GPU run the IDENTICAL double-precision
// operations (shared defib.h), differing only by the compiler's freedom to use
// fused multiply-add. Over thousands of forward-Euler steps that FMA divergence
// accumulates to ~1e-9 in the residual metric, so we verify to 1e-6 -- tight
// enough to catch any real bug, honest about the FMA reality (PATTERNS.md §4).
static constexpr double TOLERANCE = 1.0e-6;

int main(int argc, char** argv) {
    // ---- 1. Load the sweep --------------------------------------------------
    const std::string path =
        (argc > 1) ? argv[1] : "data/sample/defib_sweep.txt";
    ShockSweep sweep;
    try {
        sweep = load_sweep(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const FhnParams& p = sweep.p;

    // ---- 2. CPU reference sweep (timed) ------------------------------------
    std::vector<double> res_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    sweep_cpu(sweep, res_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU sweep (kernel timed inside the wrapper) --------------------
    std::vector<double> res_gpu;
    float gpu_kernel_ms = 0.0f;
    sweep_gpu(p, sweep.amps, res_gpu, &gpu_kernel_ms);

    // ---- 4. Verify (per-amplitude residuals agree) -------------------------
    double err = 0.0;
    for (std::size_t k = 0; k < res_cpu.size(); ++k) {
        const double d = std::fabs(res_cpu[k] - res_gpu[k]);
        if (d > err) err = d;
    }
    const bool pass = err <= TOLERANCE;

    // Recover the defibrillation threshold from BOTH result sets. Because the
    // residuals agree, the DFT index agrees too; we report the GPU's.
    const int dft_gpu = find_dft(sweep, res_gpu);

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("1-D monodomain cable (FitzHugh-Nagumo), %s shock\n",
                p.biphasic ? "biphasic" : "monophasic");
    std::printf("cable: %d cells, %d steps, dt=%.4f dx=%.3f D=%.3f\n",
                p.ncell, p.nsteps, p.dt, p.dx, p.D);
    std::printf("shock window: steps [%d,%d), success if residual < %.4f\n",
                p.shock_start, p.shock_start + p.shock_len, sweep.success_thresh);
    std::printf("shock amplitude sweep (residual activity after shock):\n");
    for (std::size_t k = 0; k < sweep.amps.size(); ++k) {
        const bool ok = res_gpu[k] < sweep.success_thresh;
        std::printf("  amp=%.3f  residual=%.6f  %s\n",
                    sweep.amps[k], res_gpu[k], ok ? "DEFIBRILLATED" : "failed");
    }
    if (dft_gpu >= 0)
        std::printf("DFT: amplitude %.3f (index %d) -- weakest shock that terminated activity\n",
                    sweep.amps[static_cast<std::size_t>(dft_gpu)], dft_gpu);
    else
        std::printf("DFT: none of the tested shocks defibrillated the cable\n");
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-06)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%d amplitudes)\n",
                 path.c_str(), static_cast<int>(sweep.amps.size()));
    std::fprintf(stderr, "[timing] CPU sweep: %.3f ms   GPU sweep: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- the GPU's edge grows "
                         "with the number of amplitudes and cable size; this tiny "
                         "sweep is dominated by launch/copy overhead.\n");
    std::fprintf(stderr, "[verify] max residual diff = %.3e  (tolerance %.1e)\n",
                 err, TOLERANCE);

    // Exit code feeds the demo's pass/fail gate.
    return pass ? 0 : 1;
}
