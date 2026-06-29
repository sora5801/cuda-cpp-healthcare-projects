// ===========================================================================
// src/main.cu  --  Entry point: FEP/TI free-energy estimate, verified two ways
// ---------------------------------------------------------------------------
// Project 1.5 : Free Energy Perturbation / Thermodynamic Integration
//
// WHAT THIS PROGRAM COMPUTES
//   The free-energy difference DeltaG between two harmonic "states" A and B of a
//   1-D model system, via Thermodynamic Integration (TI) over an alchemical
//   lambda-pathway. For each of W lambda-windows we estimate the equilibrium
//   average < dU/dlambda > by Metropolis Monte Carlo, then integrate those over
//   lambda with the trapezoid rule:  DeltaG_TI = integral_0^1 < dU/dlambda > dlambda.
//   (See ../THEORY.md for the science, the math, and why this is reduced-scope.)
//
// THE 5-STEP SHAPE EVERY PROJECT IN THIS REPO FOLLOWS
//   1. Load the AlchemyConfig (from data/sample, or a built-in fallback).
//   2. CPU reference: sample every window serially            (reference_cpu.cpp).
//   3. GPU: one thread per window, same MC chain each         (kernels.cu).
//   4. VERIFY twice:
//        (a) GPU per-window averages match the CPU reference  (same RNG+math),
//        (b) the TI estimate matches the CLOSED-FORM DeltaG    (the science).
//   5. REPORT: deterministic result -> stdout; timing/diagnostics -> stderr.
//
//   stdout is kept byte-for-byte deterministic (fixed precision; counter-based
//   RNG) so demo/run_demo can diff it against demo/expected_output.txt. Anything
//   that varies run-to-run (wall-clock timings) goes to stderr.
//
// Code tour: start here, then alchemy.h (model + RNG + MC sampler), kernels.cu
// (the GPU ensemble), reference_cpu.cpp (the serial baseline).
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // integrate_gpu, AlchemyConfig
#include "reference_cpu.h"    // load_config, integrate_cpu
#include "alchemy.h"          // trapezoid_ti, analytic_delta_g, window_lambda
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "1.5";
static const char* PROJECT_NAME = "Free Energy Perturbation / Thermodynamic Integration";

// --- Verification tolerances (PATTERNS.md §4: be honest about floating point) -
// CPU and GPU run the IDENTICAL double-precision MC chain with the IDENTICAL
// counter-based RNG, so per-window averages agree to round-off. We allow a tiny
// slack for the only legitimate divergence: the device and host math libraries
// (exp) and FMA contraction can differ in the last ~1-2 ulp, which over a long
// chain stays far below 1e-9. So this is "essentially exact".
static constexpr double TOL_CPU_GPU = 1.0e-9;
// The TI estimate is a STATISTICAL quantity (finite MC sampling + trapezoid
// discretisation of lambda), so it only approaches the analytic DeltaG. The
// committed sample is engineered so the bias is small; we accept agreement to
// this physical tolerance and PRINT the gap so the learner sees the convergence.
static constexpr double TOL_TI_ANALYTIC = 5.0e-2;

// Built-in synthetic problem used when no data file is supplied. Chosen so the
// answer is interpretable: stiffening the spring from kA=1 to kB=4 at kT=1 gives
// the clean analytic DeltaG = 1/2 * ln(4) = ln(2) ~ 0.693147. (Same numbers as
// data/sample/alchemy_sample.txt so stdout matches with or without an argument.)
static AlchemyConfig make_synthetic() {
    AlchemyConfig c;
    c.kA = 1.0;  c.x0A = 0.0;        // state A: soft spring at origin
    c.kB = 4.0;  c.x0B = 1.0;        // state B: 4x stiffer spring, shifted to x=1
    c.kT = 1.0;                      // temperature in energy units (kB=1)
    c.windows = 11;                  // lambda = 0.0, 0.1, ..., 1.0
    c.equil   = 2000;                // burn-in MC steps (discarded)
    c.samples = 20000;               // averaged MC steps
    c.step    = 0.6;                 // trial-move half-width (~50% acceptance)
    c.x_init  = 0.0;                 // every chain starts at x=0
    return c;
}

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    AlchemyConfig c;
    const char* source;
    if (argc > 1) {
        try {
            c = load_config(argv[1]);
            source = argv[1];
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[error] %s\n", e.what());
            return 2;
        }
    } else {
        c = make_synthetic();
        source = "synthetic (built-in)";
    }
    const int W = n_windows(c);

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<double> dvals_cpu;
    std::vector<long long> acc_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    integrate_cpu(c, dvals_cpu, acc_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU ensemble (kernel timed) -----------------------------------
    std::vector<double> dvals_gpu;
    std::vector<long long> acc_gpu;
    float gpu_kernel_ms = 0.0f;
    integrate_gpu(c, dvals_gpu, acc_gpu, &gpu_kernel_ms);

    // ---- 4a. Verify GPU == CPU (per window) -------------------------------
    double worst = 0.0;
    for (int w = 0; w < W; ++w)
        worst = std::fmax(worst, std::fabs(dvals_cpu[w] - dvals_gpu[w]));
    const bool pass_cpu_gpu = worst <= TOL_CPU_GPU;

    // ---- 4b. TI integral + analytic check ---------------------------------
    // Integrate the (GPU) per-window averages over lambda -> DeltaG_TI.
    const double dG_ti       = trapezoid_ti(dvals_gpu.data(), W);
    const double dG_analytic = analytic_delta_g(c);
    const double ti_err      = std::fabs(dG_ti - dG_analytic);
    const bool pass_ti       = ti_err <= TOL_TI_ANALYTIC;

    const bool pass = pass_cpu_gpu && pass_ti;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) ----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("alchemical TI: stateA(k=%.3f) -> stateB(k=%.3f) at kT=%.3f, %d windows\n",
                c.kA, c.kB, c.kT, W);
    std::printf("MC sampling: %d equil + %d samples per window, step=%.3f\n",
                c.equil, c.samples, c.step);
    std::printf("TI curve <dU/dlambda> per window (lambda -> mean):\n");
    for (int w = 0; w < W; ++w) {
        std::printf("  w%-2d lambda=%.2f  <dU/dlambda>=%+10.5f\n",
                    w, window_lambda(c, w), dvals_gpu[w]);
    }
    std::printf("DeltaG_TI       = %+.5f  (trapezoid over lambda)\n", dG_ti);
    std::printf("DeltaG_analytic = %+.5f  (= 1/2 kT ln(kB/kA))\n", dG_analytic);
    std::printf("RESULT: %s (GPU==CPU within %.0e; TI within %.0e of analytic)\n",
                pass ? "PASS" : "FAIL", TOL_CPU_GPU, TOL_TI_ANALYTIC);

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) -----------------
    long long acc_total = 0, steps_total = 0;
    for (int w = 0; w < W; ++w) {
        acc_total   += acc_gpu[w];
        steps_total += static_cast<long long>(c.equil) + c.samples;
    }
    std::fprintf(stderr, "[data]   source: %s  (%d windows)\n", source, W);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- few windows are "
                         "launch-bound; the GPU edge grows with #windows/chain length.\n");
    std::fprintf(stderr, "[mc]     overall MC acceptance = %.1f%% (tune `step` to "
                         "trade acceptance vs exploration)\n",
                 steps_total ? 100.0 * acc_total / steps_total : 0.0);
    std::fprintf(stderr, "[verify] worst |CPU-GPU| per window = %.3e (tol %.1e)\n",
                 worst, TOL_CPU_GPU);
    std::fprintf(stderr, "[verify] |TI - analytic|            = %.3e (tol %.1e)\n",
                 ti_err, TOL_TI_ANALYTIC);

    return pass ? 0 : 1;
}
