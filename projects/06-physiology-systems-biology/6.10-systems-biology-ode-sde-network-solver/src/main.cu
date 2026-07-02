// ===========================================================================
// src/main.cu  --  Entry point: integrate a GRN ensemble, verify, report
// ---------------------------------------------------------------------------
// Project 6.10 : Systems-Biology ODE/SDE Network Solver
//
// 5-step shape (the shape every project in this repo follows):
//   1. Load the ensemble config (an alpha x n parameter sweep of the
//      repressilator gene circuit) from data/sample, or a built-in fallback.
//   2. CPU reference: integrate every member serially (reference_cpu.cpp).
//   3. GPU: one thread per member, full RK4 loop each (kernels.cu).
//   4. VERIFY: per-member results match (same RK4 arithmetic -> same numbers).
//   5. REPORT: deterministic sample members + ensemble summary -> STDOUT;
//              data source + timing + verification detail          -> STDERR.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Run-varying numbers (timings) go to STDERR, which
//   the demo shows but does not diff.
//
// Code tour: start HERE, then grn.h (the ODE + RK4 + oscillation summary),
// reference_cpu.h/.cpp (the sweep + serial baseline), kernels.cuh/.cu (the GPU
// twin). See ../THEORY.md for the science and the GPU mapping.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels.cuh"        // integrate_gpu, EnsembleConfig, MemberResult
#include "reference_cpu.h"    // load_ensemble, integrate_cpu, member_params
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "6.10";
static const char* PROJECT_NAME = "Systems-Biology ODE/SDE Network Solver";

// Verification tolerance. CPU and GPU run the SAME explicit double-precision RK4
// (grn.h), so their continuous outputs agree to ~machine precision; over a few
// hundred steps the fused-multiply-add differences accumulate to well under
// 1e-9. We check to 1e-9 (PATTERNS.md §4: "~machine precision for short
// double-precision computations"). The integer oscillation flags must match
// EXACTLY -- see the check below.
static constexpr double TOLERANCE = 1.0e-9;

// Build the built-in synthetic ensemble used when no data file is supplied.
// Matches data/sample/ensemble_params.txt so the demo output is identical
// whether or not the file is present. See data/README.md for field meanings.
static EnsembleConfig make_synthetic() {
    EnsembleConfig c;
    c.alpha0 = 1.0;      // basal leak
    c.beta   = 5.0;      // protein/mRNA decay ratio (repressilator regime)
    c.dt     = 0.05;     // timestep in mRNA lifetimes
    c.steps  = 4000;     // 200 mRNA-lifetime units of simulated time
    c.na     = 6;        // alpha sweep points
    c.nn     = 6;        // Hill-coefficient sweep points
    c.alpha_lo = 10.0; c.alpha_hi = 260.0;   // max transcription rate
    c.n_lo     = 1.0;  c.n_hi     = 3.0;     // Hill cooperativity
    // Asymmetric seed (breaks ring symmetry): m0=1, everything else 0.
    for (int j = 0; j < STATE_DIM; ++j) c.s0[j] = 0.0;
    c.s0[0] = 1.0;
    return c;
}

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    // Prefer the file argument; fall back to the built-in synthetic ensemble so
    // the program always runs (and prints the SAME thing) even with no data.
    EnsembleConfig c;
    const char* source = "synthetic (built-in)";
    if (argc > 1) {
        try {
            c = load_ensemble(argv[1]);
            source = argv[1];
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[error] %s\n", e.what());
            return 2;
        }
    } else {
        c = make_synthetic();
    }
    const int M = ensemble_size(c);

    // ---- 2. CPU reference (timed) ------------------------------------------
    std::vector<MemberResult> res_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    integrate_cpu(c, res_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU ensemble (kernel timed) ------------------------------------
    std::vector<MemberResult> res_gpu;
    float gpu_kernel_ms = 0.0f;
    integrate_gpu(c, res_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    // Continuous observables must agree within TOLERANCE; the integer
    // oscillation flag must match EXACTLY (any disagreement is a real bug).
    double worst = 0.0;
    bool flags_match = true;
    for (int i = 0; i < M; ++i) {
        worst = std::fmax(worst, std::fabs(res_cpu[i].p2_final - res_gpu[i].p2_final));
        worst = std::fmax(worst, std::fabs(res_cpu[i].p2_max   - res_gpu[i].p2_max));
        worst = std::fmax(worst, std::fabs(res_cpu[i].p2_min   - res_gpu[i].p2_min));
        if (res_cpu[i].oscillates != res_gpu[i].oscillates) flags_match = false;
    }
    const bool pass = (worst <= TOLERANCE) && flags_match;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    // Ensemble headline: how many members oscillate (the emergent genetic clock).
    int osc = 0;
    for (int i = 0; i < M; ++i) osc += res_gpu[i].oscillates;

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("repressilator ensemble: %d members (%d alpha x %d n), "
                "%d steps @ dt=%.3f (T=%.0f), beta=%.1f alpha0=%.1f\n",
                M, c.na, c.nn, c.steps, c.dt, c.steps * c.dt, c.beta, c.alpha0);
    std::printf("sample members (alpha n -> p2_final p2_min p2_max crossings osc):\n");
    // Deterministic picks spanning the sweep (first, quarter, mid, 3/4, last).
    const int picks[5] = {0, M / 4, M / 2, (3 * M) / 4, M - 1};
    for (int s = 0; s < 5; ++s) {
        const int i = picks[s];
        GrnParams pr; member_params(c, i, pr);
        std::printf("  m%-3d: %6.1f %4.2f -> %9.4f %9.4f %9.4f %3d %d\n",
                    i, pr.alpha, pr.n,
                    res_gpu[i].p2_final, res_gpu[i].p2_min, res_gpu[i].p2_max,
                    res_gpu[i].zero_cross, res_gpu[i].oscillates);
    }
    std::printf("ensemble: %d/%d members sustain oscillations\n", osc, M);
    std::printf("RESULT: %s (GPU ensemble matches CPU within tol=1.0e-09)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%d members)\n", source, M);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- the GPU's edge grows with "
                         "ensemble size; real sweeps/UQ run 10^4-10^6 members.\n");
    std::fprintf(stderr, "[verify] worst continuous diff = %.3e (tol %.1e); osc flags match: %s\n",
                 worst, TOLERANCE, flags_match ? "yes" : "NO");

    return pass ? 0 : 1;
}
