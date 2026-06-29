// ===========================================================================
// src/main.cu  --  Entry point: alchemical delta-G_solv via TI + BAR, verified
// ---------------------------------------------------------------------------
// Project 1.32 : Alchemical Hydration Free Energy (delta-G_solv)
//
// 5-step shape (the standard flagship layout):
//   1. Load the calculation config (lambda windows, walkers, MC steps, system).
//   2. Build the synthetic solvent bath (deterministic geometry).
//   3. CPU reference: run every (window, walker) Metropolis chain serially.
//   4. GPU: one thread per walker, the SAME chain each (kernels.cu).
//   5. VERIFY the GPU per-walker results match the CPU's; REDUCE to per-window
//      <dU/dlambda>; estimate delta-G_solv by Thermodynamic Integration AND by
//      BAR; print a DETERMINISTIC report to stdout and timing to stderr.
//
// Code tour: start here, then alchemy.h (the physics + MC walker), reference_cpu
// (driver + TI/BAR), kernels.cu (the GPU ensemble).
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // run_gpu, AlchConfig, BathStorage, WalkerResult
#include "reference_cpu.h"    // load_config, build_bath, run_cpu, reduce_windows, estimate_*
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "1.32";
static const char* PROJECT_NAME = "Alchemical Hydration Free Energy (dG_solv)";

// CPU and GPU run the IDENTICAL double-precision Metropolis chain per walker, so
// per-walker sums agree to ~round-off. We allow a hair more than machine epsilon
// because each walker accumulates thousands of double adds whose FMA contraction
// may differ between nvcc and the host compiler (PATTERNS.md section 4, the
// "short double-precision" class). 1e-9 on energies of order ~1 is conservative.
static constexpr double TOLERANCE = 1.0e-9;

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/alchemy_config.txt";
    AlchConfig cfg;
    try {
        cfg = load_config(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. Build the synthetic solvent bath -------------------------------
    BathStorage bath = build_bath(cfg.sys, cfg.sys.n_solvent, cfg.bath_seed);
    const int W = total_walkers(cfg);

    // ---- 3. CPU reference (timed) ------------------------------------------
    std::vector<WalkerResult> walk_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    run_cpu(cfg, bath, walk_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 4. GPU ensemble (kernel timed) ------------------------------------
    std::vector<WalkerResult> walk_gpu;
    float gpu_kernel_ms = 0.0f;
    run_gpu(cfg, bath, walk_gpu, &gpu_kernel_ms);

    // ---- 5a. Verify per-walker agreement -----------------------------------
    // Compare the raw accumulators each walker produced; if these match, every
    // downstream average and free energy matches too. We track the worst absolute
    // difference over the dU/dlambda and BAR sums.
    double worst = 0.0;
    for (int i = 0; i < W; ++i) {
        worst = std::fmax(worst, std::fabs(walk_cpu[i].sum_dudl   - walk_gpu[i].sum_dudl));
        worst = std::fmax(worst, std::fabs(walk_cpu[i].sum_du_fwd - walk_gpu[i].sum_du_fwd));
        worst = std::fmax(worst, std::fabs(walk_cpu[i].sum_du_bwd - walk_gpu[i].sum_du_bwd));
    }
    const bool pass = worst <= TOLERANCE;

    // ---- 5b. Reduce to per-window stats and estimate delta-G ----------------
    // Use the GPU results downstream (they passed verification); the CPU's would
    // give the identical numbers. TI integrates <dU/dlambda> over lambda; BAR
    // combines adjacent windows' energy differences -- two independent estimators
    // of the same delta-G_solv, a classic cross-check in free-energy work.
    std::vector<WindowStats> stats = reduce_windows(cfg, walk_gpu);
    const double g_ti  = estimate_ti(cfg, stats);
    const double g_bar = estimate_bar(cfg, stats);

    // ---- 5c. Deterministic report -> STDOUT --------------------------------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("system: %d solvent sites, T=%.3f, eps=%.3f, sigma=%.3f, q=%.3f, alpha_sc=%.3f\n",
                cfg.sys.n_solvent, cfg.sys.temperature, cfg.sys.epsilon,
                cfg.sys.sigma, cfg.sys.q_solute, cfg.sys.alpha_sc);
    std::printf("sampling: %d windows x %d walkers (%d MC equil + %d prod steps each)\n",
                cfg.n_windows, cfg.n_walkers, cfg.n_equil, cfg.n_prod);
    std::printf("lambda schedule (lambda  <dU/dlambda>  accept%%):\n");
    for (int w = 0; w < cfg.n_windows; ++w) {
        std::printf("  w%-2d  lambda=%.4f  dUdl=%+10.4f  acc=%5.1f%%\n",
                    w, stats[w].lambda, stats[w].mean_dudl, 100.0 * stats[w].accept_frac);
    }
    std::printf("delta-G_solv (TI, trapezoid)  = %+9.4f  [reduced eps units]\n", g_ti);
    std::printf("delta-G_solv (BAR, pairwise)  = %+9.4f  [reduced eps units]\n", g_bar);
    std::printf("TI/BAR agreement: |dG_TI - dG_BAR| = %.4f\n", std::fabs(g_ti - g_bar));
    std::printf("RESULT: %s (GPU per-walker sums match CPU within tol=%.1e)\n",
                pass ? "PASS" : "FAIL", TOLERANCE);

    // ---- 5d. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   config: %s  (%d walkers total)\n", path.c_str(), W);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernel: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- the GPU edge grows with windows*walkers; "
                         "real FEP runs many GPUs over thousands of walkers.\n");
    std::fprintf(stderr, "[verify] worst per-walker |diff| = %.3e  (tolerance %.1e)\n", worst, TOLERANCE);

    return pass ? 0 : 1;
}
