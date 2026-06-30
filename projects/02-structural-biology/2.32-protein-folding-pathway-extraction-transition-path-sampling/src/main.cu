// ===========================================================================
// src/main.cu  --  Entry point: run TPS on CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 2.32 : Protein Folding Pathway Extraction (Transition Path Sampling)
//                -- a REDUCED-SCOPE teaching version (CLAUDE.md §13). The full
//                research method runs all-atom MD; we run 1-D Brownian dynamics
//                on a double-well free-energy surface. See ../THEORY.md.
//
// 5-step shape (the shape EVERY project in this repo follows):
//   1. Load the simulation parameters (data/sample, or a built-in fallback).
//   2. CPU reference TPS (reference_cpu.cpp).
//   3. GPU TPS (kernels.cu) -- IDENTICAL shooting moves (shared run_shot).
//   4. VERIFY: the integer tallies match EXACTLY (atomics commute on ints).
//   5. REPORT: deterministic transition stats + committor curve to stdout;
//      timing to stderr (so demo/run_demo can diff stdout byte-for-byte).
//
// Code tour: start here, then tps_physics.h (RNG + BD + shooting move),
// kernels.cuh -> kernels.cu, reference_cpu.cpp. The science/GPU-mapping is in
// ../THEORY.md.
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // tps_gpu, TpsProblem, TpsTally
#include "reference_cpu.h"    // load_tps_problem, tps_cpu
#include "util/io.hpp"        // util::CpuTimer

// These tokens identify the program; they stay in sync with expected_output.txt.
static const char* PROJECT_ID   = "2.32";
static const char* PROJECT_NAME = "Protein Folding Pathway Extraction (TPS, 1-D teaching model)";

// Build the built-in synthetic problem used when no data file is supplied. These
// MUST match data/sample/tps_params.txt so the demo's expected_output is stable
// whether or not the sample file is passed.
//   barrier=5kT, centred double well, basins at x0 +/- w, 4096 shooters, 20 bins.
static TpsProblem make_synthetic() {
    TpsProblem p;
    SimParams& s = p.sp;
    s.barrier    = 5.0;       // 5 kT barrier -- a real, rarely-crossed folding barrier
    s.x0         = 0.5;       // landscape centre (transition-state position)
    s.w          = 0.4;       // basin half-separation: A at 0.1, B at 0.9
    s.D          = 1.0;       // reduced diffusion constant
    s.dt         = 0.0005;    // BD timestep (reduced units)
    s.basin_tol  = 0.05;      // within 0.05 of a minimum counts as "arrived"
    s.max_steps  = 20000;     // per-leg step budget (rare-event safety net)
    s.n_shooters = 4096;      // independent shooting moves
    s.n_bins     = 20;        // committor-histogram resolution along x
    s.seed       = 20240517ULL;
    return p;
}

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    TpsProblem prob;
    const char* source = "synthetic (built-in)";
    if (argc > 1) {
        try {
            prob = load_tps_problem(argv[1]);
            source = argv[1];
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[error] %s\n", e.what());
            return 2;
        }
    } else {
        prob = make_synthetic();
    }
    const SimParams& P = prob.sp;

    // ---- 2. CPU reference (timed) -----------------------------------------
    TpsTally tally_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    tps_cpu(prob, tally_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU TPS (kernel timed) ----------------------------------------
    TpsTally tally_gpu;
    float gpu_kernel_ms = 0.0f;
    tps_gpu(prob, tally_gpu, &gpu_kernel_ms);

    // ---- 4. Verify (exact integer match) ----------------------------------
    // Every counter is integer and both sides ran the identical shooting moves,
    // so the GPU tally must equal the CPU tally bit-for-bit. Any mismatch is a
    // real bug (RNG divergence, an atomics error), not floating-point noise.
    int mismatches = 0;
    if (tally_cpu.n_transitions != tally_gpu.n_transitions) ++mismatches;
    if (tally_cpu.n_fwd_to_B    != tally_gpu.n_fwd_to_B)    ++mismatches;
    for (int b = 0; b < P.n_bins; ++b) {
        if (tally_cpu.shots_per_bin[b]     != tally_gpu.shots_per_bin[b])     ++mismatches;
        if (tally_cpu.committed_per_bin[b] != tally_gpu.committed_per_bin[b]) ++mismatches;
    }
    const bool pass = (mismatches == 0);

    // Find the TRANSITION-STATE bin: the first bin whose committor p_B crosses
    // 1/2 (p_B = 0.5 is the rigorous transition-state definition; THEORY §committor).
    // We report it as an integer bin index so stdout stays deterministic.
    int ts_bin = -1;
    for (int b = 0; b < P.n_bins; ++b) {
        long long n = tally_gpu.shots_per_bin[b];
        if (n > 0 && 2 * tally_gpu.committed_per_bin[b] >= n) { ts_bin = b; break; }
    }

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) ----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("double well: barrier=%.1f kT, x0=%.2f, w=%.2f (basin A @ %.2f, basin B @ %.2f)\n",
                P.barrier, P.x0, P.w, P.x0 - P.w, P.x0 + P.w);
    std::printf("shooters=%d, max_steps/leg=%d, dt=%.4f, bins=%d, seed=%llu\n",
                P.n_shooters, P.max_steps, P.dt, P.n_bins,
                static_cast<unsigned long long>(P.seed));
    std::printf("transition paths accepted = %lld of %d shots (%.1f%%)\n",
                tally_gpu.n_transitions, P.n_shooters,
                100.0 * tally_gpu.n_transitions / P.n_shooters);
    std::printf("forward legs committing to folded basin B = %lld\n", tally_gpu.n_fwd_to_B);
    std::printf("transition-state bin (committor p_B first >= 0.5) = %d\n", ts_bin);

    // Committor curve p_B(bin) as a fixed-point percentage so stdout is
    // deterministic (we never print a raw double here). Empty bins print "  -".
    std::printf("committor p_B per bin (%% to folded basin B):\n");
    for (int b = 0; b < P.n_bins; ++b) {
        long long n = tally_gpu.shots_per_bin[b];
        if (n > 0) {
            // Integer-rounded percentage: (100*committed + n/2) / n. Pure integer
            // math => identical on every machine and run.
            long long pct = (100 * tally_gpu.committed_per_bin[b] + n / 2) / n;
            std::printf(" %3lld", pct);
        } else {
            std::printf("   -");
        }
    }
    std::printf("\n");
    std::printf("RESULT: %s (GPU TPS tally matches CPU exactly)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) -----------------
    std::fprintf(stderr, "[data]   source: %s\n", source);
    std::fprintf(stderr, "[timing] CPU TPS: %.3f ms   GPU TPS: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- the GPU edge grows with shooter count; "
                         "real TPS runs thousands of all-atom MD shots.\n");
    std::fprintf(stderr, "[verify] tally mismatches = %d (integer tally => atomics commute)\n",
                 mismatches);

    return pass ? 0 : 1;
}
