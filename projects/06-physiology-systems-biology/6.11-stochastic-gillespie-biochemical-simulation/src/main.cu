// ===========================================================================
// src/main.cu  --  Entry point: load config, run CPU + GPU SSA, verify, report
// ---------------------------------------------------------------------------
// Project 6.11 : Stochastic (Gillespie) Biochemical Simulation
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the ensemble config (data/sample, else a built-in synthetic default).
//   2. CPU reference: run every trajectory serially (reference_cpu.cpp).
//   3. GPU: one thread per trajectory, full SSA loop each (kernels.cu).
//   4. VERIFY: per-trajectory results match EXACTLY (same RNG + same logic).
//   5. REPORT: deterministic ensemble statistics to stdout; timing to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Anything that varies run-to-run (timings) goes to
//   STDERR, which the demo shows but does not diff. Determinism holds because
//   the SSA is seeded from fixed integers and every summary number is either an
//   integer or an exact sum -- no floating-point atomics, no run-to-run reorder.
//
// THE SCIENCE CHECK (beyond CPU==GPU): the sample is a birth-death gene model
//   whose stationary distribution is Poisson(k_prod/k_deg). We therefore also
//   report how close the ensemble mean lands to that analytic value -- validating
//   that the SSA actually samples the right physics (PATTERNS.md section 4).
//
// Code tour: start HERE, then ssa.h (the SSA core), kernels.cu, reference_cpu.cpp.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // simulate_gpu (GPU path), ReactionNetwork
#include "reference_cpu.h"    // EnsembleConfig, load_config, simulate_cpu, analytic_mean
#include "util/io.hpp"        // util::CpuTimer

// Identify the program. Kept in sync with demo/expected_output.txt.
static const char* PROJECT_ID   = "6.11";
static const char* PROJECT_NAME = "Stochastic (Gillespie) Biochemical Simulation";

// Species 0 is the mRNA (M) in the sample birth-death model.
static const int SPECIES_M = 0;

// ---------------------------------------------------------------------------
// default_config: the built-in synthetic problem if no data file is supplied.
//   Birth-death gene expression: production k_prod=10, degradation k_deg=0.5.
//   => analytic stationary mean = k_prod/k_deg = 20 molecules (Poisson(20)).
//   256 trajectories, run long enough (t_end=50, i.e. 25 mean lifetimes) to
//   reach the stationary regime. These EXACT scalars drive expected_output.txt.
// ---------------------------------------------------------------------------
static EnsembleConfig default_config() {
    EnsembleConfig c;
    c.k_prod    = 10.0;
    c.k_deg     = 0.5;
    c.m0        = 0;          // start empty; the ensemble relaxes to the mean
    c.t_end     = 50.0;
    c.n_traj    = 256;
    c.base_seed = 20240611ULL;
    return c;
}

int main(int argc, char** argv) {
    // ---- 1. Load the ensemble config ---------------------------------------
    // If a path is given AND parses, use it; otherwise fall back to the built-in
    // synthetic config (so the program always runs, even with no data file).
    EnsembleConfig c;
    const char* source = "synthetic (built-in)";
    if (argc > 1) {
        try {
            c = load_config(argv[1]);
            source = argv[1];
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[warn] %s -- using built-in synthetic config\n", e.what());
            c = default_config();
        }
    } else {
        c = default_config();
    }
    const int M = c.n_traj;

    // ---- 2. CPU reference (timed) ------------------------------------------
    std::vector<TrajectoryResult> res_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    simulate_cpu(c, res_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU ensemble (kernel timed inside the wrapper) -----------------
    std::vector<TrajectoryResult> res_gpu;
    float gpu_kernel_ms = 0.0f;
    simulate_gpu(c, res_gpu, &gpu_kernel_ms);

    // ---- 4. Verify: EXACT per-trajectory agreement -------------------------
    // Same RNG stream + same integer/step-function math on both sides => every
    // trajectory's final counts, event count, and time-average must be IDENTICAL.
    // final_count and n_events are integers (compared for exact equality);
    // time_avg is a sum of the SAME doubles in the SAME order on both sides, so
    // it too matches bit-for-bit. We report the worst |diff| for honesty.
    bool exact = true;
    double worst_time_avg = 0.0;
    for (int i = 0; i < M; ++i) {
        if (res_cpu[i].final_count[SPECIES_M] != res_gpu[i].final_count[SPECIES_M]) exact = false;
        if (res_cpu[i].n_events != res_gpu[i].n_events) exact = false;
        worst_time_avg = std::fmax(worst_time_avg,
            std::fabs(res_cpu[i].time_avg[SPECIES_M] - res_gpu[i].time_avg[SPECIES_M]));
    }
    // time_avg is expected to be bit-identical (worst == 0); guard with a tiny
    // epsilon only to be robust to any future compiler reassociation.
    const bool pass = exact && (worst_time_avg <= 1.0e-9);

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    // Ensemble statistics from the GPU results (identical to CPU, verified above).
    //   mean_of_timeavg : average over trajectories of each trajectory's own
    //                     time-averaged count -> estimator of the stationary mean.
    //   mean_final      : average final molecule count (a second estimator).
    //   total_events    : summed reaction firings (an integer -> deterministic).
    double sum_timeavg = 0.0, sum_final = 0.0;
    unsigned long long total_events = 0ULL;
    for (int i = 0; i < M; ++i) {
        sum_timeavg += res_gpu[i].time_avg[SPECIES_M];
        sum_final   += static_cast<double>(res_gpu[i].final_count[SPECIES_M]);
        total_events += res_gpu[i].n_events;
    }
    const double mean_timeavg = sum_timeavg / M;
    const double mean_final   = sum_final / M;
    const double target       = analytic_mean(c);   // k_prod / k_deg

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("Model: birth-death gene expression  (0 -> M @ k_prod, M -> 0 @ k_deg)\n");
    std::printf("k_prod=%.3f  k_deg=%.3f  m0=%llu  t_end=%.1f  trajectories=%d\n",
                c.k_prod, c.k_deg, (unsigned long long)c.m0, c.t_end, M);
    std::printf("sample trajectories (idx: events finalM timeAvgM):\n");
    // Print a fixed, deterministic set of trajectory summaries (first, quartiles,
    // last). These are integers / exact sums, so they are reproducible.
    const int picks[5] = {0, M / 4, M / 2, (3 * M) / 4, M - 1};
    for (int s = 0; s < 5; ++s) {
        const int i = picks[s];
        std::printf("  t%-5d: %6llu %7llu %10.4f\n",
                    i, (unsigned long long)res_gpu[i].n_events,
                    (unsigned long long)res_gpu[i].final_count[SPECIES_M],
                    res_gpu[i].time_avg[SPECIES_M]);
    }
    std::printf("ensemble mean of time-avg M = %.4f   (analytic Poisson mean = %.4f)\n",
                mean_timeavg, target);
    std::printf("ensemble mean of final   M = %.4f\n", mean_final);
    std::printf("total reaction events = %llu\n", total_events);
    std::printf("RESULT: %s (GPU ensemble matches CPU exactly, per-trajectory)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%d trajectories, seed=%llu)\n",
                 source, M, (unsigned long long)c.base_seed);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- SSA is divergent (each trajectory "
                         "fires a different number of events); the GPU's edge grows with "
                         "trajectory count.\n");
    std::fprintf(stderr, "[verify] exact per-trajectory match=%s; worst time-avg diff = %.3e\n",
                 exact ? "yes" : "NO", worst_time_avg);
    std::fprintf(stderr, "[science] recovered mean %.4f vs analytic %.4f (relative error %.2f%%)\n",
                 mean_timeavg, target, target > 0 ? 100.0 * std::fabs(mean_timeavg - target) / target : 0.0);

    return pass ? 0 : 1;
}
