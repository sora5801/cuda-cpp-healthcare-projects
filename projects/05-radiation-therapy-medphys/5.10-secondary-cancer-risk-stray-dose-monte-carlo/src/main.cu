// ===========================================================================
// src/main.cu  --  Entry point: run stray-dose MC on CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 5.10 : Secondary Cancer Risk & Stray-Dose Monte Carlo
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the phantom + beam + variance-reduction problem (data/sample).
//   2. CPU reference Monte Carlo (reference_cpu.cpp)          -> trusted answer.
//   3. GPU Monte Carlo (kernels.cu) -- IDENTICAL histories     -> the thing taught.
//   4. VERIFY: the fixed-point per-organ dose tallies match EXACTLY (integer
//      atomics commute) -> correctness with zero tolerance.
//   5. REPORT: deterministic per-organ stray dose + BEIR-VII secondary-cancer
//      risk to stdout; timing to stderr.
//
//   STDOUT is kept byte-for-byte deterministic (integer fixed-point dose printed
//   exactly; risk derived from those exact integers with fixed precision) so
//   demo/run_demo can diff it against demo/expected_output.txt. Anything that
//   varies run-to-run (wall-clock timings) goes to STDERR, shown but not diffed.
//
// READ THIS FIRST in the code tour, then stray_physics.h -> kernels.cu, and
// reference_cpu.cpp for the baseline. See ../THEORY.md for the "why".
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // dose_gpu, StrayProblem
#include "reference_cpu.h"    // load_stray_problem, stray_cpu, Organ
#include "risk_model.h"       // organ_lar, fixed_to_dose
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "5.10";
static const char* PROJECT_NAME = "Secondary Cancer Risk & Stray-Dose Monte Carlo";

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/phantom.txt";
    StrayProblem prob;
    try {
        prob = load_stray_problem(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const SimParams& sp = prob.sp;

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<unsigned long long> dose_c;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    stray_cpu(prob, dose_c);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU Monte Carlo (kernel timed) --------------------------------
    std::vector<unsigned long long> dose_g;
    float gpu_kernel_ms = 0.0f;
    dose_gpu(prob, dose_g, &gpu_kernel_ms);

    // ---- 4. Verify (EXACT integer match; atomics commute on fixed-point) ---
    int mismatches = 0;
    for (int o = 0; o < sp.n_organs; ++o)
        if (dose_c[o] != dose_g[o]) ++mismatches;
    const bool pass = (mismatches == 0);

    // Reference the target's in-field dose to express stray dose as a ratio.
    // Organ 0 is the treated target; its dose is the large primary dose.
    const unsigned long long target_fixed = dose_g.empty() ? 0ULL : dose_g[0];

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("phantom: %d organs, field ends at organ %d, mu=%.3f /cm, organ=%.1f cm\n",
                sp.n_organs, sp.field_end, sp.mu, sp.organ_cm);
    std::printf("histories = %llu, VR = survival-biasing + forced-detection + roulette\n",
                sp.n_histories);
    std::printf("scatter_frac=%.2f sidescatter=%.4f leakage=%.2e neutron=%.2e\n",
                sp.scatter_frac, sp.sidescatter, sp.leakage_frac, sp.neutron_frac);

    // Per-organ table. We print the EXACT fixed-point dose integer (deterministic)
    // plus the stray-to-target ratio and BEIR-VII lifetime risk, both derived from
    // those exact integers with fixed precision -> byte-identical every run.
    std::printf("organ                 dose_fixed   stray/target   LAR(per1e4)\n");
    double total_lar = 0.0;
    for (int o = 0; o < sp.n_organs; ++o) {
        const unsigned long long df = dose_g[o];
        const double ratio = (target_fixed > 0)
                             ? static_cast<double>(df) / static_cast<double>(target_fixed)
                             : 0.0;
        const double lar = organ_lar(prob.organs[o].risk_coeff, df);
        // Out-of-field organs (index >= field_end) are the secondary-cancer sites.
        if (o >= sp.field_end) total_lar += lar;
        std::printf("%-18s %14llu   %10.3e   %10.4e\n",
                    prob.organs[o].name.c_str(), df, ratio, lar);
    }
    std::printf("total out-of-field secondary-cancer LAR = %.4e per 10^4 persons\n",
                total_lar);
    std::printf("RESULT: %s (GPU dose tally matches CPU exactly)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s\n", path.c_str());
    std::fprintf(stderr, "[timing] CPU MC: %.3f ms   GPU MC: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- speed-up grows with history "
                         "count; real stray-dose plans run 1e9-1e12 histories.\n");
    std::fprintf(stderr, "[verify] organ mismatches = %d (fixed-point dose => atomics commute)\n",
                 mismatches);
    std::fprintf(stderr, "[note]   reduced-scope teaching model; risk coefficients are "
                         "illustrative, NOT clinical (see THEORY.md).\n");

    return pass ? 0 : 1;
}
