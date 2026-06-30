// ===========================================================================
// src/main.cu  --  Entry point: load data, run CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 2.29 : Ion Channel Gating & Permeation Simulation
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the permeation problem (data/sample, or a built-in fallback).
//   2. Compute the CPU reference (reference_cpu.cpp)         -> trusted answer.
//   3. Compute the GPU result    (kernels.cu)                -> the thing taught.
//   4. VERIFY: assert GPU == CPU EXACTLY (integer tallies)   -> correctness.
//   5. REPORT: deterministic result to stdout; timing to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Anything that varies run-to-run (timings) goes to
//   STDERR, which the demo shows but does not diff (PATTERNS.md §3).
//
// READ THIS FIRST in the code tour, then channel_physics.h (the shared physics),
// kernels.cuh -> kernels.cu, and reference_cpu.cpp for the baseline. See
// ../THEORY.md for the "why".
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // permeation_gpu (GPU path), PermeationProblem/Result
#include "reference_cpu.h"    // load_permeation_problem, permeation_cpu
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "2.29";
static const char* PROJECT_NAME = "Ion Channel Gating & Permeation Simulation";

// ---------------------------------------------------------------------------
// find_peak_bin: index of the most-occupied z-bin. With a forward voltage and a
// central PMF barrier, ions pile up on the UPHILL (intracellular) side of the
// barrier -- a physically meaningful, deterministic feature we report and that
// cross-checks the science (THEORY.md "How we verify correctness").
// ---------------------------------------------------------------------------
static int find_peak_bin(const std::vector<unsigned long long>& occ) {
    int peak = 0;
    for (int b = 1; b < static_cast<int>(occ.size()); ++b)
        if (occ[b] > occ[peak]) peak = b;
    return peak;
}

int main(int argc, char** argv) {
    // ---- 1. Load the problem -----------------------------------------------
    const std::string path =
        (argc > 1) ? argv[1] : "data/sample/channel_params.txt";
    PermeationProblem prob;
    try {
        prob = load_permeation_problem(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) ------------------------------------------
    PermeationResult res_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    permeation_cpu(prob, res_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU result (kernel timed inside the wrapper) -------------------
    PermeationResult res_gpu;
    float gpu_kernel_ms = 0.0f;
    permeation_gpu(prob, res_gpu, &gpu_kernel_ms);

    // ---- 4. Verify (EXACT integer match -- atomics commute on ints) --------
    int occ_mismatches = 0;
    unsigned long long total_occ = 0;
    for (int b = 0; b < prob.cp.n_bins; ++b) {
        if (res_cpu.occupancy[b] != res_gpu.occupancy[b]) ++occ_mismatches;
        total_occ += res_gpu.occupancy[b];
    }
    const bool counts_match =
        (res_cpu.fwd == res_gpu.fwd) && (res_cpu.rev == res_gpu.rev);
    const bool pass = (occ_mismatches == 0) && counts_match;

    // Derived, reported physics: net forward flux per ion-step (the conductance
    // proxy) and the peak-occupancy bin. Both come from the integer tallies, so
    // they are identical on CPU and GPU.
    const double net_flux = net_flux_per_ion_step(res_gpu, prob);
    const int peak_bin = find_peak_bin(res_gpu.occupancy);

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("pore L=%.2f nm, %d bins, barrier U=%.2f kT (sigma=%.2f nm), "
                "q=%+.0f, V=%.2f kT/e\n",
                prob.cp.L, prob.cp.n_bins, prob.cp.U_barrier, prob.cp.sigma,
                prob.cp.q, prob.cp.V);
    std::printf("BD: D=%.3f nm^2/step, dt=%.3f, steps=%d, ions=%llu\n",
                prob.cp.D, prob.cp.dt, prob.cp.n_steps, prob.n_ions);
    std::printf("permeations: forward=%llu  reverse=%llu  net=%lld\n",
                res_gpu.fwd, res_gpu.rev,
                static_cast<long long>(res_gpu.fwd) -
                static_cast<long long>(res_gpu.rev));
    // net flux printed at fixed precision so stdout is byte-stable across runs.
    std::printf("net flux per ion-step = %.6e (conductance proxy)\n", net_flux);
    std::printf("occupancy histogram (ion-steps per z-bin):\n ");
    for (int b = 0; b < prob.cp.n_bins; ++b)
        std::printf(" %llu", res_gpu.occupancy[b]);
    std::printf("\n");
    std::printf("peak occupancy at bin %d (z ~ %.2f nm)\n",
                peak_bin, (peak_bin + 0.5) * prob.cp.L / prob.cp.n_bins);
    std::printf("RESULT: %s (GPU tallies match CPU exactly)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s\n", path.c_str());
    std::fprintf(stderr, "[timing] CPU BD: %.3f ms   GPU BD: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- the GPU's edge grows with "
                         "ion count; real channel studies run 1e5-1e7 ions x 1e6+ steps.\n");
    std::fprintf(stderr, "[verify] occupancy bin mismatches = %d, fwd/rev match = %s "
                         "(integer tallies => atomics commute)\n",
                 occ_mismatches, counts_match ? "yes" : "NO");
    std::fprintf(stderr, "[verify] total ion-steps tallied = %llu (== ions*steps = %llu)\n",
                 total_occ, prob.n_ions * static_cast<unsigned long long>(prob.cp.n_steps));

    return pass ? 0 : 1;
}
