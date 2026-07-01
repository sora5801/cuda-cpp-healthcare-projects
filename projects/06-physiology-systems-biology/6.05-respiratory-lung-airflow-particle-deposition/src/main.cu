// ===========================================================================
// src/main.cu  --  Entry point: load, run CPU + GPU deposition, verify, report
// ---------------------------------------------------------------------------
// Project 6.5 : Respiratory / Lung Airflow & Particle Deposition
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the deposition experiment (data/sample, or a built-in synthetic
//      fallback) and build the airway geometry.
//   2. CPU reference: track all particles serially (reference_cpu.cpp).
//   3. GPU result: track all particles in parallel (kernels.cu) -- IDENTICAL
//      histories via the shared lung_physics.h.
//   4. VERIFY: the two INTEGER per-generation tallies must match EXACTLY (atomic
//      integer adds commute -> deterministic, CPU-matching sum).
//   5. REPORT: a deterministic deposition histogram to stdout; timing to stderr.
//
//   STDOUT is kept byte-for-byte deterministic so demo/run_demo can diff it
//   against demo/expected_output.txt. Anything that varies run-to-run (timings)
//   goes to STDERR, which the demo shows but does not diff.
//
// Code tour: read this first, then lung_physics.h (the shared physics),
// kernels.cuh -> kernels.cu (the GPU twin), and reference_cpu.cpp (the baseline).
// See ../THEORY.md for the science and the GPU mapping.
// ===========================================================================
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // deposition_gpu (GPU path), lung::Airway
#include "reference_cpu.h"    // load_problem, build_airway, deposition_cpu
#include "util/io.hpp"        // util::CpuTimer

// These two tokens identify the program and must match demo/expected_output.txt.
static const char* PROJECT_ID   = "6.5";
static const char* PROJECT_NAME = "Respiratory / Lung Airflow & Particle Deposition";

// ---------------------------------------------------------------------------
// make_synthetic_problem: the built-in experiment used when no data file is
// supplied. A 5-micron aerosol (unit density) inhaled at 30 L/min through the
// first 16 conducting-airway generations, 200000 particle histories, fixed seed.
// These EXACT values are what demo/expected_output.txt encodes, so they must not
// change casually. (data/sample/lung_params.txt holds the same numbers.)
// ---------------------------------------------------------------------------
static DepositionProblem make_synthetic_problem() {
    DepositionProblem p;
    p.d_p         = 5.0e-6;          // 5 micron diameter                     [m]
    p.rho_p       = 1000.0;          // unit density (water-like)        [kg/m^3]
    p.n_gen       = 16;              // conducting airways (trachea..terminal)
    p.flow_rate   = 30.0e-3 / 60.0;  // 30 L/min -> m^3/s
    p.n_particles = 200000;          // histories
    p.seed        = 12345ULL;        // fixed for reproducibility
    return p;
}

int main(int argc, char** argv) {
    // ---- 1. Load the problem + build the airway ----------------------------
    DepositionProblem prob;
    const char* source = "synthetic (built-in)";
    if (argc > 1) {
        try {
            prob = load_problem(argv[1]);   // parse the sample file
            source = argv[1];
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[error] %s\n", e.what());
            return 2;
        }
    } else {
        prob = make_synthetic_problem();
    }
    const lung::Airway aw = build_airway(prob);   // shared geometry (CPU==GPU)

    // ---- 2. CPU reference (timed) ------------------------------------------
    std::vector<uint64_t> tally_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    deposition_cpu(prob, aw, tally_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU result (kernel timed inside the wrapper) -------------------
    std::vector<uint64_t> tally_gpu;
    float gpu_kernel_ms = 0.0f;
    deposition_gpu(prob, aw, tally_gpu, &gpu_kernel_ms);

    // ---- 4. Verify (exact integer match) -----------------------------------
    const int n_slots = prob.n_gen + 1;         // generations + exhaled bucket
    int mismatches = 0;
    uint64_t deposited = 0;                       // particles that hit a wall
    int peak_gen = 0;                             // generation with most deposits
    for (int s = 0; s < n_slots; ++s) {
        if (tally_cpu[s] != tally_gpu[s]) ++mismatches;
        if (s < prob.n_gen) {
            deposited += tally_gpu[s];
            if (tally_gpu[s] > tally_gpu[peak_gen]) peak_gen = s;
        }
    }
    const uint64_t exhaled = tally_gpu[prob.n_gen];
    const bool pass = (mismatches == 0);

    // Deposition fraction in fixed-point permille (integer) so stdout is exactly
    // reproducible -- no float formatting of a run-dependent ratio. permille =
    // round(1000 * deposited / n). Integer rounding: (2*num + den) / (2*den).
    const uint64_t dep_permille =
        (2ULL * deposited * 1000ULL + prob.n_particles) / (2ULL * prob.n_particles);

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("aerosol: d_p = %.1f um, rho_p = %.0f kg/m^3\n",
                prob.d_p * 1e6, prob.rho_p);
    std::printf("airway : %d generations, flow = %.1f L/min\n",
                prob.n_gen, prob.flow_rate * 60.0 * 1000.0);
    std::printf("particles = %llu, seed = %llu\n",
                static_cast<unsigned long long>(prob.n_particles),
                static_cast<unsigned long long>(prob.seed));
    std::printf("deposited = %llu of %llu (%llu.%01llu%%), exhaled = %llu\n",
                static_cast<unsigned long long>(deposited),
                static_cast<unsigned long long>(prob.n_particles),
                static_cast<unsigned long long>(dep_permille / 10ULL),
                static_cast<unsigned long long>(dep_permille % 10ULL),
                static_cast<unsigned long long>(exhaled));
    std::printf("peak deposition generation = %d\n", peak_gen);
    std::printf("deposition per generation (counts):\n ");
    for (int g = 0; g < prob.n_gen; ++g)
        std::printf(" %llu", static_cast<unsigned long long>(tally_gpu[g]));
    std::printf("\n");
    std::printf("RESULT: %s (GPU deposition tally matches CPU exactly)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s\n", source);
    std::fprintf(stderr, "[timing] CPU tracking: %.3f ms   GPU tracking: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- speed-up grows with particle "
                         "count; whole-lung studies track 1e7-1e8 particles.\n");
    std::fprintf(stderr, "[verify] generation mismatches = %d (integer tally => atomics commute)\n",
                 mismatches);

    // Exit code feeds the demo's pass/fail gate.
    return pass ? 0 : 1;
}
