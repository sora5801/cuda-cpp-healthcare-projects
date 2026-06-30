// ===========================================================================
// src/main.cu  --  Entry point: build system, run CPU + GPU MD, verify, report
// ---------------------------------------------------------------------------
// Project 2.19 : Membrane Protein Simulation   (reduced-scope teaching version)
//
// WHAT THIS FILE DOES  (the 5-step shape EVERY project in this repo follows)
//   1. Load SimParams from data/sample (or fail loudly).
//   2. Build the SAME initial bilayer+protein system twice (CPU copy, GPU copy).
//   3. Run the CPU reference MD (reference_cpu.cpp) -> trusted trajectory.
//      Run the GPU MD (kernels.cu)                  -> the thing taught.
//   4. VERIFY: the final positions/velocities agree within tolerance, AND the
//      physics observables (bilayer thickness, potential energy) agree.
//   5. REPORT: deterministic observables + a few sampled bead positions to
//      stdout; timings and run-varying detail to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt; timings (which vary) go to STDERR.
//
//   NOT FOR CLINICAL USE. This is a tiny COARSE-GRAINED teaching model, not a
//   validated membrane simulation. See README "Limitations" and THEORY.
//
// READ THIS FIRST in the code tour, then membrane.h (the physics), kernels.cuh
// -> kernels.cu (GPU), reference_cpu.cpp (CPU baseline). See ../THEORY.md.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // simulate_gpu, SimParams, System, Vec3
#include "reference_cpu.h"    // load_params, build_system, simulate_cpu, observables
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "2.19";
static const char* PROJECT_NAME = "Membrane Protein Simulation (coarse-grained, reduced scope)";

// Verification tolerance on final bead positions/velocities. Both paths run the
// IDENTICAL double-precision math (membrane.h) in the SAME order, but the GPU's
// fused multiply-add (FMA) contracts a*b+c differently from the host compiler,
// so over hundreds of MD steps the two trajectories drift at the ~1e-6 level
// even in double precision. We therefore verify to a physically-negligible
// 1e-4 (PATTERNS.md section 4: long iterative solver), NOT to round-off, and we
// say so honestly. (See THEORY "Numerical considerations".)
static constexpr double TOLERANCE = 1.0e-4;

int main(int argc, char** argv) {
    // ---- 1. Load parameters ------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/membrane_sample.txt";
    SimParams P;
    try {
        P = load_params(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. Build the same initial system twice ----------------------------
    // build_system is deterministic, so cpu_sys and gpu_sys start identical.
    System cpu_sys, gpu_sys;
    build_system(P, cpu_sys);
    build_system(P, gpu_sys);

    // Record the INITIAL observables so the report shows the equilibration move.
    const double thick0 = bilayer_thickness(P, cpu_sys);
    const double energy0 = total_potential_energy(P, cpu_sys);

    // ---- 3a. CPU reference (timed) -----------------------------------------
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    simulate_cpu(P, cpu_sys);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3b. GPU simulation (loop timed via CUDA events) -------------------
    float gpu_kernel_ms = 0.0f;
    simulate_gpu(P, gpu_sys, &gpu_kernel_ms);

    // ---- 4. Verify: final state agrees -------------------------------------
    double worst_pos = 0.0, worst_vel = 0.0;
    for (int i = 0; i < P.n_beads; ++i) {
        worst_pos = std::fmax(worst_pos, length(cpu_sys.pos[i] - gpu_sys.pos[i]));
        worst_vel = std::fmax(worst_vel, length(cpu_sys.vel[i] - gpu_sys.vel[i]));
    }
    const bool pass = (worst_pos <= TOLERANCE) && (worst_vel <= TOLERANCE);

    // Physics observables from the (trusted) CPU final state for the report.
    const double thick_cpu = bilayer_thickness(P, cpu_sys);
    const double thick_gpu = bilayer_thickness(P, gpu_sys);
    const double energy_cpu = total_potential_energy(P, cpu_sys);
    const double energy_gpu = total_potential_energy(P, gpu_sys);

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("system: %d lipids (3 beads each) + %d protein beads = %d beads, %d steps\n",
                P.n_lipids, P.n_prot, P.n_beads, P.steps);
    std::printf("box: %.2f x %.2f (x,y periodic; z free slab)   dt=%.4f  kT=%.3f  gamma=%.3f\n",
                P.box_x, P.box_y, P.dt, P.temperature, P.gamma);
    std::printf("initial : bilayer_thickness = %.6f   potential_energy = %.6f\n",
                thick0, energy0);
    std::printf("final   : bilayer_thickness = %.6f   potential_energy = %.6f\n",
                thick_cpu, energy_cpu);
    // A few deterministic sampled bead positions (first head, a tail, a protein).
    // We sample fixed indices so the printed lines are reproducible.
    const int i_head = 0;                          // first lipid's head bead
    const int i_tail = 2;                          // first lipid's lower tail bead
    const int i_prot = 3 * P.n_lipids;             // first protein bead (if any)
    auto pr = [&](const char* label, int idx) {
        if (idx < 0 || idx >= P.n_beads) return;
        const Vec3 p = cpu_sys.pos[idx];
        std::printf("%s bead[%d] pos = (%.6f, %.6f, %.6f)\n", label, idx, p.x, p.y, p.z);
    };
    pr("head   ", i_head);
    pr("tail   ", i_tail);
    if (P.n_prot > 0) pr("protein", i_prot);
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-04)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%d beads, %d bonds)\n",
                 path.c_str(), P.n_beads, static_cast<int>(cpu_sys.bond_i.size()));
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny O(N^2) all-pairs system "
                         "is launch-bound; the GPU's edge grows with bead count.\n");
    std::fprintf(stderr, "[verify] worst |dpos| = %.3e   worst |dvel| = %.3e   (tol %.1e)\n",
                 worst_pos, worst_vel, TOLERANCE);
    std::fprintf(stderr, "[verify] thickness  CPU=%.6f  GPU=%.6f  (diff %.3e)\n",
                 thick_cpu, thick_gpu, std::fabs(thick_cpu - thick_gpu));
    std::fprintf(stderr, "[verify] energy     CPU=%.6f  GPU=%.6f  (diff %.3e)\n",
                 energy_cpu, energy_gpu, std::fabs(energy_cpu - energy_gpu));

    return pass ? 0 : 1;
}
