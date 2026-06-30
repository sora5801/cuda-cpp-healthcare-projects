// ===========================================================================
// src/main.cu  --  Entry point: run HPS MD on CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 2.30 : Protein Solubility & Phase Separation Simulation
//
// 5-step shape (the shape EVERY project in this repo follows):
//   1. Load the system (data/sample, or a CLI-supplied file).
//   2. CPU reference simulation (reference_cpu.cpp) -> trusted final summary.
//   3. GPU simulation (kernels.cu) -- SAME shared physics, run in parallel.
//   4. VERIFY: the CPU and GPU final-state summaries agree within tolerance.
//   5. REPORT: deterministic summary to stdout; timing to stderr.
//
//   STDOUT is kept byte-for-byte deterministic (fixed-precision printing) so
//   demo/run_demo can diff it against demo/expected_output.txt. Run-to-run
//   varying numbers (wall-clock timings) go to STDERR, which the demo shows but
//   does not diff (docs/PATTERNS.md §3).
//
// Code tour: start here, then hps_model.h (the shared physics), reference_cpu.cpp
// (the serial baseline), kernels.cu (the GPU twin). The science / GPU-mapping is
// in ../THEORY.md.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>

#include "kernels.cuh"        // run_gpu
#include "reference_cpu.h"    // load_system, run_cpu, System, SimSummary
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "2.30";
static const char* PROJECT_NAME = "Protein Solubility & Phase Separation Simulation";

// Verification tolerances. We use FP64 throughout and the GPU/CPU run the
// IDENTICAL fixed-order arithmetic (shared bead_force()), so on this stable
// sample they agree to ~1e-15 -- essentially machine precision. We still verify
// against a small NON-ZERO tolerance rather than demand bit-identity, because
// the GPU's fused multiply-add (FMA) contracts a*b+c differently from the host
// compiler; on a longer or more chaotic run those last-bit differences would
// grow (MD is chaotic -- a real lesson, docs/PATTERNS.md §4). 1e-6 is far below
// any physically meaningful energy here yet comfortably above the observed drift.
//   * Energies / checksum: small absolute tolerance.
//   * Integer order parameters (n_condensed, max density): must match EXACTLY.
static constexpr double ENERGY_ATOL   = 1.0e-6; // reduced energy units
static constexpr double CHECKSUM_ATOL = 1.0e-6; // sum over N beads of coordinates

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path =
        (argc > 1) ? argv[1] : "data/sample/system.txt";
    System sys;
    try {
        sys = load_system(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    SimSummary cpu{};
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    run_cpu(sys, cpu);                 // runs on a COPY; sys stays the initial state
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU simulation (kernel time measured inside) ------------------
    SimSummary gpu{};
    float gpu_kernel_ms = 0.0f;
    run_gpu(sys, gpu, &gpu_kernel_ms); // same initial `sys`, independent device copy

    // ---- 4. Verify ---------------------------------------------------------
    const double d_pe  = std::fabs(cpu.potential - gpu.potential);
    const double d_ke  = std::fabs(cpu.kinetic   - gpu.kinetic);
    const double d_chk = std::fabs(cpu.pos_checksum - gpu.pos_checksum);
    const bool pass =
        (d_pe  <= ENERGY_ATOL) &&
        (d_ke  <= ENERGY_ATOL) &&
        (d_chk <= CHECKSUM_ATOL) &&
        (cpu.n_condensed == gpu.n_condensed) &&
        (cpu.max_local_density == gpu.max_local_density);

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    // Fixed precision so the bytes are reproducible. We print the GPU result
    // (verified equal to the CPU) as the canonical answer.
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("HPS coarse-grained LLPS model (synthetic, reduced units)\n");
    std::printf("beads=%d  chains=%d  chain_len=%d  box=%.3f  steps=%d\n",
                sys.p.n_beads, sys.p.n_chains, sys.p.chain_len, sys.p.box, sys.p.n_steps);
    std::printf("final potential energy = %.6f\n", gpu.potential);
    std::printf("final kinetic   energy = %.6f\n", gpu.kinetic);
    std::printf("position checksum      = %.6f\n", gpu.pos_checksum);
    std::printf("phase order parameters:\n");
    std::printf("  max  local density (neighbours within r_cut) = %.0f\n", gpu.max_local_density);
    std::printf("  mean local density                           = %.4f\n", gpu.mean_local_density);
    std::printf("  condensed beads (>=4 neighbours)             = %d of %d\n",
                gpu.n_condensed, sys.p.n_beads);
    std::printf("RESULT: %s (GPU matches CPU within tolerance)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s\n", path.c_str());
    std::fprintf(stderr, "[timing] CPU MD: %.3f ms   GPU MD: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- tiny N over many tiny "
                         "kernel launches is launch-bound; the GPU's O(N^2) force "
                         "edge grows with bead count.\n");
    std::fprintf(stderr, "[verify] |dPE|=%.3e |dKE|=%.3e |dchecksum|=%.3e "
                         "(energy tol %.1e, checksum tol %.1e)\n",
                 d_pe, d_ke, d_chk, ENERGY_ATOL, CHECKSUM_ATOL);
    std::fprintf(stderr, "[verify] n_condensed: cpu=%d gpu=%d ; max_density: cpu=%.0f gpu=%.0f\n",
                 cpu.n_condensed, gpu.n_condensed, cpu.max_local_density, gpu.max_local_density);

    // Exit code feeds the demo's pass/fail gate.
    return pass ? 0 : 1;
}
