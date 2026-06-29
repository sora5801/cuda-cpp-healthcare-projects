// ===========================================================================
// src/main.cu  --  Entry point: load system, run CPU + GPU MD, verify, report
// ---------------------------------------------------------------------------
// Project 1.1 : Molecular Dynamics Engine  (reduced-scope teaching version)
//
// THE 5-STEP SHAPE (every project in this repo follows it)
//   1. Load the problem: an MD system (atoms + parameters) from data/sample, or a
//      deterministic built-in fallback.
//   2. CPU reference: integrate the trajectory serially (reference_cpu.cpp) ->
//      the trusted observables.
//   3. GPU: integrate the SAME trajectory with the tiled all-pairs kernels
//      (kernels.cu).
//   4. VERIFY: assert the GPU observables match the CPU's within a documented
//      physical tolerance (THEORY §numerics explains why it is not bit-exact).
//   5. REPORT: deterministic results to STDOUT (diffed by the demo); timing and
//      run-varying numbers to STDERR (shown, not diffed).
//
//   Code tour: read this, then md.h (the physics), reference_cpu.cpp (the serial
//   driver), kernels.cuh -> kernels.cu (the GPU twin). See ../THEORY.md for "why".
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>

#include "kernels.cuh"        // integrate_gpu
#include "reference_cpu.h"    // load_system, make_default_system, integrate_cpu
#include "util/io.hpp"        // util::CpuTimer

// Project identity (kept in sync with demo/expected_output.txt).
static const char* PROJECT_ID   = "1.1";
static const char* PROJECT_NAME = "Molecular Dynamics Engine";

// Verification tolerance. Velocity-Verlet runs the SAME double-precision math on
// CPU and GPU, but the GPU fuses multiply-adds (FMA) and sums the all-pairs force
// in a different order than the serial CPU loop. Over many steps these ~1e-15
// per-operation differences accumulate, so the trajectories diverge by a tiny,
// physically negligible amount (PATTERNS.md §4: the "long iterative solver" case).
// We therefore verify the energy observables to a small ABSOLUTE tolerance rather
// than pretending the results are bit-identical. THEORY §numerics quantifies this.
static constexpr double ENERGY_TOL   = 1.0e-6;   // on E0, E_final, max_drift
// The position checksum sums n coordinates each O(box); allow it to drift a bit
// more in absolute terms (it is the most chaos-sensitive observable).
static constexpr double CHECKSUM_TOL = 1.0e-4;

int main(int argc, char** argv) {
    // ---- 1. Load the system ------------------------------------------------
    MdSystem sys;
    const char* source = "synthetic (built-in)";
    if (argc > 1) {
        try {
            sys = load_system(argv[1]);
            source = argv[1];
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[error] %s\n", e.what());
            return 2;
        }
    } else {
        sys = make_default_system();
    }
    const SimParams& p = sys.params;

    // ---- 2. CPU reference (timed) ------------------------------------------
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    MdResult cpu = integrate_cpu(sys);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU run (kernel time measured inside) --------------------------
    float gpu_kernel_ms = 0.0f;
    MdResult gpu = integrate_gpu(sys, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    const double d_E0    = std::fabs(cpu.E0        - gpu.E0);
    const double d_Ef    = std::fabs(cpu.E_final   - gpu.E_final);
    const double d_drift = std::fabs(cpu.max_drift - gpu.max_drift);
    const double d_csum  = std::fabs(cpu.pos_checksum - gpu.pos_checksum);
    const bool pass = (d_E0 <= ENERGY_TOL) && (d_Ef <= ENERGY_TOL) &&
                      (d_drift <= ENERGY_TOL) && (d_csum <= CHECKSUM_TOL);

    // ---- 5a. Deterministic report -> STDOUT (diffed) -----------------------
    // We print the GPU observables (verified == CPU) at fixed precision. Energy
    // conservation (E_final ~= E0, small max_drift) is the headline correctness
    // signal of a working MD integrator -- the physics teaching point here.
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("Lennard-Jones fluid, velocity-Verlet (reduced units)\n");
    std::printf("atoms = %d  box = %.3f  dt = %.4f  steps = %d  rcut = %.3f\n",
                p.n, p.box, p.dt, p.steps, p.rcut);
    std::printf("E0          = %.6f\n", gpu.E0);
    std::printf("E_final     = %.6f\n", gpu.E_final);
    std::printf("max |dE|    = %.6e\n", gpu.max_drift);
    std::printf("rel drift   = %.6e\n",
                (gpu.E0 != 0.0) ? gpu.max_drift / std::fabs(gpu.E0) : 0.0);
    std::printf("T_final     = %.6f\n", gpu.T_final);
    std::printf("pos_chksum  = %.6f\n", gpu.pos_checksum);
    std::printf("RESULT: %s (GPU matches CPU: dE<=%.1e, dchksum<=%.1e)\n",
                pass ? "PASS" : "FAIL", ENERGY_TOL, CHECKSUM_TOL);

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%d atoms, %d steps)\n",
                 source, p.n, p.steps);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernels: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- at this tiny N the GPU is "
                         "launch-bound; the all-pairs O(N^2) GPU win grows with N.\n");
    std::fprintf(stderr, "[verify] dE0=%.2e dE_final=%.2e dDrift=%.2e dChksum=%.2e\n",
                 d_E0, d_Ef, d_drift, d_csum);

    return pass ? 0 : 1;
}
