// ===========================================================================
// src/main.cu  --  Entry point: load system, solve on CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 2.27 : Polarizable Water Model GPU Dynamics
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the polarizable-water cluster (data/sample, or a built-in fallback).
//   2. Solve the induced dipoles on the CPU (reference_cpu.cpp) -> trusted answer.
//   3. Solve them on the GPU (kernels.cu)                        -> the thing taught.
//   4. VERIFY: GPU dipoles + energy agree with CPU within tolerance.
//   5. REPORT: deterministic result to STDOUT; timing to STDERR.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Run-to-run varying numbers (timings) go to STDERR.
//
//   The committed sample is engineered to TEACH (data/README.md): the first
//   site is an isolated polarizable site in a known uniform external field, so
//   its converged dipole must equal the ANALYTIC mu = alpha*E -- a physics check
//   on top of the CPU==GPU check (PATTERNS.md §4, §6). The remaining sites form a
//   small water-like cluster whose mutual polarization needs the SCF loop.
//
// READ THIS FIRST in the code tour, then polar.h (the physics), kernels.cuh ->
// kernels.cu (GPU), and reference_cpu.cpp (baseline). See ../THEORY.md for "why".
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // solve_dipoles_gpu (GPU path)
#include "reference_cpu.h"    // PolarSystem, load_system, solve_dipoles_cpu
#include "util/io.hpp"        // util::CpuTimer

// Identity tokens (kept in sync with demo/expected_output.txt).
static const char* PROJECT_ID   = "2.27";
static const char* PROJECT_NAME = "Polarizable Water Model GPU Dynamics";

// Verification tolerance. The CPU and GPU run the SAME double-precision Jacobi
// arithmetic (polar.h) and reduce in fixed point, so they agree to ~1e-11. We
// verify to 1e-9 -- generous enough to absorb the last fixed-point digit, tight
// enough to catch any real divergence. (THEORY.md §Numerics explains the choice.)
static constexpr double TOLERANCE = 1.0e-9;

// ---------------------------------------------------------------------------
// build_synthetic_system: a tiny, fully-deterministic fallback used when no data
//   file is supplied. It is the SAME content as data/sample/water_cluster.txt so
//   the demo's expected output holds with or without the file:
//     * site 0 : an isolated polarizable site (q=0) in a uniform field Eext, far
//                from the cluster -> its dipole tests the analytic mu = alpha*E.
//     * sites 1.. : two water-like molecules (O carries the polarizable charge,
//                two H carry positive fixed charges) close enough to mutually
//                polarize, so the SCF loop actually does work.
//   Coordinates in Angstrom, charges in e, alpha in A^3.
// ---------------------------------------------------------------------------
static PolarSystem build_synthetic_system() {
    PolarSystem s;
    s.a_thole   = 0.39;       // AMOEBA-like Thole screening
    s.max_iters = 200;
    s.tol       = 1.0e-9;
    s.Eext      = Vec3{0.0, 0.0, 0.05};   // uniform external field along +z (e/A^2)

    auto add = [&](double x, double y, double z, double q, double alpha) {
        s.sites.push_back(Site{Vec3{x, y, z}, q, alpha});
    };
    // Isolated probe far away (50 A) so the cluster's field at it is negligible.
    add(50.0, 0.0, 0.0,  0.0, 1.444);           // site 0: polarizable probe
    // Water A (TIP4P-ish geometry), O is the polarizable carrier of -0.834 e.
    add( 0.000, 0.000, 0.000, -0.834, 1.444);   // O  (polarizable)
    add( 0.757, 0.586, 0.000,  0.417, 0.0);     // H  (fixed)
    add(-0.757, 0.586, 0.000,  0.417, 0.0);     // H  (fixed)
    // Water B, displaced ~2.9 A along x (a hydrogen-bond-like separation).
    add( 2.900, 0.000, 0.000, -0.834, 1.444);   // O  (polarizable)
    add( 3.657, 0.586, 0.000,  0.417, 0.0);     // H  (fixed)
    add( 3.657,-0.586, 0.000,  0.417, 0.0);     // H  (fixed)
    return s;
}

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    PolarSystem sys;
    const char* source = "synthetic (built-in)";
    bool loaded = false;
    if (argc > 1) {
        try {
            sys = load_system(argv[1]);
            source = argv[1];
            loaded = true;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[warn] could not load '%s' (%s); using built-in synthetic system\n",
                         argv[1], e.what());
        }
    }
    if (!loaded) sys = build_synthetic_system();
    const int N = num_sites(sys);

    // ---- 2. CPU reference (timed) ------------------------------------------
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    const SolveResult cpu = solve_dipoles_cpu(sys);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU solve (kernels timed inside the wrapper) -------------------
    float gpu_kernel_ms = 0.0f;
    const SolveResult gpu = solve_dipoles_gpu(sys, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    // (a) Largest per-component dipole difference across all sites.
    double dip_err = 0.0;
    for (int i = 0; i < N; ++i) {
        dip_err = std::fmax(dip_err, std::fabs(cpu.mu[i].x - gpu.mu[i].x));
        dip_err = std::fmax(dip_err, std::fabs(cpu.mu[i].y - gpu.mu[i].y));
        dip_err = std::fmax(dip_err, std::fabs(cpu.mu[i].z - gpu.mu[i].z));
    }
    // (b) Polarization-energy difference (internal units).
    const double e_err = std::fabs(cpu.U_pol - gpu.U_pol);
    const bool pass = (dip_err <= TOLERANCE) && (e_err <= TOLERANCE);

    // (c) PHYSICS cross-check: site 0 is an isolated probe in the uniform field
    //     Eext, so its converged dipole must equal the analytic mu = alpha * Eext
    //     (cluster ~50 A away contributes < 1e-6). magnitude of |mu0| vs |alpha*E|.
    const double alpha0 = sys.sites[0].alpha;
    const double mu0_mag = std::sqrt(gpu.mu[0].x * gpu.mu[0].x +
                                     gpu.mu[0].y * gpu.mu[0].y +
                                     gpu.mu[0].z * gpu.mu[0].z);
    const double eext_mag = std::sqrt(sys.Eext.x * sys.Eext.x +
                                      sys.Eext.y * sys.Eext.y +
                                      sys.Eext.z * sys.Eext.z);
    const double mu0_analytic = alpha0 * eext_mag;

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("self-consistent induced dipoles (Jacobi SCF) on %d sites\n", N);
    std::printf("a_thole = %.3f  tol = %.1e  max_iters = %d\n",
                sys.a_thole, sys.tol, sys.max_iters);
    std::printf("converged in %d sweeps\n", gpu.iters);
    std::printf("induced dipole magnitude |mu| per site (e*A):\n");
    for (int i = 0; i < N; ++i) {
        const double m = std::sqrt(gpu.mu[i].x * gpu.mu[i].x +
                                   gpu.mu[i].y * gpu.mu[i].y +
                                   gpu.mu[i].z * gpu.mu[i].z);
        std::printf("  site %2d: q=%+.3f alpha=%.3f  |mu|=%.9f\n",
                    i, sys.sites[i].q, sys.sites[i].alpha, m);
    }
    std::printf("polarization energy U_pol = %.9f e^2/A = %.6f kcal/mol\n",
                gpu.U_pol, gpu.U_pol_kcal);
    std::printf("probe check: |mu0| = %.9f  analytic alpha*Eext = %.9f\n",
                mu0_mag, mu0_analytic);
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-09)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d sites)\n", source, N);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny cluster is launch-bound; "
                         "the GPU's edge grows with site count (real boxes have 10^3-10^6 sites).\n");
    std::fprintf(stderr, "[verify] worst dipole diff = %.3e   energy diff = %.3e   (tol %.1e)\n",
                 dip_err, e_err, TOLERANCE);
    std::fprintf(stderr, "[verify] probe |mu0| error vs analytic = %.3e e*A\n",
                 std::fabs(mu0_mag - mu0_analytic));
    std::fprintf(stderr, "[solver] CPU sweeps=%d (residual %.3e)   GPU sweeps=%d (residual %.3e)\n",
                 cpu.iters, cpu.final_dmu, gpu.iters, gpu.final_dmu);

    return pass ? 0 : 1;
}
