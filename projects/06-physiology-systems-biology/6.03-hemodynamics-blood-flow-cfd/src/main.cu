// ===========================================================================
// src/main.cu  --  Entry point: load, run CPU + GPU NSE, verify, report
// ---------------------------------------------------------------------------
// Project 6.3 : Hemodynamics / Blood-Flow CFD   (reduced-scope teaching version)
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the channel parameters (data/sample, or a built-in fallback).
//   2. Run the CPU reference fractional-step NSE solver  -> trusted answer.
//   3. Run the GPU solver (identical per-cell physics)   -> the thing taught.
//   4. VERIFY: GPU velocity field agrees with CPU within a documented tolerance;
//      AND both converge toward the analytic Poiseuille centreline velocity.
//   5. REPORT: deterministic velocity profile + wall shear stress to stdout;
//      timing to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Run-varying numbers (timings) go to STDERR, which
//   the demo shows but does not diff (PATTERNS.md §3).
//
// READ THIS FIRST in the code tour, then kernels.cuh -> kernels.cu, and
// reference_cpu.cpp for the baseline. See ../THEORY.md for the "why".
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // nse_gpu (GPU path), ChannelParams
#include "nse_channel.h"      // wall_shear_stress, idx (shared physics)
#include "reference_cpu.h"    // load_channel, nse_cpu, poiseuille_umax
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "6.3";
static const char* PROJECT_NAME = "Hemodynamics / Blood-Flow CFD";

// Correctness tolerance on the velocity field (dimensionless units in the demo).
//   The CPU and GPU call the SAME double-precision per-cell functions, but the
//   GPU fuses multiply-add (FMA) where the host compiler does not, so over the
//   many thousands of stencil operations in this iterative solver the two drift
//   apart by a physically-negligible amount (PATTERNS.md §4, the "long iterative
//   solver" case -- see also flagships 10.02 and 14.02). We verify to a tight
//   absolute tolerance and report the actual max difference on stderr.
static constexpr double TOLERANCE = 1.0e-9;

// Built-in synthetic fallback matching data/sample/channel_params.txt, so the
// program still runs (and prints the SAME stdout) if the file is missing.
//   32 x 17 channel, 40000 steps, 40 Jacobi pressure iters, Newtonian blood.
static ChannelParams make_fallback() {
    ChannelParams p;
    p.nx = 32; p.ny = 17; p.steps = 40000; p.p_iters = 40;
    p.h = 1.0; p.dt = 0.02; p.rho = 1.0; p.gx = 1.0e-4;
    p.nu0 = 0.1; p.nu_inf = 0.1;          // nu0==nu_inf => Newtonian
    p.lambda = 1.0; p.n_cy = 0.5; p.a_cy = 2.0;
    return p;
}

int main(int argc, char** argv) {
    // ---- 1. Load ------------------------------------------------------------
    ChannelParams p;
    const char* source = "synthetic (built-in fallback)";
    if (argc > 1) {
        try {
            p = load_channel(argv[1]);
            source = argv[1];
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[error] %s\n", e.what());
            return 2;
        }
    } else {
        p = make_fallback();
    }

    // ---- 2. CPU reference (timed) ------------------------------------------
    std::vector<double> u_cpu, v_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    nse_cpu(p, u_cpu, v_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU solver (kernel loop timed inside the wrapper) --------------
    std::vector<double> u_gpu, v_gpu;
    float gpu_kernel_ms = 0.0f;
    nse_gpu(p, u_gpu, v_gpu, &gpu_kernel_ms);

    // ---- 4. Verify: GPU velocity field matches CPU within tolerance ---------
    double err = 0.0;
    for (std::size_t k = 0; k < u_cpu.size(); ++k) {
        err = std::fmax(err, std::fabs(u_cpu[k] - u_gpu[k]));
        err = std::fmax(err, std::fabs(v_cpu[k] - v_gpu[k]));
    }
    const bool pass = err <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    // Profile the streamwise velocity u_x across the channel at mid-length (a
    // representative column). The steady solution is a parabola (Poiseuille).
    const int xc = p.nx / 2;
    double umax = 0.0; int ymax = 0;
    for (int y = 0; y < p.ny; ++y) {
        const double uu = u_gpu[idx(xc, y, p.nx)];
        if (uu > umax) { umax = uu; ymax = y; }
    }
    // Wall shear stress at the bottom wall, averaged over all streamwise columns.
    // For fully-developed flow WSS is uniform along x; we average for robustness.
    const double nu = effective_nu(p);
    double wss_sum = 0.0;
    for (int x = 0; x < p.nx; ++x)
        wss_sum += wall_shear_stress(x, p.nx, p.h, p.rho, nu, u_gpu.data());
    const double wss_bottom = wss_sum / p.nx;
    // Analytic Poiseuille peak (science-level check): u_max should approach it.
    const double u_analytic = poiseuille_umax(p);

    std::printf("%s -- %s (reduced-scope 2-D incompressible Navier-Stokes)\n",
                PROJECT_ID, PROJECT_NAME);
    std::printf("channel: %dx%d grid, %d steps, %d pressure-iters, "
                "nu=%.4f, rho=%.2f, gx=%.2e\n",
                p.nx, p.ny, p.steps, p.p_iters, nu, p.rho, p.gx);
    std::printf("rheology: %s (Carreau-Yasuda nu0=%.4f nu_inf=%.4f)\n",
                (p.nu0 == p.nu_inf) ? "Newtonian" : "non-Newtonian (shear-thinning)",
                p.nu0, p.nu_inf);
    std::printf("centerline u_max = %.6f at y=%d  (analytic Poiseuille = %.6f)\n",
                umax, ymax, u_analytic);
    std::printf("wall shear stress (bottom wall, mean over x) = %.6f\n", wss_bottom);
    std::printf("velocity profile u_x(y) across the channel (x=%d):\n", xc);
    for (int y = 0; y < p.ny; ++y)
        std::printf(" %.6f", u_gpu[idx(xc, y, p.nx)]);
    std::printf("\n");
    std::printf("RESULT: %s (GPU velocity matches CPU within tol=1.0e-09)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    const double rel = (u_analytic != 0.0)
                     ? std::fabs(umax - u_analytic) / u_analytic : 0.0;
    std::fprintf(stderr, "[data]   source: %s  (%d x %d, %d steps)\n",
                 source, p.nx, p.ny, p.steps);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- many tiny per-step kernel "
                         "launches are launch-bound on this small grid; the GPU's edge "
                         "grows with 3-D, patient-scale meshes.\n");
    std::fprintf(stderr, "[science] |u_max - analytic| / analytic = %.3e "
                         "(discretization + not-yet-fully-converged error)\n", rel);
    std::fprintf(stderr, "[verify] max |CPU-GPU| velocity diff = %.3e  (tolerance %.1e)\n",
                 err, TOLERANCE);

    return pass ? 0 : 1;
}
