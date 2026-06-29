// ===========================================================================
// src/main.cu  --  Entry point: run LBM on CPU + GPU, verify, report profile
// ---------------------------------------------------------------------------
// Project 6.04 : Lattice-Boltzmann Blood/Airflow Solver
//
// 5-step shape:
//   1. Load the channel parameters (data/sample).
//   2. CPU reference LBM (reference_cpu.cpp).
//   3. GPU LBM (kernels.cu) -- identical per-node physics (lbm_d2q9.h).
//   4. VERIFY: the GPU velocity field matches the CPU field within tolerance.
//   5. REPORT: deterministic across-channel velocity profile to stdout.
//
// Physically the driven channel develops toward a parabolic (Poiseuille)
// velocity profile -- maximum at the centreline, ~0 at the no-slip walls.
//
// Code tour: start here, then lbm_d2q9.h (the per-node update), kernels.cu,
// reference_cpu.cpp. The physics/GPU-mapping is in ../THEORY.md.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // lbm_gpu, LbmParams
#include "reference_cpu.h"    // load_lbm, lbm_cpu, velocity_field
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "6.4";
static const char* PROJECT_NAME = "Lattice-Boltzmann Blood/Airflow Solver";

// CPU and GPU differ only by float rounding / FMA over many steps; LBM is
// diffusive (errors damp), so a small absolute tolerance on velocity holds.
static constexpr double TOLERANCE = 1.0e-6;

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/channel_params.txt";
    LbmParams p;
    try {
        p = load_lbm(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<double> f_cpu, ux_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    lbm_cpu(p, f_cpu);
    const double cpu_ms = cpu_timer.stop_ms();
    velocity_field(p, f_cpu, ux_cpu);

    // ---- 3. GPU LBM (loop timed) ------------------------------------------
    std::vector<double> f_gpu, ux_gpu;
    float gpu_kernel_ms = 0.0f;
    lbm_gpu(p, f_gpu, &gpu_kernel_ms);
    velocity_field(p, f_gpu, ux_gpu);

    // ---- 4. Verify (velocity fields agree) --------------------------------
    double err = 0.0;
    for (std::size_t k = 0; k < ux_cpu.size(); ++k) {
        const double d = std::fabs(ux_cpu[k] - ux_gpu[k]);
        if (d > err) err = d;
    }
    const bool pass = err <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    const double nu = (p.tau - 0.5) / 3.0;          // kinematic viscosity
    const int xc = p.nx / 2;                         // a column to profile
    double umax = 0.0; int ymax = 0;
    for (int y = 0; y < p.ny; ++y) {
        const double u = ux_gpu[static_cast<std::size_t>(y) * p.nx + xc];
        if (u > umax) { umax = u; ymax = y; }
    }
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("D2Q9 channel: %dx%d lattice, %d steps, tau=%.3f (nu=%.4f), gx=%.2e\n",
                p.nx, p.ny, p.steps, p.tau, nu, p.gx);
    std::printf("centerline u_max = %.6f at y=%d\n", umax, ymax);
    std::printf("velocity profile u_x(y) across the channel (x=%d):\n", xc);
    for (int y = 0; y < p.ny; ++y)
        std::printf(" %.6f", ux_gpu[static_cast<std::size_t>(y) * p.nx + xc]);
    std::printf("\n");
    std::printf("RESULT: %s (GPU velocity matches CPU within tol=1.0e-06)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d x %d, %d steps)\n",
                 path.c_str(), p.nx, p.ny, p.steps);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- LBM throughput (lattice updates/s) scales with "
                         "grid size; production runs are 3-D and far larger.\n");
    std::fprintf(stderr, "[verify] max velocity diff = %.3e  (tolerance %.1e)\n", err, TOLERANCE);

    return pass ? 0 : 1;
}
