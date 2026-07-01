// ===========================================================================
// src/main.cu  --  Entry point: solve the slab on CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 5.6 : GPU Boltzmann Transport (Deterministic Dose)
//
// 5-step shape (the shape every project in this repo follows):
//   1. Load the slab problem (data/sample).
//   2. CPU reference source iteration (reference_cpu.cpp) -> trusted flux.
//   3. GPU source iteration (kernels.cu)  -- identical per-cell physics
//      (boltzmann_sn.h) -> the thing being taught.
//   4. VERIFY: the GPU scalar flux matches the CPU flux within tolerance, AND
//      an ANALYTIC check on a known homogeneous sub-problem (see below).
//   5. REPORT: a deterministic flux/dose profile to stdout; timing to stderr.
//
// THE SCIENCE IN ONE LINE
//   We deterministically transport particles through a layered 1-D slab and
//   report the scalar flux phi(x) and an absorbed-dose proxy across it -- the
//   same quantity a deterministic clinical dose engine (Acuros XB) produces,
//   here on the smallest problem that still exercises discrete ordinates +
//   source iteration + the transport sweep.
//
// Code tour: start here, then boltzmann_sn.h (the per-cell update), then
//   reference_cpu.cpp (CPU sweep) and kernels.cu (GPU sweep). Physics and the
//   GPU mapping are in ../THEORY.md.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"       // solve_sn_gpu
#include "reference_cpu.h"   // SlabProblem, SnQuadrature, load_slab, solve_sn_cpu, ...
#include "util/io.hpp"       // util::CpuTimer

static const char* PROJECT_ID   = "5.6";
static const char* PROJECT_NAME = "GPU Boltzmann Transport (Deterministic Dose)";

// CPU and GPU run byte-identical per-cell math (boltzmann_sn.h) and sum ordinate
// contributions in the SAME fixed order, so they differ only by the compiler's
// freedom to fuse multiply-adds. Over a few hundred converged iterations in
// double precision that stays near machine epsilon, so a tight tolerance holds.
static constexpr double TOLERANCE = 1.0e-11;

int main(int argc, char** argv) {
    // ---- 1. Load ----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/slab_problem.txt";
    SlabProblem p;
    try {
        p = load_slab(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // Build the S_N angular quadrature once; both paths share it.
    SnQuadrature quad;
    try {
        quad = make_gauss_legendre(p.nord);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<double> phi_cpu;
    int iters_cpu = 0;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    solve_sn_cpu(p, quad, phi_cpu, iters_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU solve (loop timed inside) ---------------------------------
    std::vector<double> phi_gpu;
    int   iters_gpu = 0;
    float gpu_kernel_ms = 0.0f;
    solve_sn_gpu(p, quad, phi_gpu, iters_gpu, &gpu_kernel_ms);

    // ---- 4a. Verify GPU vs CPU (scalar flux agrees) -----------------------
    double err = 0.0;
    for (int i = 0; i < p.ncell; ++i) {
        const double d = std::fabs(phi_cpu[i] - phi_gpu[i]);
        if (d > err) err = d;
    }
    const bool pass_gpu = err <= TOLERANCE;

    // ---- 4b. Analytic check on the INFINITE-MEDIUM limit ------------------
    // In an infinite uniform medium the transport solution reaches the balance
    // "absorption rate = source rate", so
    //   Sigma_a * phi = q   =>   phi_inf = q / Sigma_a = q / (Sigma_t - Sigma_s).
    // Our slab has vacuum boundaries (psi_bc = 0), so flux LEAKS out at the faces
    // and the finite-slab flux stays BELOW phi_inf everywhere -- but the source
    // cells should sit closest to it. We locate the first source cell (q>0) and
    // report its flux next to phi_inf so the learner sees the physics recovered,
    // not just CPU==GPU. This is a sanity indicator printed to stderr, NOT a
    // pass/fail gate (boundary leakage makes exact equality wrong on a finite slab).
    int isrc = -1;
    for (int i = 0; i < p.ncell; ++i) if (p.q[i] > 0.0) { isrc = i; break; }
    const int    ic       = (isrc >= 0) ? isrc : p.ncell / 2;  // a source cell if any
    const double sigma_a  = p.sigma_t[ic] - p.sigma_s[ic];     // absorption there
    const double phi_inf  = (sigma_a > 0.0) ? p.q[ic] / sigma_a : 0.0;

    // Deposition (dose proxy) from the CPU flux (equals GPU within tolerance).
    std::vector<double> dep;
    deposition_field(p, phi_cpu, dep);

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    // Fixed-width %.6e so the bytes are identical every run (both solvers are
    // deterministic; the timings that DO vary go to stderr).
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("S_%d discrete ordinates, %d cells over %.3f cm (h=%.4f cm), tol=%.1e\n",
                p.nord, p.ncell, p.width, p.h(), p.tol);
    std::printf("source iterations: CPU=%d GPU=%d\n", iters_cpu, iters_gpu);

    // Column headers, then per-cell x-center, scalar flux, dose proxy.
    std::printf("  cell        x_cm         scalar_flux         dose_proxy\n");
    for (int i = 0; i < p.ncell; ++i) {
        const double xc = (i + 0.5) * p.h();   // cell-center coordinate [cm]
        std::printf("  %4d   %10.4f   %18.6e   %16.6e\n", i, xc, phi_gpu[i], dep[i]);
    }

    // A couple of scalar summaries that recover the physics.
    double phi_max = 0.0; int imax = 0;
    for (int i = 0; i < p.ncell; ++i)
        if (phi_gpu[i] > phi_max) { phi_max = phi_gpu[i]; imax = i; }
    std::printf("peak scalar flux = %.6e at cell %d (x=%.4f cm)\n",
                phi_max, imax, (imax + 0.5) * p.h());
    std::printf("RESULT: %s (GPU flux matches CPU within tol=%.1e)\n",
                pass_gpu ? "PASS" : "FAIL", TOLERANCE);

    // ---- 5b. Varying detail + analytic sanity -> STDERR -------------------
    std::fprintf(stderr, "[data]    source: %s  (%d cells, S_%d, %.3f cm)\n",
                 path.c_str(), p.ncell, p.nord, p.width);
    std::fprintf(stderr, "[timing]  CPU: %.3f ms   GPU: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing]  teaching artifact -- tiny slabs are launch-bound; the "
                         "GPU's edge grows with (ncell x nord) and multi-D meshes.\n");
    std::fprintf(stderr, "[verify]  max |phi_cpu - phi_gpu| = %.3e  (tolerance %.1e)\n",
                 err, TOLERANCE);
    std::fprintf(stderr, "[physics] source-cell %d flux = %.6e   infinite-medium phi = q/Sigma_a "
                         "= %.6e  (finite slab leaks at the vacuum faces, so flux < phi_inf)\n",
                 ic, phi_gpu[ic], phi_inf);

    return pass_gpu ? 0 : 1;
}
