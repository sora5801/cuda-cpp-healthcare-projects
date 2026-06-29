// ===========================================================================
// src/main.cu  --  Entry point: simulate soft tissue, verify, report
// ---------------------------------------------------------------------------
// Project 10.02 : Real-Time Soft-Tissue Deformation for Surgical Simulation
//
// 5-step shape:
//   1. Load params + build the pinned mesh (data/sample + init_mesh).
//   2. CPU reference PBD simulation (reference_cpu.cpp).
//   3. GPU PBD simulation (kernels.cu) -- identical per-particle physics (pbd.h).
//   4. VERIFY: the final mesh positions match (CPU vs GPU).
//   5. REPORT: deterministic sampled particle positions + drape depth.
//
// The pinned top edge holds; the rest of the sheet drapes downward under gravity.
//
// Code tour: start here, then pbd.h (the PBD math), kernels.cu, reference_cpu.cpp.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // simulate_gpu, PbdParams, Vec3
#include "reference_cpu.h"    // load_pbd, init_mesh, simulate_cpu
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "10.2";
static const char* PROJECT_NAME = "Real-Time Soft-Tissue Deformation (PBD)";

// Tolerance on final particle positions (magnitude ~O(10) here). Over the many
// thousands of constraint iterations, the CPU and GPU drift at the ~1e-5 level
// because their floating-point contraction (FMA) differs -- so we verify to a
// physically-negligible 1e-3, not to round-off. (See THEORY "Numerical notes".)
static constexpr double TOLERANCE = 1.0e-3;

int main(int argc, char** argv) {
    // ---- 1. Load + build the mesh -----------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/cloth_params.txt";
    PbdParams P;
    try {
        P = load_pbd(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    std::vector<Vec3> x0, v0;
    std::vector<double> w;
    init_mesh(P, x0, v0, w);

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<Vec3> x_cpu = x0, v_cpu = v0;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    simulate_cpu(P, x_cpu, v_cpu, w);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU simulation (loop timed) -----------------------------------
    std::vector<Vec3> x_gpu = x0, v_gpu = v0;
    float gpu_kernel_ms = 0.0f;
    simulate_gpu(P, x_gpu, v_gpu, w, &gpu_kernel_ms);

    // ---- 4. Verify (final positions agree) --------------------------------
    double worst = 0.0;
    for (std::size_t i = 0; i < x_cpu.size(); ++i)
        worst = std::fmax(worst, length(x_cpu[i] - x_gpu[i]));
    const bool pass = worst <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    // Drape depth = how far the lowest free particle sagged (-min y).
    double min_y = 0.0;
    for (const Vec3& p : x_gpu) min_y = std::fmin(min_y, p.y);
    auto P_at = [&](int r, int c) { return x_gpu[r * P.C + c]; };

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("PBD mesh: %dx%d particles (top row pinned), %d steps, %d iters, "
                "stiffness=%.2f, gravity=%.1f\n",
                P.R, P.C, P.steps, P.iters, P.stiffness, P.gravity);
    const Vec3 a = P_at(P.R - 1, 0), b = P_at(P.R - 1, P.C - 1), m = P_at(P.R / 2, P.C / 2);
    std::printf("free corner (%d,0)      = (%.4f, %.4f, %.4f)\n", P.R - 1, a.x, a.y, a.z);
    std::printf("free corner (%d,%d)     = (%.4f, %.4f, %.4f)\n", P.R - 1, P.C - 1, b.x, b.y, b.z);
    std::printf("center      (%d,%d)     = (%.4f, %.4f, %.4f)\n", P.R / 2, P.C / 2, m.x, m.y, m.z);
    std::printf("drape depth (max sag)  = %.4f\n", -min_y);
    std::printf("RESULT: %s (GPU mesh matches CPU within tol=1.0e-03)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d particles)\n", path.c_str(), P.R * P.C);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- surgical sims need sub-10 ms on 10^5+ particles; "
                         "the GPU's edge grows with mesh size.\n");
    std::fprintf(stderr, "[verify] worst particle position diff = %.3e  (tolerance %.1e)\n", worst, TOLERANCE);

    return pass ? 0 : 1;
}
