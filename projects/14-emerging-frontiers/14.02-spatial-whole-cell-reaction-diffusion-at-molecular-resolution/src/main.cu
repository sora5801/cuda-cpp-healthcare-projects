// ===========================================================================
// src/main.cu  --  Entry point: run reaction-diffusion, verify, report
// ---------------------------------------------------------------------------
// Project 14.02 : Spatial / Whole-Cell Reaction-Diffusion (teaching stencil)
//
// 5-step shape:
//   1. Load params + build the seeded grid (data/sample + init_fields).
//   2. CPU reference RD simulation (reference_cpu.cpp).
//   3. GPU RD simulation (kernels.cu) -- identical per-cell stencil (rd.h).
//   4. VERIFY: the final U/V fields match (within FP tolerance; see THEORY).
//   5. REPORT: deterministic pattern metrics + sampled cells.
//
// From a tiny seed the V field self-organizes into a Turing pattern.
//
// Code tour: start here, then rd.h (the stencil), kernels.cu, reference_cpu.cpp.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // simulate_gpu, RdParams
#include "reference_cpu.h"    // load_rd, init_fields, simulate_cpu
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "14.2";
static const char* PROJECT_NAME = "Spatial Reaction-Diffusion (Gray-Scott)";

// Over thousands of nonlinear steps, CPU and GPU drift at the float-FMA level;
// the field values are O(1), so a 1e-3 tolerance is well within "same pattern".
static constexpr double TOLERANCE = 1.0e-3;

int main(int argc, char** argv) {
    // ---- 1. Load + seed ----------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/grayscott_params.txt";
    RdParams P;
    try {
        P = load_rd(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    std::vector<double> U0, V0;
    init_fields(P, U0, V0);

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<double> U_cpu = U0, V_cpu = V0;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    simulate_cpu(P, U_cpu, V_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU simulation (loop timed) -----------------------------------
    std::vector<double> U_gpu = U0, V_gpu = V0;
    float gpu_kernel_ms = 0.0f;
    simulate_gpu(P, U_gpu, V_gpu, &gpu_kernel_ms);

    // ---- 4. Verify (final fields agree) -----------------------------------
    double worst = 0.0;
    for (int i = 0; i < P.nx * P.ny; ++i) {
        worst = std::fmax(worst, std::fabs(U_cpu[i] - U_gpu[i]));
        worst = std::fmax(worst, std::fabs(V_cpu[i] - V_gpu[i]));
    }
    const bool pass = worst <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    // Pattern metrics from the GPU field: total V "mass", peak V, and how many
    // cells are "active" (V above a threshold) -- a measure of pattern coverage.
    double total_v = 0.0, max_v = 0.0;
    int active = 0;
    for (int i = 0; i < P.nx * P.ny; ++i) {
        total_v += V_gpu[i];
        max_v = std::fmax(max_v, V_gpu[i]);
        if (V_gpu[i] > 0.2) ++active;
    }
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("Gray-Scott: %dx%d grid, %d steps, Du=%.3f Dv=%.3f F=%.4f k=%.4f\n",
                P.nx, P.ny, P.steps, P.Du, P.Dv, P.F, P.k);
    std::printf("pattern: total V=%.4f, max V=%.4f, active cells (V>0.2)=%d of %d\n",
                total_v, max_v, active, P.nx * P.ny);
    std::printf("V along center row (8 samples):");
    const int cy = P.ny / 2;
    for (int s = 0; s < 8; ++s) {
        const int x = (s * (P.nx - 1)) / 7;
        std::printf(" %.4f", V_gpu[cy * P.nx + x]);
    }
    std::printf("\n");
    std::printf("RESULT: %s (GPU field matches CPU within tol=1.0e-03)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d cells, %d steps)\n", path.c_str(), P.nx * P.ny, P.steps);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- the GPU's edge grows with grid size; whole-cell "
                         "molecular RD needs multi-GPU systems.\n");
    std::fprintf(stderr, "[verify] worst field diff = %.3e  (tolerance %.1e)\n", worst, TOLERANCE);

    return pass ? 0 : 1;
}
