// ===========================================================================
// src/main.cu  --  Entry point: run the multi-scale cable, verify, report
// ---------------------------------------------------------------------------
// Project 6.14 : Multi-Scale Physiological Modeling
//
// 5-step shape (the shape every project in this repo follows):
//   1. Load the cable configuration (data/sample or a --arg path).
//   2. CPU reference: serial split-step monodomain simulation (reference_cpu.cpp).
//   3. GPU: one thread per node, ping-pong diffusion, per-step kernels (kernels.cu).
//   4. VERIFY: the GPU final field + activation map match the CPU within a
//      documented, physically-negligible tolerance (long iterative solver -> FMA
//      divergence, PATTERNS.md section 4).
//   5. REPORT: a deterministic activation map + conduction velocity to STDOUT;
//      timings (run-varying) to STDERR.
//
// Code tour: start here, then multiscale.h (the shared per-node physics),
//   kernels.cu (the GPU stepping), reference_cpu.cpp (the serial baseline).
// ===========================================================================
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "kernels.cuh"        // simulate_gpu, CableConfig, CableResult
#include "reference_cpu.h"    // load_cable, simulate_cpu, summarize_activation
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "6.14";
static const char* PROJECT_NAME = "Multi-Scale Physiological Modeling";

// Verification tolerance. This is a LONG iterative double-precision solver:
// thousands of split steps, each with a stencil and an RK4. The GPU's fused
// multiply-add (FMA) and the host compiler's separate mul/add diverge by ~1e-13
// per operation, which accumulates. We therefore verify to a small PHYSICAL
// tolerance rather than pretending the two are bit-identical (PATTERNS.md sec 4,
// and THEORY.md "How we verify correctness"). 1e-6 on a v-field of O(1) is far
// below any physiological or discretization error.
static constexpr double TOLERANCE = 1.0e-6;

// Largest absolute difference between two equal-length double fields (or +inf on
// a size mismatch, so a shape bug can never be mistaken for agreement).
static double max_field_diff(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        worst = std::fmax(worst, std::fabs(a[i] - b[i]));
    return worst;
}

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/cable.txt";
    CableConfig c;
    try {
        c = load_cable(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    CableResult res_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    simulate_cpu(c, res_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU simulation (kernel time measured inside the wrapper) ------
    CableResult res_gpu;
    float gpu_kernel_ms = 0.0f;
    simulate_gpu(c, res_gpu, &gpu_kernel_ms);

    // ---- 4. Verify: compare the final v field, w field, and activation map -
    const double dv  = max_field_diff(res_cpu.v_final, res_gpu.v_final);
    const double dw  = max_field_diff(res_cpu.w_final, res_gpu.w_final);
    const double dact = max_field_diff(res_cpu.activation_time, res_gpu.activation_time);
    const double worst = std::fmax(dv, std::fmax(dw, dact));
    const bool pass = worst <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) ----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("1-D monodomain cable: %d nodes, dx=%.3f, dt=%.4f, %d steps (T=%.2f)\n",
                c.n, c.dx, c.dt, c.steps, total_time(c));
    std::printf("FHN cell: a=%.3f eps=%.3f b=%.3f | tissue D=%.3f | stim %d left nodes\n",
                c.p.a, c.p.eps, c.p.b, c.p.D, c.stim_nodes);

    // Activation map at a handful of evenly-spaced nodes: a propagating action
    // potential makes the activation time INCREASE with position -- the KNOWN
    // answer this synthetic problem is engineered to recover.
    std::printf("activation map (node : x : t_activation):\n");
    const int picks = 6;
    for (int s = 0; s < picks; ++s) {
        const int i = (c.n - 1) * s / (picks - 1);   // 0 .. n-1 inclusive
        const double x = i * c.dx;
        const double ta = res_gpu.activation_time[i];
        if (ta >= 0.0) std::printf("  n%-5d %7.3f  %8.4f\n", i, x, ta);
        else           std::printf("  n%-5d %7.3f  %8s\n",   i, x, "n/a");
    }
    std::printf("nodes activated: %d / %d\n", res_gpu.n_activated, c.n);
    std::printf("conduction velocity: %.4f (space/time)\n", res_gpu.conduction_velocity);
    std::printf("RESULT: %s (GPU field matches CPU within tol=%.1e)\n",
                pass ? "PASS" : "FAIL", TOLERANCE);

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) -----------------
    std::fprintf(stderr, "[data]   source: %s  (%d nodes x %d steps = %lld node-steps)\n",
                 path.c_str(), c.n, c.steps,
                 static_cast<long long>(c.n) * c.steps);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU (kernels): %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- many tiny per-step launches are launch-bound "
                         "on a short cable; the GPU's edge grows with node count (organ-scale meshes).\n");
    std::fprintf(stderr, "[verify] worst field diff = %.3e  (v=%.3e w=%.3e act=%.3e; tol=%.1e)\n",
                 worst, dv, dw, dact, TOLERANCE);

    return pass ? 0 : 1;
}
