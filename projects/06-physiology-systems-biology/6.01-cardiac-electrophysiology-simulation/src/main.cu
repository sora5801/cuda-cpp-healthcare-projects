// ===========================================================================
// src/main.cu  --  Entry point: run monodomain on CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 6.1 : Cardiac Electrophysiology Simulation
//
// 5-step shape (the shape EVERY project in this repo follows):
//   1. Load the tissue/model parameters (data/sample).
//   2. CPU reference monodomain solve (reference_cpu.cpp).
//   3. GPU monodomain solve (kernels.cu) -- identical per-cell physics
//      (cardiac_cell.h), so the results must match.
//   4. VERIFY: the GPU voltage field matches the CPU field within tolerance.
//   5. REPORT: a deterministic summary of the final electrical state to stdout;
//      timing (run-to-run varying) to stderr.
//
//   Physically we spark a small S1 patch on a resting sheet of tissue and watch
//   the ACTION-POTENTIAL WAVE spread outward. The deterministic report shows a
//   1-D slice of the voltage through the tissue centre -- the travelling front
//   is visible as the transition from depolarised (~1) to resting (~0).
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Timings go to STDERR (shown, not diffed).
//
// Code tour: start here, then cardiac_cell.h (the per-cell physics), kernels.cu,
// reference_cpu.cpp. The science/GPU-mapping is in ../THEORY.md.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // monodomain_gpu, MonodomainParams
#include "reference_cpu.h"    // load_monodomain, monodomain_cpu, init_state
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "6.1";
static const char* PROJECT_NAME = "Cardiac Electrophysiology Simulation";

// Verification tolerance. CPU and GPU run the SAME double-precision operations
// via the shared cardiac_cell.h, but the GPU may fuse multiply-adds (FMA)
// differently, so over thousands of reaction+diffusion steps the two fields can
// drift by ~1e-9. We verify to 1e-6 -- far below any physically meaningful
// voltage difference -- and say so honestly (docs/PATTERNS.md section 4).
static constexpr double TOLERANCE = 1.0e-6;

int main(int argc, char** argv) {
    // ---- 1. Load ----------------------------------------------------------
    const std::string path =
        (argc > 1) ? argv[1] : "data/sample/tissue_params.txt";
    MonodomainParams p;
    try {
        p = load_monodomain(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<double> V_cpu, w_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    monodomain_cpu(p, V_cpu, w_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU solve (loop timed inside the wrapper) ---------------------
    std::vector<double> V_gpu, w_gpu;
    float gpu_kernel_ms = 0.0f;
    monodomain_gpu(p, V_gpu, w_gpu, &gpu_kernel_ms);

    // ---- 4. Verify (voltage fields agree) ---------------------------------
    double err = 0.0;
    for (std::size_t k = 0; k < V_cpu.size(); ++k) {
        const double d = std::fabs(V_cpu[k] - V_gpu[k]);
        if (d > err) err = d;
    }
    const bool pass = err <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) ----------
    // Summary statistics of the final voltage field (from the GPU result).
    double vmax = -1e30, vmin = 1e30;
    int activated = 0;                    // cells still depolarised (V > 0.5)
    for (double v : V_gpu) {
        if (v > vmax) vmax = v;
        if (v < vmin) vmin = v;
        if (v > 0.5) ++activated;
    }
    const double frac_active = static_cast<double>(activated) /
                               static_cast<double>(V_gpu.size());

    // A horizontal slice of V through the middle row -- the travelling wavefront
    // shows up as the depolarised(~1) -> resting(~0) transition along the slice.
    const int yc = p.ny / 2;

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("monodomain (FitzHugh-Nagumo reaction + diffusion), operator split\n");
    std::printf("grid %dx%d, %d steps, dt=%.4f dx=%.3f D=%.4f (CFL dt_max=%.4f)\n",
                p.nx, p.ny, p.steps, p.dt, p.dx, p.D, cfl_limit(p));
    std::printf("FHN: a=%.3f eps=%.4f b=%.3f | S1 patch %dx%d at (%d,%d) V=%.2f\n",
                p.a, p.eps, p.b, p.stim_w, p.stim_h, p.stim_x0, p.stim_y0, p.stim_v);
    std::printf("final V: min=%.6f max=%.6f | activated(V>0.5)=%d (%.1f%%)\n",
                vmin, vmax, activated, 100.0 * frac_active);
    std::printf("voltage slice V(x, y=%d):\n", yc);
    for (int x = 0; x < p.nx; ++x)
        std::printf(" %.4f", V_gpu[cell_idx(x, yc, p.nx)]);
    std::printf("\n");
    std::printf("RESULT: %s (GPU voltage matches CPU within tol=%.1e)\n",
                pass ? "PASS" : "FAIL", TOLERANCE);

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) -----------------
    std::fprintf(stderr, "[data]   source: %s  (%d x %d grid, %d steps)\n",
                 path.c_str(), p.nx, p.ny, p.steps);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU (kernel loop): %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- on this tiny grid the "
                         "per-step launch overhead can dominate; the GPU's edge "
                         "grows with grid size (a real heart is ~10^8 cells, 3-D).\n");
    std::fprintf(stderr, "[verify] max |V_cpu - V_gpu| = %.3e  (tolerance %.1e)\n",
                 err, TOLERANCE);

    // Exit code feeds the demo's pass/fail gate.
    return pass ? 0 : 1;
}
