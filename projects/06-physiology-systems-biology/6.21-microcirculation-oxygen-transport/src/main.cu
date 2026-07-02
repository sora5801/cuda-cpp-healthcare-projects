// ===========================================================================
// src/main.cu  --  Entry point: load problem, run CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 6.21 : Microcirculation & Oxygen Transport
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the problem (a tissue grid + capillary segments) from data/sample.
//   2. Compute the CPU reference PO2 field (reference_cpu.cpp)  -> trusted answer.
//   3. Compute the GPU PO2 field           (kernels.cu)          -> the thing taught.
//   4. VERIFY: assert GPU agrees with CPU within a tolerance    -> correctness.
//   5. REPORT: deterministic result to stdout; timing to stderr.
//
//   STDOUT is kept byte-for-byte deterministic so demo/run_demo can diff it
//   against demo/expected_output.txt. Anything that varies run-to-run (timings)
//   goes to STDERR, which the demo shows but does not diff.
//
//   The deterministic report is a small, meaningful summary of the oxygenation
//   field: its min / mean / max PO2, the HYPOXIC FRACTION (share of tissue below
//   a clinical hypoxia threshold), and PO2 sampled at a few fixed grid points.
//   A physiologist reads this as "does this capillary layout oxygenate the tissue,
//   and where are the hypoxic corners?" -- the real question the model answers.
//
// READ THIS FIRST in the code tour, then oxygen.h (physics) -> reference_cpu.h
// (containers + solve_point) -> kernels.cuh -> kernels.cu. See ../THEORY.md.
// ===========================================================================
#include <algorithm>   // std::sort (for a deterministic percentile, if needed)
#include <cmath>       // std::fabs
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // solve_gpu (GPU path), TissueGrid, OxySource
#include "reference_cpu.h"    // load_problem, solve_cpu, grid_point_coords
#include "util/io.hpp"        // util::CpuTimer

// These two tokens identify the program. They MUST stay in sync with
// demo/expected_output.txt (captured from a real run).
static const char* PROJECT_ID   = "6.21";
static const char* PROJECT_NAME = "Microcirculation & Oxygen Transport";

// Correctness tolerance. CPU and GPU run the SAME double-precision solve_point()
// math, summing sources in the SAME order, so they agree to floating-point
// round-off. 1e-9 mmHg is far below any physiological significance and well above
// the ~1e-12 round-off we actually see (PATTERNS.md section 4: exact-ops case).
static constexpr double TOLERANCE = 1.0e-9;

// Clinical-ish hypoxia threshold (mmHg). Tissue PO2 below this is considered
// hypoxic for the summary "hypoxic fraction". ~10 mmHg is a common teaching
// cutoff for the onset of tissue hypoxia; it is a label, not a diagnosis.
static constexpr double HYPOXIA_MMHG = 10.0;

int main(int argc, char** argv) {
    // ---- 1. Load the problem -----------------------------------------------
    const std::string path = (argc > 1) ? argv[1]
                                        : "data/sample/microvessel_network.txt";
    OxyProblem problem;
    try {
        problem = load_problem(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int n     = grid_size(problem.grid);
    const int n_src = static_cast<int>(problem.sources.size());

    // ---- 2. CPU reference (timed) ------------------------------------------
    std::vector<double> po2_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    solve_cpu(problem, po2_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU result (kernel timed inside the wrapper) -------------------
    std::vector<double> po2_gpu;
    float gpu_kernel_ms = 0.0f;
    solve_gpu(problem, po2_gpu, &gpu_kernel_ms);

    // ---- 4. Verify: worst per-point |CPU - GPU| ----------------------------
    double worst = 0.0;
    for (int i = 0; i < n; ++i) {
        worst = std::fmax(worst, std::fabs(po2_cpu[i] - po2_gpu[i]));
    }
    const bool pass = worst <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    // Field statistics computed from the GPU result (which equals the CPU one).
    double sum = 0.0, mn = po2_gpu[0], mx = po2_gpu[0];
    int hypoxic = 0;
    for (int i = 0; i < n; ++i) {
        const double p = po2_gpu[i];
        sum += p;
        mn = std::fmin(mn, p);
        mx = std::fmax(mx, p);
        if (p < HYPOXIA_MMHG) ++hypoxic;
    }
    const double mean = sum / n;
    const double hypoxic_pct = 100.0 * hypoxic / n;

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("Tissue grid: %d x %d x %d = %d points, spacing %.1f um; %d capillary sources\n",
                problem.grid.nx, problem.grid.ny, problem.grid.nz, n,
                problem.grid.spacing, n_src);
    std::printf("PO2 field (mmHg): min %.4f  mean %.4f  max %.4f\n", mn, mean, mx);
    std::printf("hypoxic fraction (PO2 < %.1f mmHg): %.2f%% (%d / %d points)\n",
                HYPOXIA_MMHG, hypoxic_pct, hypoxic, n);

    // Sample PO2 at five fixed grid points (corners + centre) so the learner can
    // see the spatial variation. Indices are deterministic functions of the grid.
    const int ic = ((problem.grid.nz / 2) * problem.grid.ny + problem.grid.ny / 2)
                       * problem.grid.nx + problem.grid.nx / 2;   // centre point
    const int picks[5] = {0, problem.grid.nx - 1, n / 2, ic, n - 1};
    const char* labels[5] = {"origin", "x-edge", "mid-index", "centre", "far-corner"};
    std::printf("sample PO2 (mmHg) at fixed grid points:\n");
    for (int s = 0; s < 5; ++s) {
        double px, py, pz;
        grid_point_coords(problem.grid, picks[s], px, py, pz);
        std::printf("  %-10s idx %-5d (%.1f,%.1f,%.1f) um: %.4f\n",
                    labels[s], picks[s], px, py, pz, po2_gpu[picks[s]]);
    }
    std::printf("RESULT: %s (GPU field matches CPU within tol=1.0e-09)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%d grid points, %d sources)\n",
                 path.c_str(), n, n_src);
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- the O(N_grid*N_src) direct sum's "
                         "GPU edge grows with grid/source count; a real solver uses an FMM.\n");
    std::fprintf(stderr, "[verify] worst per-point diff = %.3e mmHg  (tolerance %.1e)\n",
                 worst, TOLERANCE);

    // Exit code feeds the demo's pass/fail gate.
    return pass ? 0 : 1;
}
