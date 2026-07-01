// ===========================================================================
// src/main.cu  --  Entry point: load CT, run CPU + GPU SC dose, verify, report
// ---------------------------------------------------------------------------
// Project 5.4 : Collapsed-Cone / Superposition-Convolution Dose  (2-D teaching model)
//
// THE 5-STEP SHAPE (every project in this repo follows it)
//   1. Load the density grid + beam/kernel parameters (data/sample).
//   2. CPU reference: TERMA ray-trace (terma_cpu) then collapsed-cone
//      superposition (dose_cpu) -> the trusted integer dose grid.
//   3. GPU: both stages in kernels.cu (dose_gpu) -> the same grid, computed with
//      one thread per column (stage 1) and one thread per source voxel (stage 2).
//   4. VERIFY: assert the GPU integer dose grid EQUALS the CPU grid exactly (the
//      integer/fixed-point trick makes this an exact ==, not a tolerance), and
//      cross-check the double-precision TERMA within a tiny FP tolerance.
//   5. REPORT: a deterministic central-axis depth-dose profile to stdout; timing
//      to stderr. stdout is what demo/run_demo diffs against expected_output.txt.
//
// Code tour: start here, then ccc_physics.h (the shared physics), kernels.cuh ->
// kernels.cu (the GPU twins), reference_cpu.cpp (the serial baseline). The "why"
// lives in ../THEORY.md.
// ===========================================================================
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

#include "kernels.cuh"        // dose_gpu (GPU path), CccParams, DoseProblem
#include "reference_cpu.h"    // load_dose_problem, terma_cpu, dose_cpu
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "5.4";
static const char* PROJECT_NAME = "Collapsed-Cone / Superposition-Convolution Dose";

// TERMA is computed in double precision by the SAME code on both sides, so it is
// expected to match to the last bit; we still verify it within a tiny tolerance
// in case a compiler contracts an FMA differently. The DOSE grid is INTEGER, so
// it must match EXACTLY (== 0 mismatches) -- that is the headline check.
static constexpr double TERMA_TOL = 1.0e-9;

int main(int argc, char** argv) {
    // ---- 1. Load the problem -----------------------------------------------
    const std::string path =
        (argc > 1) ? argv[1] : "data/sample/phantom.txt";
    DoseProblem prob;
    try {
        prob = load_dose_problem(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const CccParams& P = prob.P;
    const int total = P.nx * P.ny;

    // ---- 2. CPU reference (both stages, timed) -----------------------------
    std::vector<double>    terma_cpu_v;
    std::vector<long long> dose_cpu_v;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    terma_cpu(prob, terma_cpu_v);
    dose_cpu(prob, terma_cpu_v, dose_cpu_v);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU (both stages, kernel-timed inside the wrapper) -------------
    std::vector<double>    terma_gpu_v;
    std::vector<long long> dose_gpu_v;
    float gpu_kernel_ms = 0.0f;
    dose_gpu(prob, dose_gpu_v, terma_gpu_v, &gpu_kernel_ms);

    // ---- 4. Verify ----------------------------------------------------------
    // (a) DOSE: exact integer equality (the determinism payoff, PATTERNS.md §3).
    long long dose_mismatches = 0;
    unsigned long long total_units = 0ULL;
    for (int i = 0; i < total; ++i) {
        if (dose_cpu_v[i] != dose_gpu_v[i]) ++dose_mismatches;
        total_units += static_cast<unsigned long long>(dose_gpu_v[i]);
    }
    // (b) TERMA: double-precision cross-check within a tiny tolerance.
    double terma_max_abs = 0.0;
    for (int i = 0; i < total; ++i) {
        const double d = std::fabs(terma_cpu_v[i] - terma_gpu_v[i]);
        if (d > terma_max_abs) terma_max_abs = d;
    }
    const bool pass = (dose_mismatches == 0) && (terma_max_abs <= TERMA_TOL);

    // Central beam column (integer-defined, so deterministic) for the profile.
    const int cx = (P.beam_x0 + P.beam_x1) / 2;

    // Peak-dose voxel (first-wins scan => deterministic tie-break).
    int peak_idx = 0;
    for (int i = 1; i < total; ++i)
        if (dose_gpu_v[i] > dose_gpu_v[peak_idx]) peak_idx = i;
    const int peak_x = peak_idx % P.nx;
    const int peak_y = peak_idx / P.nx;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("grid %dx%d voxels @ %.2f cm, mu/rho=%.4f cm^2/g, %d cones, a=%.3f /(g/cm^2)\n",
                P.nx, P.ny, P.voxel_cm, P.mu_over_rho, P.n_cones, P.kernel_a);
    std::printf("beam columns [%d..%d], dose_scale=%.0f units/dose\n",
                P.beam_x0, P.beam_x1, P.dose_scale);
    std::printf("total deposited = %llu dose-units; peak voxel (x=%d,y=%d)\n",
                total_units, peak_x, peak_y);
    // Central-axis depth-dose: integer dose-units down column cx, one per row.
    // This is the classic PDD (percent-depth-dose) curve every med-phys learner
    // recognizes: build-up near the surface, then exponential-ish falloff, with a
    // kink at any density interface (lung/bone) -- the whole point of SC dose.
    std::printf("central-axis depth-dose (column x=%d), dose-units per row y=0..%d:\n ",
                cx, P.ny - 1);
    for (int y = 0; y < P.ny; ++y)
        std::printf(" %lld", dose_gpu_v[static_cast<size_t>(y) * P.nx + cx]);
    std::printf("\n");
    std::printf("RESULT: %s (GPU dose grid matches CPU exactly; TERMA within %.0e)\n",
                pass ? "PASS" : "FAIL", TERMA_TOL);

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s\n", path.c_str());
    std::fprintf(stderr, "[timing] CPU (TERMA+CCC): %.3f ms   GPU (TERMA+CCC): %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- a tiny grid is launch-bound; "
                         "the GPU's edge grows toward clinical 512^3 volumes x ~400 cones.\n");
    std::fprintf(stderr, "[verify] dose-grid mismatches = %lld (integer => atomics commute)\n",
                 dose_mismatches);
    std::fprintf(stderr, "[verify] TERMA max_abs_err = %.3e (tol %.1e)\n",
                 terma_max_abs, TERMA_TOL);

    return pass ? 0 : 1;
}
