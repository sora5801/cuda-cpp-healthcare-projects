// ===========================================================================
// src/main.cu  --  Entry point: integrate a QM/MM trajectory ensemble, verify, report
// ---------------------------------------------------------------------------
// Project 1.23 : QM/MM Molecular Dynamics   (reduced-scope teaching version)
//
// WHAT THIS FILE DOES  (the 5-step shape every project in this repo follows)
//   1. Load the ensemble config (a field x initial-position parameter sweep).
//   2. CPU reference: integrate every trajectory serially (reference_cpu.cpp).
//   3. GPU: one thread per trajectory, full velocity-Verlet loop each (kernels.cu).
//   4. VERIFY: per-member results match (same QM/MM core -> same numbers).
//   5. REPORT: deterministic sample trajectories + ensemble summary to STDOUT;
//      timings and run-varying detail to STDERR.
//
//   STDOUT is kept byte-for-byte deterministic (fixed precision) so
//   demo/run_demo can diff it against demo/expected_output.txt. Timings go to
//   STDERR, which the demo shows but does not diff (PATTERNS.md §3).
//
// Code tour: start here, then qmmm.h (the QM solve + Verlet), kernels.cu (GPU
// twin), reference_cpu.cpp (serial baseline). See ../THEORY.md for the "why".
// ===========================================================================
#include <cmath>     // std::fabs, std::fmax
#include <cstdio>    // std::printf, std::fprintf
#include <string>
#include <vector>

#include "kernels.cuh"        // integrate_gpu, EnsembleConfig, qmmm::TrajResult
#include "reference_cpu.h"    // load_ensemble, integrate_cpu, member_params
#include "qmmm.h"             // qmmm constants (for the R0-style summary)
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "1.23";
static const char* PROJECT_NAME = "QM/MM Molecular Dynamics";

// Verification tolerance. The CPU and GPU run the SAME double-precision Verlet
// loop calling the SAME __host__ __device__ core, so they agree to ~round-off.
// Over a few thousand steps the GPU's fused multiply-add (FMA) can diverge from
// the host compiler's separate mul+add by a few ULPs, so we verify to a tight
// but non-zero tolerance rather than demanding bit-identity (PATTERNS.md §4,
// "machine-precision band"). The measured worst diff is printed to stderr.
static constexpr double TOLERANCE = 1.0e-9;

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/ensemble_params.txt";
    EnsembleConfig c;
    try {
        c = load_ensemble(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int M = ensemble_size(c);

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<qmmm::TrajResult> res_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    integrate_cpu(c, res_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU ensemble (kernel timed) -----------------------------------
    std::vector<qmmm::TrajResult> res_gpu;
    float gpu_kernel_ms = 0.0f;
    integrate_gpu(c, res_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    // Compare every continuous per-member output; take the worst absolute diff.
    double worst = 0.0;
    for (int i = 0; i < M; ++i) {
        worst = std::fmax(worst, std::fabs(res_cpu[i].final_x      - res_gpu[i].final_x));
        worst = std::fmax(worst, std::fabs(res_cpu[i].final_energy - res_gpu[i].final_energy));
        worst = std::fmax(worst, std::fabs(res_cpu[i].min_gap      - res_gpu[i].min_gap));
        worst = std::fmax(worst, std::fabs(res_cpu[i].frac_product - res_gpu[i].frac_product));
    }
    const bool pass = worst <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    // Ensemble summary: how many trajectories ended with the proton TRANSFERRED
    // (on the acceptor side, x>0), and the average final adiabatic energy.
    int transferred = 0;
    double sum_final_energy = 0.0, min_gap_overall = 1e300;
    for (int i = 0; i < M; ++i) {
        if (res_gpu[i].transferred) ++transferred;
        sum_final_energy += res_gpu[i].final_energy;
        min_gap_overall = std::fmin(min_gap_overall, res_gpu[i].min_gap);
    }

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("Reduced-scope teaching model: proton transfer on a 2-state QM surface\n");
    std::printf("  with classical (MM) electrostatic embedding + velocity-Verlet MD.\n");
    std::printf("QM/MM ensemble: %d trajectories (%d field x %d x0), %d Verlet steps @ dt=%.3f\n",
                M, c.nf, c.nx, c.steps, c.dt);
    std::printf("electronic coupling beta=%.3f  well minima x=[%.2f,%.2f]  proton mass=%.2f\n",
                qmmm::COUPLING, qmmm::X_L, qmmm::X_R, qmmm::PROTON_MASS);
    std::printf("sample trajectories (field x0 -> final_x final_E min_gap %%product transferred):\n");
    const int picks[5] = {0, M / 4, M / 2, (3 * M) / 4, M - 1};
    for (int s = 0; s < 5; ++s) {
        const int i = picks[s];
        double field, x0; member_params(c, i, field, x0);
        const qmmm::TrajResult& r = res_gpu[i];
        std::printf("  t%-5d: %+.3f %+.3f -> %+8.5f %+9.5f %8.5f %6.2f %d\n",
                    i, field, x0, r.final_x, r.final_energy, r.min_gap,
                    100.0 * r.frac_product, r.transferred);
    }
    std::printf("ensemble: %d/%d trajectories transferred; mean final E = %+.5f; overall min gap = %.5f\n",
                transferred, M, sum_final_energy / M, min_gap_overall);
    std::printf("RESULT: %s (GPU ensemble matches CPU within tol=1.0e-09)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%d trajectories)\n", path.c_str(), M);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernel: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- speed-up grows with ensemble size; real "
                         "reactive sampling launches 10^3-10^5 trajectories.\n");
    std::fprintf(stderr, "[verify] worst per-member diff = %.3e  (tolerance %.1e)\n", worst, TOLERANCE);

    // Exit code feeds the demo's pass/fail gate.
    return pass ? 0 : 1;
}
