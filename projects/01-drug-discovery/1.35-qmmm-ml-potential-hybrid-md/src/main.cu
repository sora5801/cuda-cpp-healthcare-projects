// ===========================================================================
// src/main.cu  --  Entry point: integrate the hybrid-MD ensemble, verify, report
// ---------------------------------------------------------------------------
// Project 1.35 : QMMM/ML Potential Hybrid MD   (reduced-scope teaching version)
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the ensemble config (M trajectories, dt, steps, perturbation amp).
//   2. CPU reference: run every trajectory serially         -> trusted answer.
//   3. GPU: one thread per trajectory (full velocity-Verlet) -> the thing taught.
//   4. VERIFY: assert the GPU summaries match the CPU within a tolerance.
//   5. REPORT: deterministic results to stdout; timing to stderr.
//
//   Each "trajectory" is a tiny hybrid NNP/MM molecular-dynamics run: a 1-D chain
//   whose reactive center is described by a (surrogate) neural-network potential
//   and whose environment is classical Lennard-Jones, coupled by mechanical
//   embedding across a link-atom boundary. See nnpmm.h for the physics and
//   ../THEORY.md for the "why". REDUCED-SCOPE TEACHING VERSION (CLAUDE.md §13):
//   the NNP weights are fixed/synthetic, standing in for a model trained on QM
//   data (MACE/NequIP on Transition1x/SPICE). NOT a clinical or research tool.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Run-varying numbers (timings) go to STDERR, which
//   the demo shows but does not diff.
//
// READ THIS FIRST in the code tour, then nnpmm.h (the physics), kernels.cu (the
// GPU path), reference_cpu.cpp (the baseline).
// ===========================================================================
#include <cmath>     // std::fabs, std::fmax
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // integrate_gpu, EnsembleConfig, TrajResult
#include "reference_cpu.h"    // load_ensemble, integrate_cpu, ensemble_size
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "1.35";
static const char* PROJECT_NAME = "QMMM/ML Potential Hybrid MD";

// Verification tolerance. The whole trajectory is computed in DOUBLE precision
// on both sides via the SAME shared functions (nnpmm.h), so the only source of
// disagreement is floating-point reassociation: the GPU fuses multiply-adds
// (FMA) where the host compiler may not, and that ~1e-15 per-op difference
// compounds over hundreds of integration steps. 1e-6 is comfortably above that
// drift yet far below any physically meaningful energy scale here -- an honest,
// documented tolerance (docs/PATTERNS.md §4, the "long iterative solver" case).
static constexpr double TOLERANCE = 1.0e-6;

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1]
                                        : "data/sample/ensemble_params.txt";
    EnsembleConfig c;
    try {
        c = load_ensemble(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int M = ensemble_size(c);

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<TrajResult> res_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    integrate_cpu(c, res_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU ensemble (kernel timed) -----------------------------------
    std::vector<TrajResult> res_gpu;
    float gpu_kernel_ms = 0.0f;
    integrate_gpu(c, res_gpu, &gpu_kernel_ms);

    // ---- 4. Verify (worst per-member difference across all summary fields) -
    double worst = 0.0;
    for (int i = 0; i < M; ++i) {
        worst = std::fmax(worst, std::fabs(res_cpu[i].final_pe     - res_gpu[i].final_pe));
        worst = std::fmax(worst, std::fabs(res_cpu[i].final_total  - res_gpu[i].final_total));
        worst = std::fmax(worst, std::fabs(res_cpu[i].max_force    - res_gpu[i].max_force));
        worst = std::fmax(worst, std::fabs(res_cpu[i].energy_drift - res_gpu[i].energy_drift));
    }
    const bool pass = worst <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) ----------
    // A physical sanity check we report so the science -- not just CPU==GPU
    // agreement -- is visible (docs/PATTERNS.md §4, the "stronger check"):
    //   * worst energy conservation = max over members of |final - initial|
    //     total energy. Velocity-Verlet is SYMPLECTIC, so this stays small and
    //     BOUNDED (no secular drift), validating that the dynamics are stable.
    //   * we also print the unperturbed member (idx M/2) as a stable anchor.
    const int mid = M / 2;                        // the (near-)unperturbed member
    double worst_conservation = 0.0;
    for (int i = 0; i < M; ++i)
        worst_conservation = std::fmax(worst_conservation, res_gpu[i].energy_drift);

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("REDUCED-SCOPE TEACHING VERSION: NNP weights are fixed/synthetic.\n");
    std::printf("hybrid NNP/MM chain: %d atoms (%d MM, link@%d, %d ML), NNP=%dx%d MLP\n",
                N_ATOMS, LINK_IDX, LINK_IDX, N_ATOMS - LINK_IDX, N_HID, N_G);
    std::printf("ensemble: %d trajectories, %d steps @ dt=%.3f, link perturbation +/-%.3f\n",
                M, c.steps, c.dt, c.amp);
    std::printf("sample members (idx perturb -> finalPE finalE maxForce):\n");
    // Five evenly-spaced members so the table is stable regardless of M (>=5).
    const int picks[5] = {0, M / 4, M / 2, (3 * M) / 4, M - 1};
    for (int s = 0; s < 5; ++s) {
        const int i = picks[s];
        const double perturb = member_perturbation(i, M, c.amp);
        std::printf("  m%-4d %+.3f -> %12.6f %12.6f %12.6f\n",
                    i, perturb, res_gpu[i].final_pe,
                    res_gpu[i].final_total, res_gpu[i].max_force);
    }
    std::printf("unperturbed (m%d): finalPE=%.6f  finalE=%.6f\n",
                mid, res_gpu[mid].final_pe, res_gpu[mid].final_total);
    std::printf("worst energy conservation (max |finalE - initialE|) = %.6f\n",
                worst_conservation);
    std::printf("RESULT: %s (GPU ensemble matches CPU within tol=1.0e-06)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) -----------------
    std::fprintf(stderr, "[data]   source: %s  (%d members)\n", path.c_str(), M);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- the GPU's edge grows with "
                         "ensemble size; real active-learning runs 10^4-10^6 trajectories.\n");
    std::fprintf(stderr, "[verify] worst per-member diff = %.3e  (tolerance %.1e)\n",
                 worst, TOLERANCE);

    return pass ? 0 : 1;
}
