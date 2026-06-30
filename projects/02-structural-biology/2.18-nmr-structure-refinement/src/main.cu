// ===========================================================================
// src/main.cu  --  Entry point: anneal an NMR ensemble, verify, report
// ---------------------------------------------------------------------------
// Project 2.18 : NMR Structure Refinement
//
// 5-step shape (the same skeleton every project in this repo follows):
//   1. Load the refinement job (chain + NOE restraints + annealing schedule).
//   2. CPU reference: anneal every replica serially (reference_cpu.cpp).
//   3. GPU: one thread per replica, full SA loop each (kernels.cu).
//   4. VERIFY: per-replica results match (same shared annealer -> same numbers).
//   5. REPORT: deterministic best-replica structure + ensemble summary to stdout;
//      timing + run-varying detail to stderr (so stdout is byte-stable for the demo).
//
// Code tour: start here, then nmr_refine.h (RNG + energy + the SA loop), then
// kernels.cu (the one-thread-per-replica launch) and reference_cpu.cpp (the serial
// twin). README "Code tour" walks the same path.
// ===========================================================================
#include <cmath>      // std::fabs, std::fmax
#include <cstdio>     // std::printf, std::fprintf
#include <string>
#include <vector>

#include "kernels.cuh"        // anneal_ensemble_gpu, RefineConfig, ReplicaResult
#include "reference_cpu.h"    // load_config, anneal_ensemble_cpu
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "2.18";
static const char* PROJECT_NAME = "NMR Structure Refinement";

// Verification tolerance on the per-replica BEST ENERGY. The CPU and GPU run the
// identical shared annealer (nmr_refine.h), so in exact arithmetic every replica
// would match to the bit. In practice the host compiler and nvcc can contract
// multiply-adds differently, nudging an energy by ~1e-9; over a long Metropolis
// trajectory that is a genuine, teachable effect (PATTERNS.md section 4). We
// therefore verify the continuous energy to a small physical tolerance and the
// DISCRETE restraint-satisfaction count EXACTLY (an integer cannot drift). The
// energy unit is arbitrary "restraint energy"; 1e-4 is far below any structural
// significance here. THEORY.md section 6 expands on this.
static constexpr double ENERGY_TOL = 1.0e-4;

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/restraints.txt";
    RefineConfig c;
    try {
        c = load_config(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int M = c.n_replicas;

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<ReplicaResult> res_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    anneal_ensemble_cpu(c, res_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU ensemble (kernel timed) -----------------------------------
    std::vector<ReplicaResult> res_gpu;
    float gpu_kernel_ms = 0.0f;
    anneal_ensemble_gpu(c, res_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    // Energy: max |dE| over replicas (continuous, tolerance ENERGY_TOL).
    // Satisfaction count + accepted moves: must match EXACTLY (integers).
    double worst_energy = 0.0;
    bool   discrete_ok  = true;
    for (int r = 0; r < M; ++r) {
        worst_energy = std::fmax(worst_energy,
                                 std::fabs(res_cpu[r].final_energy - res_gpu[r].final_energy));
        if (res_cpu[r].n_satisfied != res_gpu[r].n_satisfied) discrete_ok = false;
        if (res_cpu[r].accepted    != res_gpu[r].accepted)    discrete_ok = false;
    }
    const bool pass = discrete_ok && (worst_energy <= ENERGY_TOL);

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    // The published "structure" is the LOWEST-ENERGY replica of the ensemble.
    // Find it from the GPU results (identical to CPU within tolerance). We pick
    // the best by (energy, then index) so ties resolve deterministically.
    int best = 0;
    for (int r = 1; r < M; ++r) {
        if (res_gpu[r].final_energy < res_gpu[best].final_energy) best = r;
    }

    // Ensemble-wide summary: how many replicas converged to a "good" structure
    // (all NOE restraints satisfied), and the spread of best energies.
    int all_sat = 0;
    double sum_e = 0.0, max_e = 0.0;
    for (int r = 0; r < M; ++r) {
        if (res_gpu[r].n_satisfied == c.n_restraints) ++all_sat;
        sum_e += res_gpu[r].final_energy;
        max_e = std::fmax(max_e, res_gpu[r].final_energy);
    }

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("ensemble SA: %d replicas x %d steps; chain=%d beads, %d NOE restraints\n",
                M, c.n_steps, c.n_beads, c.n_restraints);
    std::printf("schedule: T %.2f -> %.2f (geometric), trial sigma=%.2f A, bond=%.2f A\n",
                c.T_hot, c.T_cold, c.step_sigma, c.bond_len);
    std::printf("best replica: #%d  energy=%.4f  restraints satisfied=%d/%d\n",
                best, res_gpu[best].final_energy, res_gpu[best].n_satisfied, c.n_restraints);

    // A few deterministic sample replicas across the ensemble (index, energy,
    // satisfied count). %.4f keeps the print stable to within ENERGY_TOL.
    std::printf("sample replicas (idx -> energy satisfied):\n");
    const int picks[5] = {0, M / 4, M / 2, (3 * M) / 4, M - 1};
    for (int s = 0; s < 5; ++s) {
        const int r = picks[s];
        std::printf("  r%-4d: %9.4f  %d/%d\n",
                    r, res_gpu[r].final_energy, res_gpu[r].n_satisfied, c.n_restraints);
    }
    std::printf("ensemble: %d/%d replicas satisfy all restraints; mean best energy=%.4f; max=%.4f\n",
                all_sat, M, sum_e / M, max_e);
    std::printf("RESULT: %s (GPU ensemble matches CPU: counts exact, energy within %.1e)\n",
                pass ? "PASS" : "FAIL", ENERGY_TOL);

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d replicas)\n", path.c_str(), M);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernel: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- real NMR ensembles run 100s-1000s of "
                         "replicas with full force fields; the GPU edge grows with replica count.\n");
    std::fprintf(stderr, "[verify] worst per-replica energy diff = %.3e (tol %.1e); "
                         "discrete counts match = %s\n",
                 worst_energy, ENERGY_TOL, discrete_ok ? "yes" : "no");

    return pass ? 0 : 1;
}
