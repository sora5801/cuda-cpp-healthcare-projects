// ===========================================================================
// src/main.cu  --  Entry point: run condensate ensemble, verify, propose next
// ---------------------------------------------------------------------------
// Project 2.34 : Biophysical Simulation of Biomolecular Condensates
//                (Active Learning Loop)  --  reduced-scope teaching version
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the experiment config (data/sample/condensate_ensemble.txt).
//   2. CPU reference: integrate every replica serially (reference_cpu.cpp).
//   3. GPU: one thread per replica, full CG-MD trajectory each (kernels.cu).
//   4. VERIFY: per-replica (D, Rg) match between CPU and GPU within tolerance.
//   5. REPORT: deterministic ensemble table + the active-learning PROPOSAL
//      (the lambda whose measured diffusion best matches the target) to stdout;
//      timings to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Run-varying numbers (timings) go to STDERR, which
//   the demo shows but does not diff (PATTERNS.md section 3).
//
//   Code tour: start here, then condensate.h (the physics + integrator),
//   kernels.cuh -> kernels.cu (the GPU twin), reference_cpu.cpp (baseline + the
//   active-learning acquisition). See ../THEORY.md for the "why".
// ===========================================================================
#include <cmath>     // std::fabs, std::fmax
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // integrate_gpu, EnsembleConfig, ReplicaResult
#include "reference_cpu.h"    // load_ensemble, integrate_cpu, propose_next_lambda
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "2.34";
static const char* PROJECT_NAME = "Biophysical Simulation of Biomolecular Condensates (Active Learning Loop)";

// Correctness tolerance. Each trajectory is hundreds of double-precision steps,
// each with a fused-multiply-add the GPU and host compiler schedule slightly
// differently, so CPU and GPU drift by ~1e-9..1e-12 by the end even though they
// run identical math. We verify the measured properties (D, Rg) to a physically
// negligible tolerance rather than pretending they are bit-identical (PATTERNS s4).
static constexpr double TOLERANCE = 1.0e-6;

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1]
                                        : "data/sample/condensate_ensemble.txt";
    EnsembleConfig c;
    try {
        c = load_ensemble(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int M = ensemble_size(c);

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<ReplicaResult> res_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    integrate_cpu(c, res_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU ensemble (kernel timed) -----------------------------------
    std::vector<ReplicaResult> res_gpu;
    float gpu_kernel_ms = 0.0f;
    integrate_gpu(c, res_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    // Worst absolute disagreement in the two reported properties across members.
    double worst = 0.0;
    for (int i = 0; i < M; ++i) {
        worst = std::fmax(worst, std::fabs(res_cpu[i].diffusion - res_gpu[i].diffusion));
        worst = std::fmax(worst, std::fabs(res_cpu[i].rg        - res_gpu[i].rg));
    }
    const bool pass = worst <= TOLERANCE;

    // ---- 5. The active-learning proposal (deterministic, from the CPU result)
    // The loop's headline output: which candidate stickiness Bayesian optimization
    // would simulate next, i.e. the one whose measured diffusion best matches the
    // experimental target_D. We use the CPU results so the proposal is independent
    // of the (verified-equal) GPU path -> reproducible stdout.
    int best_m = 0;
    const double best_lambda = propose_next_lambda(c, res_cpu, &best_m);

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("reduced-scope teaching model: coarse-grained Brownian-dynamics condensate ensemble\n");
    std::printf("ensemble: %d candidate sequences (stickiness lambda in [%.2f, %.2f])\n",
                M, c.lambda_lo, c.lambda_hi);
    std::printf("CG-MD: %d beads, %d steps (dt=%.3f, eq=%d), kT=%.2f, target D=%.5f\n",
                c.model.n_beads, c.model.steps, c.model.dt, c.model.eq_steps,
                c.model.kT, c.target_D);
    std::printf("sample replicas (lambda -> Rg  D  |D-target|):\n");
    // Print five evenly-spaced members so the table is fixed-size and shows the
    // lambda -> (compactness, mobility) trend the model is meant to teach.
    const int picks[5] = {0, M / 4, M / 2, (3 * M) / 4, M - 1};
    for (int s = 0; s < 5; ++s) {
        const int i = picks[s];
        std::printf("  m%-4d lambda=%.3f -> Rg=%.5f  D=%.5f  |dD|=%.5f\n",
                    i, res_cpu[i].lambda, res_cpu[i].rg, res_cpu[i].diffusion,
                    std::fabs(res_cpu[i].diffusion - c.target_D));
    }
    // The active-learning headline: the proposed next sequence.
    std::printf("active-learning proposal: member m%d, lambda=%.3f "
                "(D=%.5f closest to target %.5f)\n",
                best_m, best_lambda, res_cpu[best_m].diffusion, c.target_D);
    std::printf("RESULT: %s (GPU ensemble matches CPU within tol=1.0e-06)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%d replicas)\n", path.c_str(), M);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- the GPU's edge grows with ensemble size; "
                         "real active-learning iterations run hundreds of replicas.\n");
    std::fprintf(stderr, "[verify] worst per-replica diff = %.3e  (tolerance %.1e)\n",
                 worst, TOLERANCE);

    return pass ? 0 : 1;
}
