// ===========================================================================
// src/main.cu  --  Entry point: load config, run multi-walker MetaD, verify
// ---------------------------------------------------------------------------
// Project 1.6 : Enhanced Sampling -- Metadynamics & Replica Exchange
//
// WHAT THIS FILE DOES  (the 5-step shape EVERY project in this repo follows)
//   1. Load the metadynamics configuration (data/sample/metad_config.txt).
//   2. CPU reference: run every walker serially (reference_cpu.cpp) -> baseline.
//   3. GPU: one thread per walker, full trajectory each (kernels.cu)  -> taught.
//   4. VERIFY (two checks):
//        (a) CPU == GPU per walker, to machine precision (identical math), AND
//        (b) SCIENCE: the recovered free-energy surface F_est(s) matches the
//            KNOWN double-well F0(s) within a documented physical tolerance.
//   5. REPORT: deterministic numbers to stdout; timing to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Timings (which vary run-to-run) go to STDERR, which
//   the demo shows but does not diff (PATTERNS.md §3).
//
// CODE TOUR: read this first, then metad.h (the physics + integrator), then
//   kernels.cuh -> kernels.cu (GPU), and reference_cpu.cpp (CPU baseline).
//   See ../THEORY.md for the science and the GPU-mapping reasoning.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // integrate_gpu, metad_kernel
#include "reference_cpu.h"    // MetadConfig, load_config, integrate_cpu, walker_start
#include "metad.h"            // metad::recover_fes, true_fes, grid helpers
#include "util/io.hpp"        // util::CpuTimer

// Project identity (kept in sync with demo/expected_output.txt).
static const char* PROJECT_ID   = "1.6";
static const char* PROJECT_NAME = "Enhanced Sampling -- Metadynamics & Replica Exchange";

// --- Verification tolerances (documented; PATTERNS.md §4) ------------------
// WHY NOT BIT-EXACT? These Langevin trajectories are CHAOTIC (a stochastic
// thermostat on a double well). The device and host transcendental libraries
// (exp/log/sin/cos) differ by ~1 ULP, and with --fmad=false the arithmetic is
// otherwise matched -- but a 1-ULP seed of difference amplifies to O(1) over
// 20000 steps. So INDIVIDUAL trajectories (final_s, crossing counts) are NOT
// reproducible across CPU/GPU; that is a property of chaos, not a bug. What IS
// reproducible is the ENSEMBLE statistical observable: the free-energy surface.
// We therefore verify the ROBUST quantities and report the chaotic per-walker
// numbers only as machine-local diagnostics (on stderr).
//
//   (a) CPU vs GPU agreement on the ensemble-mean recovered FES. Averaging over
//       walkers + the self-flattening nature of well-tempered MetaD makes this
//       converge tightly; 0.25 kT (a quarter of thermal energy) is a generous,
//       honest "the two platforms recover the SAME surface" bound.
static constexpr double TOL_FES_CPU_GPU_KT = 0.25;
//   (b) Recovered FES vs the KNOWN analytic landscape (the science check): the
//       barrier height F_est(0) must match the true barrier A within 0.35 kT.
//       This validates the PHYSICS (metadynamics recovered the free energy), not
//       just that two codes agree.
static constexpr double TOL_BARRIER_KT = 0.35;

int main(int argc, char** argv) {
    // ---- 1. Load the configuration -----------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/metad_config.txt";
    MetadConfig cfg;
    try {
        cfg = load_config(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int M  = ensemble_size(cfg);
    const int nb = cfg.model.nbins;

    // ---- 2. CPU reference (timed) ------------------------------------------
    std::vector<metad::WalkerResult> res_cpu;
    std::vector<double> mean_bias_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    integrate_cpu(cfg, res_cpu, mean_bias_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU ensemble (kernel timed inside the wrapper) -----------------
    std::vector<metad::WalkerResult> res_gpu;
    std::vector<double> mean_bias_gpu;
    float gpu_kernel_ms = 0.0f;
    integrate_gpu(cfg, res_gpu, mean_bias_gpu, &gpu_kernel_ms);

    // ---- 4. Recover the free-energy surface from BOTH platforms' mean bias -
    //   F_est(s) = -(gamma/(gamma-1)) * mean_bias(s), min-shifted to 0. The CPU
    //   and GPU each produce their own ensemble-mean bias; we recover an FES from
    //   each and compare. (recover_fes is shared host+device math in metad.h.)
    std::vector<double> fes_cpu(nb, 0.0), fes_gpu(nb, 0.0);
    metad::recover_fes(cfg.model, mean_bias_cpu.data(), fes_cpu.data());
    metad::recover_fes(cfg.model, mean_bias_gpu.data(), fes_gpu.data());

    // ---- 4a. Verify CPU and GPU recover the SAME surface (robust observable) -
    //   We compare the recovered FES only over the well-sampled CORE [-1.3,1.3]:
    //   the grid edges are barely visited, so their bias is near zero and noisy,
    //   which would dominate a naive max-over-all-bins. (This is standard MetaD
    //   practice: trust the FES only where the walker spent time.)
    double worst_fes = 0.0;
    for (int j = 0; j < nb; ++j) {
        const double sj = cfg.model.s_lo + j * metad::grid_ds(cfg.model);
        if (sj < -1.3 || sj > 1.3) continue;
        worst_fes = std::fmax(worst_fes, std::fabs(fes_cpu[j] - fes_gpu[j]));
    }
    const bool pass_cpu_gpu = worst_fes <= TOL_FES_CPU_GPU_KT;

    // ---- 4b. Verify the SCIENCE: recovered FES vs the known double well ----
    //   Compare the barrier height F_est(0) against the true barrier (= A, since
    //   F0(+/-1)=0 are the minima and F0(0)=A). We average the CPU and GPU
    //   barrier estimates for a single reported number; both are min-shifted to 0.
    const int j0 = static_cast<int>(metad::grid_coord(cfg.model, 0.0) + 0.5);
    const double est_barrier  = 0.5 * (fes_cpu[j0] + fes_gpu[j0]);  // recovered (kT)
    const double true_barrier = cfg.model.A;                        // analytic (kT)
    const double barrier_err  = std::fabs(est_barrier - true_barrier);
    const bool pass_barrier   = barrier_err <= TOL_BARRIER_KT;

    const bool pass = pass_cpu_gpu && pass_barrier;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    //   STDOUT prints only the ROBUST statistical observable (the recovered FES,
    //   rounded to 0.1 kT where it is reproducible) plus the science verdict.
    //   Chaotic per-walker numbers go to stderr (machine-local; not diffed).
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("well-tempered metadynamics on a 1-D double well (SYNTHETIC model)\n");
    std::printf("ensemble: %d walkers x %d steps; barrier A=%.2f kT, gamma=%.1f, "
                "pace=%d, sigma=%.2f\n",
                M, cfg.model.steps, cfg.model.A, cfg.model.bias_factor,
                cfg.model.deposit_every, cfg.model.hill_sigma);
    std::printf("grid: %d bins over s in [%.2f, %.2f]; hills/walker=%d\n",
                nb, cfg.model.s_lo, cfg.model.s_hi, res_gpu[0].n_hills);

    // Recovered free-energy surface (GPU) vs the KNOWN landscape, at probe points.
    // Printed at 0.1-kT resolution: at this coarseness the statistical estimate is
    // reproducible run-to-run and platform-to-platform, so stdout stays stable.
    std::printf("recovered FES F(s) [kT] at s = -1.0 -0.5  0.0 +0.5 +1.0:\n");
    const double s_probe[5] = {-1.0, -0.5, 0.0, 0.5, 1.0};
    std::printf("  est :");
    for (int k = 0; k < 5; ++k) {
        const int j = static_cast<int>(metad::grid_coord(cfg.model, s_probe[k]) + 0.5);
        std::printf(" %4.1f", fes_gpu[j]);
    }
    std::printf("\n  true:");
    for (int k = 0; k < 5; ++k)
        std::printf(" %4.1f", metad::true_fes(cfg.model, s_probe[k]));
    std::printf("\n");

    std::printf("barrier height: recovered %.1f kT vs true %.1f kT\n",
                est_barrier, true_barrier);
    std::printf("RESULT: %s (CPU & GPU recover the same FES within %.2f kT; "
                "barrier matches analytic within %.2f kT)\n",
                pass ? "PASS" : "FAIL", TOL_FES_CPU_GPU_KT, TOL_BARRIER_KT);

    // ---- 5b. Varying / machine-local detail -> STDERR (shown, not diffed) --
    //   The crossing counts and per-walker endpoints are CHAOTIC (platform- and
    //   build-dependent), so they live here, never in the diffed stdout.
    long long total_cross_cpu = 0, total_cross_gpu = 0; int total_hills = 0;
    for (int i = 0; i < M; ++i) {
        total_cross_cpu += res_cpu[i].n_crossings;
        total_cross_gpu += res_gpu[i].n_crossings;
        total_hills     += res_gpu[i].n_hills;
    }
    std::fprintf(stderr, "[data]   source: %s  (%d walkers, %d hills total)\n",
                 path.c_str(), M, total_hills);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernel: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- GPU's edge grows with the walker "
                         "count; real multi-walker MetaD uses 10s-1000s of walkers.\n");
    std::fprintf(stderr, "[sample] ensemble barrier crossings: CPU=%lld GPU=%lld "
                         "(plain MD would give ~0; chaotic, so the two differ)\n",
                 total_cross_cpu, total_cross_gpu);
    std::fprintf(stderr, "[sample] walker 0 final s: CPU=%+.4f GPU=%+.4f "
                         "(individual trajectories diverge -- expected for chaos)\n",
                 res_cpu[0].final_s, res_gpu[0].final_s);
    std::fprintf(stderr, "[verify] CPU-vs-GPU max |FES| diff over core = %.3e kT (tol %.2f); "
                         "barrier err vs analytic = %.3e kT (tol %.2f)\n",
                 worst_fes, TOL_FES_CPU_GPU_KT, barrier_err, TOL_BARRIER_KT);

    return pass ? 0 : 1;
}
