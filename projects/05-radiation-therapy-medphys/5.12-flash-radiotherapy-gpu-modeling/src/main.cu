// ===========================================================================
// src/main.cu  --  Entry point: integrate the FLASH ensemble, verify, report
// ---------------------------------------------------------------------------
// Project 5.12 : FLASH Radiotherapy GPU Modeling
//
// WHAT THIS FILE DOES  (the 5-step shape every project in this repo follows)
//   1. Load the ensemble config (a pO2 sweep x {conventional, FLASH} delivery).
//   2. CPU reference: integrate every member serially (reference_cpu.cpp).
//   3. GPU: one thread per member, full pulse-train RK4 each (kernels.cu).
//   4. VERIFY: per-member results match (shared integrate_voxel -> same numbers).
//   5. REPORT: deterministic FLASH-vs-conventional table + summary to stdout;
//      timing and run-varying detail to stderr.
//
//   STDOUT is kept byte-for-byte deterministic (fixed-width, fixed-precision) so
//   demo/run_demo can diff it against demo/expected_output.txt. Timings (which
//   vary run to run) go to STDERR, which the demo shows but does not diff.
//
//   THE TEACHING PAYOFF: the printed table shows, per oxygen level, the oxygen-
//   fixed damage under conventional vs FLASH delivery and the resulting SPARING
//   factor. FLASH spares oxygenated (normal-tissue) voxels the most and hypoxic
//   (tumour-core) voxels the least -- the qualitative FLASH signature, emergent
//   from the shared ODE, not hard-coded. (See ../THEORY.md; educational model.)
//
// Code tour: start here, then flash.h (the ODE core), kernels.cu (GPU twin),
// reference_cpu.cpp (baseline). See ../THEORY.md for the "why".
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // integrate_gpu, EnsembleConfig, VoxelResult
#include "reference_cpu.h"    // load_ensemble, integrate_cpu, member_job, member_axes
#include "util/io.hpp"        // util::CpuTimer

// Program identity (kept in sync with demo/expected_output.txt).
static const char* PROJECT_ID   = "5.12";
static const char* PROJECT_NAME = "FLASH Radiotherapy GPU Modeling";

// Correctness tolerance. Both sides run the SAME double-precision RK4 via
// flash.h (integrate_voxel), so the only differences are the GPU's fused
// multiply-add rounding vs the host compiler's. Over these short integrations
// that stays deep below 1e-9; we verify to 1e-9 and say so honestly.
static constexpr double TOLERANCE = 1.0e-9;

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/flash_ensemble.txt";
    EnsembleConfig c;
    try {
        c = load_ensemble(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int M = ensemble_size(c);

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<VoxelResult> res_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    integrate_cpu(c, res_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU ensemble (kernel timed) -----------------------------------
    std::vector<VoxelResult> res_gpu;
    float gpu_kernel_ms = 0.0f;
    integrate_gpu(c, res_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    // Worst absolute difference across every scalar of every member.
    double worst = 0.0;
    for (int i = 0; i < M; ++i) {
        worst = std::fmax(worst, std::fabs(res_cpu[i].fixed_damage - res_gpu[i].fixed_damage));
        worst = std::fmax(worst, std::fabs(res_cpu[i].min_O2       - res_gpu[i].min_O2));
        worst = std::fmax(worst, std::fabs(res_cpu[i].eff_O2       - res_gpu[i].eff_O2));
    }
    const bool pass = worst <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    // Header: what the ensemble is.
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("[educational reduced-scope model -- not for clinical use]\n");
    std::printf("ensemble: %d pO2 levels x 2 delivery modes = %d members\n", c.n_po2, M);
    std::printf("dose = %.1f Gy in %d pulses; conv gap = %.5f s, FLASH gap = %.5f s\n",
                c.total_dose, c.n_pulses,
                c.conv_steps_per_gap * c.dt, c.flash_steps_per_gap * c.dt);

    // The comparison table: for each pO2, show conventional vs FLASH oxygen-fixed
    // damage, the minimum O2 reached under FLASH (the depletion depth), and the
    // sparing factor = conv_damage / flash_damage (>1 means FLASH did less damage).
    std::printf("pO2[mmHg]  conv_damage  flash_damage  flash_minO2[uM]  sparing\n");
    double sum_sparing = 0.0; int counted = 0;
    for (int p = 0; p < c.n_po2; ++p) {
        const int idx_conv  = p * N_MODES + MODE_CONVENTIONAL;
        const int idx_flash = p * N_MODES + MODE_FLASH;
        const VoxelJob jc = member_job(c, idx_conv);      // to read this level's pO2
        const double dconv  = res_gpu[idx_conv ].fixed_damage;
        const double dflash = res_gpu[idx_flash].fixed_damage;
        const double sparing = (dflash > 0.0) ? dconv / dflash : 0.0;
        sum_sparing += sparing; ++counted;
        std::printf("%9.2f  %11.5f  %12.5f  %15.5f  %7.4f\n",
                    jc.po2_mmHg, dconv, dflash, res_gpu[idx_flash].min_O2, sparing);
    }

    // Summary line: mean sparing, and where sparing is largest (most oxygenated).
    const double mean_sparing = counted ? sum_sparing / counted : 0.0;
    std::printf("mean FLASH sparing factor = %.4f (conv damage / FLASH damage)\n", mean_sparing);
    std::printf("RESULT: %s (GPU ensemble matches CPU within tol=1.0e-09)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d members)\n", path.c_str(), M);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernel: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- a real FLASH map sweeps millions of "
                         "voxels; the GPU's edge grows with member count.\n");
    std::fprintf(stderr, "[verify] worst per-member diff = %.3e  (tolerance %.1e)\n", worst, TOLERANCE);

    return pass ? 0 : 1;
}
