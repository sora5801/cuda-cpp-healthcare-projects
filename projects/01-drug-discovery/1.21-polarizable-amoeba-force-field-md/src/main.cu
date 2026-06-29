// ===========================================================================
// src/main.cu  --  Entry point: solve AMOEBA induced-dipole ensemble, verify
// ---------------------------------------------------------------------------
// Project 1.21 : Polarizable / AMOEBA Force Field MD
//
// THE 5-STEP SHAPE (every project in this repo follows it)
//   1. Load the ensemble of polarization systems (from data/sample, or build a
//      built-in synthetic fallback).
//   2. CPU reference: solve every member's induced dipoles serially with CG.
//   3. GPU: one thread per member, the SAME CG solve each (kernels.cu).
//   4. VERIFY: per-member results agree (same CG -> same numbers to round-off).
//   5. REPORT: a deterministic table + ensemble summary to STDOUT; timing and
//      run-varying detail to STDERR (so demo/run_demo can diff stdout only).
//
// Code tour: start here, then amoeba.h (the physics + CG), kernels.cuh/.cu (the
// GPU mapping), reference_cpu.cpp (the serial baseline). See ../THEORY.md.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // solve_ensemble_gpu, EnsembleConfig, PerSystemResult
#include "reference_cpu.h"    // load_ensemble, make_synthetic_ensemble, integrate_cpu
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "1.21";
static const char* PROJECT_NAME = "Polarizable / AMOEBA Force Field MD";

// Verification tolerance. CPU and GPU run the IDENTICAL double-precision CG loop
// (amoeba.h, shared host+device), so the only difference is the GPU's fused
// multiply-add (FMA) contraction. Over a few dozen CG iterations that diverges by
// ~1e-12 at most; we verify the per-member energy and net dipole to 1e-9, which
// is far below any physically meaningful scale. (PATTERNS.md section 4.)
static constexpr double TOLERANCE = 1.0e-9;

// Pretty-print one member row deterministically. We print fixed-width, fixed-
// precision fields so the bytes never vary run-to-run.
static void print_member(int idx, const AtomSystem& s, const PerSystemResult& r) {
    // Distance between the two partner atoms (atoms 1 and 2 in the synthetic
    // geometry sit at +/- d on x), reported as the swept control variable.
    const double dx = s.pos[1][0] - s.pos[2][0];
    const double dy = s.pos[1][1] - s.pos[2][1];
    const double dz = s.pos[1][2] - s.pos[2][2];
    const double sep = std::sqrt(dx*dx + dy*dy + dz*dz) * 0.5;  // half-separation
    std::printf("  m%-4d n=%d sep=%.3f  iters=%2d  Upol=%+.6f  mu_x=%+.6f  max|mu|=%.6f\n",
                idx, s.n, sep, r.iters, r.upol, r.mu_total[0], r.max_mu);
}

int main(int argc, char** argv) {
    // ---- 1. Load (or synthesize) the ensemble ------------------------------
    EnsembleConfig c;
    const char* source = nullptr;
    if (argc > 1) {
        try {
            c = load_ensemble(argv[1]);
            source = argv[1];
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[error] %s\n", e.what());
            return 2;
        }
    } else {
        c = make_synthetic_ensemble(8);     // built-in fallback (clearly synthetic)
        source = "synthetic (built-in, 8 members)";
    }
    const int M = ensemble_size(c);

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<PerSystemResult> res_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    integrate_cpu(c, res_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU ensemble (kernel timed inside the wrapper) -----------------
    std::vector<PerSystemResult> res_gpu;
    float gpu_kernel_ms = 0.0f;
    solve_ensemble_gpu(c, res_gpu, &gpu_kernel_ms);

    // ---- 4. Verify: worst per-member disagreement on energy + net dipole ----
    double worst = 0.0;
    for (int i = 0; i < M; ++i) {
        worst = std::fmax(worst, std::fabs(res_cpu[i].upol - res_gpu[i].upol));
        for (int k = 0; k < 3; ++k)
            worst = std::fmax(worst,
                              std::fabs(res_cpu[i].mu_total[k] - res_gpu[i].mu_total[k]));
    }
    const bool pass = worst <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("AMOEBA induced-dipole SCF via matrix-free conjugate gradient\n");
    std::printf("ensemble: %d members, CG tol=%.1e, max_iter=%d\n", M, c.tol, c.max_iter);
    std::printf("per-member (idx, atoms, half-sep, CG iters, polarization energy, net dipole_x, peak |mu|):\n");
    for (int i = 0; i < M; ++i)
        print_member(i, c.systems[i], res_gpu[i]);

    // Ensemble summary: a single deterministic line capturing the trend. As the
    // partner atoms approach (later members), coupling strengthens -> the
    // polarization energy becomes more negative (more stabilizing). We report the
    // strongest (most negative) Upol and the largest induced dipole seen.
    double min_upol = res_gpu[0].upol, max_mu = res_gpu[0].max_mu;
    int    total_iters = 0;
    for (int i = 0; i < M; ++i) {
        if (res_gpu[i].upol < min_upol) min_upol = res_gpu[i].upol;
        if (res_gpu[i].max_mu > max_mu) max_mu  = res_gpu[i].max_mu;
        total_iters += res_gpu[i].iters;
    }
    std::printf("summary: strongest Upol=%+.6f  largest |mu|=%.6f  total CG iters=%d\n",
                min_upol, max_mu, total_iters);
    std::printf("RESULT: %s (GPU ensemble matches CPU within tol=%.1e)\n",
                pass ? "PASS" : "FAIL", TOLERANCE);

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%d members)\n", source, M);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernel: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- tiny ensembles are launch/copy "
                         "bound; the GPU's edge grows with member count (real MD: 10^5+ atoms).\n");
    std::fprintf(stderr, "[verify] worst per-member diff = %.3e  (tolerance %.1e)\n", worst, TOLERANCE);

    return pass ? 0 : 1;
}
