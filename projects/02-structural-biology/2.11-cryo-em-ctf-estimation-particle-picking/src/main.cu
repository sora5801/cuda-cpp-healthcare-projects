// ===========================================================================
// src/main.cu  --  Entry point: load micrograph, fit CTF on CPU+GPU, verify
// ---------------------------------------------------------------------------
// Project 2.11 : Cryo-EM CTF Estimation & Particle Picking
//
// 5-step shape (the repo's standard demo skeleton):
//   1. Load the synthetic micrograph (data/sample).
//   2. CPU reference: naive 2-D DFT power spectrum -> radial profile -> defocus
//      grid-search (reference_cpu.cpp).
//   3. GPU: cuFFT 2-D FFT -> radial average -> grid-search (kernels.cu).
//   4. VERIFY: the GPU and CPU recover the SAME best defocus index (exact) and
//      their NCC score curves agree within a tight tolerance.
//   5. REPORT: deterministic recovered defocus + score to stdout; timings to stderr.
//
// Code tour: start here, then ctf_model.h (the physics), reference_cpu.h/.cpp (the
// trusted baseline), then kernels.cuh -> kernels.cu (the cuFFT call + kernels).
// The science / GPU-mapping is in ../THEORY.md.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "kernels.cuh"        // radial_power_profile_gpu, fit_ctf_gpu
#include "reference_cpu.h"    // load_micrograph, *_cpu, flatten_background, configs
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "2.11";
static const char* PROJECT_NAME = "Cryo-EM CTF Estimation (cuFFT defocus fit)";

// HONEST TOLERANCES (PATTERNS.md §4). The CPU profile comes from a DOUBLE-precision
// naive DFT; the GPU profile from cuFFT's SINGLE-precision FFT. So the two radial
// profiles genuinely differ at the ~1% level, and that difference propagates into
// the NCC scores -- most visibly for POORLY-fitting candidates, whose near-zero
// correlation is sensitive to tiny profile wiggles. We therefore verify the things
// that actually matter for a CTF estimate, each to its appropriate tolerance:
//   (a) EXACT match of the recovered best-defocus INDEX  -- the headline answer.
//   (b) The NCC at the WINNING candidate agrees to BEST_TOL (the fit quality at the
//       answer is robust: both transforms see the same strong ring signal there).
//   (c) The FULL score curve agrees to CURVE_TOL, a documented single-vs-double-FFT
//       tolerance -- NOT machine precision, and we say so rather than pretend.
// This is the "be honest about floating point" rule: do not claim bit-identical
// results across two different-precision FFTs.
static constexpr double BEST_TOL  = 5.0e-3;  // NCC at the recovered defocus
static constexpr double CURVE_TOL = 5.0e-2;  // worst NCC over the whole search grid

int main(int argc, char** argv) {
    // ---- 1. Load ----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1]
                                        : "data/sample/micrograph_sample.txt";
    Micrograph m;
    try {
        m = load_micrograph(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- Search configuration (CPU and GPU share it byte-for-byte) --------
    // Defocus grid: scan a physically sensible under-focus range for cryo-EM
    // (~0.5 um to ~3.0 um). 1 um = 10000 A. n_dz candidates => the grid step.
    CtfFitConfig cfg;
    cfg.dz_min = 5000.0;     // 0.5 um (A)
    cfg.dz_max = 30000.0;    // 3.0 um (A)
    cfg.n_dz   = 251;        // 100 A spacing -> fine enough to resolve the rings
    cfg.nbins  = m.n / 2;    // radial bins: DC..Nyquist
    cfg.r_lo   = 4;          // skip the DC spike / lowest few bins
    cfg.r_hi   = cfg.nbins;  // up to Nyquist

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<double> prof_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    radial_power_profile_cpu(m, cfg.nbins, prof_cpu);          // stages 1+2
    const CtfFitResult cpu = fit_ctf_cpu(prof_cpu, m.optics, cfg); // stage 3
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU path (timed via CUDA events inside the wrappers) ----------
    std::vector<double> raw_gpu, prof_gpu;
    float gpu_spec_ms = 0.0f, gpu_fit_ms = 0.0f;
    radial_power_profile_gpu(m, cfg.nbins, raw_gpu, &gpu_spec_ms);
    // Apply the SAME background flattening the CPU used (shared function) so the
    // two profiles are post-processed identically.
    flatten_background(raw_gpu, /*win=*/4, prof_gpu);
    const CtfFitResult gpu = fit_ctf_gpu(prof_gpu, m.optics, cfg, &gpu_fit_ms);

    // ---- 4. Verify --------------------------------------------------------
    const bool idx_match = (cpu.best_idx == gpu.best_idx);
    // (b) NCC agreement AT the winning candidate (the fit quality at the answer).
    double best_diff = std::numeric_limits<double>::infinity();
    if (idx_match && cpu.best_idx >= 0)
        best_diff = std::fabs(cpu.scores[cpu.best_idx] - gpu.scores[gpu.best_idx]);
    // (c) Worst NCC difference over the WHOLE grid (single-vs-double-FFT bound).
    double worst_score = 0.0;
    for (int i = 0; i < cfg.n_dz; ++i) {
        const double d = std::fabs(cpu.scores[i] - gpu.scores[i]);
        if (d > worst_score) worst_score = d;
    }
    const bool best_close  = (best_diff  <= BEST_TOL);
    const bool curve_close = (worst_score <= CURVE_TOL);
    const bool pass = idx_match && best_close && curve_close;

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("micrograph: %dx%d px, pixel=%.2f A, lambda=%.4f A, Cs=%.3e A, ac=%.2f\n",
                m.n, m.n, m.optics.pixel_size, m.optics.lambda, m.optics.cs,
                m.optics.amp_contrast);
    std::printf("defocus search: %.0f..%.0f A over %d candidates (step %.0f A)\n",
                cfg.dz_min, cfg.dz_max, cfg.n_dz,
                (cfg.dz_max - cfg.dz_min) / (cfg.n_dz - 1));
    std::printf("CPU best: dz = %.1f A  (idx %d, NCC %.6f)\n",
                cpu.best_dz, cpu.best_idx, cpu.scores[cpu.best_idx]);
    std::printf("GPU best: dz = %.1f A  (idx %d, NCC %.6f)\n",
                gpu.best_dz, gpu.best_idx, gpu.scores[gpu.best_idx]);
    if (m.true_dz > 0.0) {
        // Synthetic data: report recovery error against the embedded ground truth.
        std::printf("true defocus (synthetic): %.1f A  -> GPU error %.1f A\n",
                    m.true_dz, gpu.best_dz - m.true_dz);
    }
    std::printf("RESULT: %s (GPU and CPU agree on defocus index; NCC at best within %.0e)\n",
                pass ? "PASS" : "FAIL", BEST_TOL);

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%dx%d micrograph)\n", path.c_str(), m.n, m.n);
    std::fprintf(stderr, "[timing] CPU naive-DFT+fit: %.3f ms   GPU cuFFT+radial: %.3f ms   GPU search: %.3f ms\n",
                 cpu_ms, gpu_spec_ms, gpu_fit_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- the naive 2-D DFT is O(N^4); cuFFT is "
                         "O(N^2 log N). The gap explodes with micrograph size.\n");
    std::fprintf(stderr, "[verify] best-index match=%s  |NCC diff at best| = %.3e (tol %.1e)  "
                         "worst |NCC diff| over grid = %.3e (tol %.1e)\n",
                 idx_match ? "yes" : "NO", best_diff, BEST_TOL, worst_score, CURVE_TOL);

    return pass ? 0 : 1;
}
