// ===========================================================================
// src/main.cu  --  Entry point: reconstruct a super-resolution image, verify
// ---------------------------------------------------------------------------
// Project 4.10 : Super-Resolution Microscopy Reconstruction  (STORM / PALM SMLM)
//
// 5-step shape (the shape every project in this repo follows):
//   1. Load the raw SMLM movie (data/sample, or the path in argv[1]).
//   2. CPU reference: detect + localize + render      (reference_cpu.cpp).
//   3. GPU pipeline:  detect + localize + atomic render (kernels.cu).
//   4. VERIFY: the localization list, the fixed-point image, and the summary all
//      match the CPU exactly (integer/render fields) or within a tiny tolerance
//      (the double-precision mean statistics). See THEORY §6.
//   5. REPORT: a DETERMINISTIC digest to stdout (diffed by the demo); timing and
//      run-varying detail to stderr (shown, not diffed).
//
// Code tour: start here, then smlm.h (the fit), reference_cpu.h/.cpp (baseline +
// shared render), kernels.cuh/.cu (the GPU twin). See ../THEORY.md for the "why".
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // smlm_gpu, FrameStack, Localization, ResultSummary
#include "reference_cpu.h"    // load_stack, detect_and_localize_cpu, render_image, summarize
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "4.10";
static const char* PROJECT_NAME = "Super-Resolution Microscopy Reconstruction";

// Verification tolerance for the double-precision MEAN statistics only. The
// integer fields (localization count, image checksum, bright-bin count) must be
// EXACT (== 0 difference): the fit is the same fixed-iteration double arithmetic
// on both sides and the render sums fixed-point integers, so those cannot drift.
// The means are averages of thousands of doubles that were computed in the same
// order on both sides; 1e-6 is generous slack for last-ULP differences. (See
// docs/PATTERNS.md §4 and THEORY §6.)
static constexpr double MEAN_TOL = 1.0e-6;

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1]
                                        : "data/sample/smlm_stack.txt";
    FrameStack stack;
    try {
        stack = load_stack(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<Localization> locs_cpu;
    std::vector<unsigned long long> img_cpu;
    int srH_cpu = 0, srW_cpu = 0;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    detect_and_localize_cpu(stack, locs_cpu);
    render_image(stack, locs_cpu, img_cpu, srH_cpu, srW_cpu);
    const double cpu_ms = cpu_timer.stop_ms();
    const ResultSummary S_cpu = summarize(locs_cpu, img_cpu, srH_cpu, srW_cpu);

    // ---- 3. GPU pipeline (kernels timed inside the wrapper) ---------------
    std::vector<Localization> locs_gpu;
    std::vector<unsigned long long> img_gpu;
    int srH_gpu = 0, srW_gpu = 0;
    float gpu_kernel_ms = 0.0f;
    const ResultSummary S_gpu =
        smlm_gpu(stack, locs_gpu, img_gpu, srH_gpu, srW_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    // (a) Same number of localizations, in the same canonical order.
    const bool count_ok = (S_cpu.n_localizations == S_gpu.n_localizations);
    // (b) The fixed-point render is bit-identical (exact integer checksum + the
    //     same number of illuminated bins).
    const bool image_ok = count_ok
        && (S_cpu.img_checksum == S_gpu.img_checksum)
        && (S_cpu.bright_bins  == S_gpu.bright_bins)
        && (srH_cpu == srH_gpu) && (srW_cpu == srW_gpu);
    // (c) Also confirm every fixed-point pixel matches (not just the checksum) --
    //     a stronger, still-exact check that no two bins swapped contents.
    bool pixels_ok = image_ok && (img_cpu.size() == img_gpu.size());
    if (pixels_ok)
        for (std::size_t i = 0; i < img_cpu.size(); ++i)
            if (img_cpu[i] != img_gpu[i]) { pixels_ok = false; break; }
    // (d) The double-precision mean statistics agree within MEAN_TOL.
    const double dmx = std::fabs(S_cpu.mean_x       - S_gpu.mean_x);
    const double dmy = std::fabs(S_cpu.mean_y       - S_gpu.mean_y);
    const double dms = std::fabs(S_cpu.mean_sigma   - S_gpu.mean_sigma);
    const double dmp = std::fabs(S_cpu.mean_photons - S_gpu.mean_photons);
    const double mean_err = std::fmax(std::fmax(dmx, dmy), std::fmax(dms, dmp));
    const bool means_ok = mean_err <= MEAN_TOL;

    const bool pass = count_ok && image_ok && pixels_ok && means_ok;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) ----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("STORM/PALM SMLM: %d frames of %dx%d px\n", stack.F, stack.H, stack.W);
    std::printf("localizations: %zu emitters fitted (7x7 patch, %d refine iters)\n",
                S_gpu.n_localizations, FIT_ITERS);
    std::printf("super-resolution image: %dx%d px (%dx upsampled), %zu illuminated bins\n",
                S_gpu.srH, S_gpu.srW, UPSAMPLE, S_gpu.bright_bins);
    std::printf("image checksum (fixed-point sum): %llu\n", S_gpu.img_checksum);
    // Mean statistics rounded to 4 decimals: deterministic and stable well
    // within MEAN_TOL, so they are safe to diff byte-for-byte.
    std::printf("mean position: x=%.4f y=%.4f px\n", S_gpu.mean_x, S_gpu.mean_y);
    std::printf("mean PSF sigma: %.4f px   mean intensity: %.4f\n",
                S_gpu.mean_sigma, S_gpu.mean_photons);
    std::printf("RESULT: %s (GPU localizations+image match CPU)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) -----------------
    std::fprintf(stderr, "[data]   source: %s  (%d frames, %dx%d, bg=%.2f, thr=%.2f)\n",
                 path.c_str(), stack.F, stack.H, stack.W, stack.background, stack.threshold);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernels: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- the tiny sample is launch-bound; "
                         "the GPU's edge grows with the 10^4-10^5 frames of a real STORM run.\n");
    std::fprintf(stderr, "[verify] count(cpu/gpu)=%zu/%zu  checksum(cpu/gpu)=%llu/%llu  "
                         "bright_bins(cpu/gpu)=%zu/%zu  pixels_exact=%s  mean_err=%.3e (tol %.1e)\n",
                 S_cpu.n_localizations, S_gpu.n_localizations,
                 S_cpu.img_checksum, S_gpu.img_checksum,
                 S_cpu.bright_bins, S_gpu.bright_bins,
                 pixels_ok ? "yes" : "NO", mean_err, MEAN_TOL);

    return pass ? 0 : 1;
}
