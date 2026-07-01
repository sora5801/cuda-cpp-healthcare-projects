// ===========================================================================
// src/main.cu  --  Entry point: load image, denoise on CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 4.9 : Image Denoising & Restoration  (Non-Local Means)
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the denoising problem (noisy image + synthetic clean ground truth +
//      NLM parameters) from data/sample, or a built-in fallback if none given.
//   2. Denoise on the CPU (reference_cpu.cpp)  -> the trusted reference.
//   3. Denoise on the GPU (kernels.cu)         -> the thing being taught.
//   4. VERIFY: assert the GPU image agrees with the CPU image within tolerance.
//   5. REPORT: deterministic results (PSNR, fixed pixel samples) to STDOUT;
//      timing + run-varying detail to STDERR.
//
//   STDOUT is kept byte-for-byte deterministic so demo/run_demo can diff it
//   against demo/expected_output.txt. The reported pixel samples and PSNR are
//   taken from the CPU reference (a fixed, order-deterministic computation), and
//   both CPU and GPU run the SAME nlm_pixel() math, so the numbers are stable
//   across machines. Timings (which vary) go to STDERR, shown but not diffed.
//
// READ THIS FIRST in the code tour, then nlm_core.h (the math) ->
// kernels.cuh -> kernels.cu (GPU), and reference_cpu.cpp (baseline). See
// ../THEORY.md for the derivation and the GPU mapping.
// ===========================================================================
#include <cmath>      // std::sin (built-in fallback phantom's deterministic ripple)
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // denoise_gpu (GPU path), Image, NlmParams
#include "reference_cpu.h"    // load_problem, denoise_cpu, psnr
#include "util/io.hpp"        // util::CpuTimer, util::max_abs_err

// Program identity, printed on the first stdout line.
static const char* PROJECT_ID   = "4.9";
static const char* PROJECT_NAME = "Image Denoising & Restoration (Non-Local Means)";

// ---------------------------------------------------------------------------
// Correctness tolerance for the GPU-vs-CPU check.
//   Both sides run the IDENTICAL nlm_pixel() arithmetic (shared nlm_core.h), so
//   in exact real arithmetic they would be bit-identical. In practice the GPU
//   may CONTRACT a multiply-add into a single fused FMA where the host compiler
//   emits a separate multiply then add; over the thousands of accumulations per
//   pixel (patch distances + weighted sums) that rounding difference reaches a
//   few 1e-6 in float. So we verify to a small ABSOLUTE tolerance and say so
//   honestly (PATTERNS.md §4) rather than pretending the results are bitwise
//   equal. 1e-4 intensity units is ~0.03 of one 8-bit grey level -- invisible.
// ---------------------------------------------------------------------------
static constexpr double TOLERANCE = 1.0e-4;

// ---------------------------------------------------------------------------
// make_builtin_problem: a tiny synthetic fallback used only if no sample file is
//   given AND the default sample is missing. It is a 12x12 two-level phantom
//   (a bright square on a dark background) plus a fixed, deterministic pseudo-
//   noise pattern -- NOT random, so the built-in path is reproducible too. The
//   committed data/sample file (from scripts/make_synthetic.py) is the primary
//   input; this exists so the program never hard-fails.
// ---------------------------------------------------------------------------
static DenoiseProblem make_builtin_problem() {
    DenoiseProblem prob;
    NlmParams& p = prob.params;
    p.width = 12; p.height = 12;
    p.patch_radius = 1; p.search_radius = 3;
    p.sigma = 0.10f; p.h = 0.12f;

    prob.clean.width = p.width; prob.clean.height = p.height;
    prob.noisy.width = p.width; prob.noisy.height = p.height;
    prob.clean.pix.resize(prob.clean.size());
    prob.noisy.pix.resize(prob.noisy.size());

    for (int r = 0; r < p.height; ++r) {
        for (int c = 0; c < p.width; ++c) {
            // Clean phantom: a centred bright block (0.8) on a dark field (0.2).
            const bool inside = (r >= 3 && r < 9 && c >= 3 && c < 9);
            const float truth = inside ? 0.8f : 0.2f;
            const std::size_t idx = (std::size_t)r * p.width + c;
            prob.clean.pix[idx] = truth;
            // Deterministic "noise": a fixed sinusoidal ripple (no RNG) so the
            // built-in problem is byte-reproducible. Amplitude ~sigma.
            const float ripple = 0.10f * std::sin(0.9f * r + 1.7f * c);
            float v = truth + ripple;
            if (v < 0.0f) v = 0.0f; if (v > 1.0f) v = 1.0f;  // keep in [0,1]
            prob.noisy.pix[idx] = v;
        }
    }
    return prob;
}

int main(int argc, char** argv) {
    // ---- 1. Load the problem ------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/phantom_sample.txt";
    DenoiseProblem prob;
    std::string source = path;
    try {
        prob = load_problem(path);
    } catch (const std::exception& e) {
        // Fall back to the built-in phantom so the program still demonstrates
        // NLM even with no data file present. We note this on stderr.
        std::fprintf(stderr, "[warn] could not load '%s' (%s); using built-in phantom.\n",
                     path.c_str(), e.what());
        prob = make_builtin_problem();
        source = "built-in synthetic phantom";
    }
    const NlmParams& p = prob.params;

    // ---- 2. CPU reference denoise (timed) ----------------------------------
    Image den_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    denoise_cpu(prob.noisy, p, den_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU denoise (kernel timed inside the wrapper) ------------------
    Image den_gpu;
    float gpu_kernel_ms = 0.0f;
    denoise_gpu(prob.noisy, p, den_gpu, &gpu_kernel_ms);

    // ---- 4. Verify GPU == CPU ----------------------------------------------
    const double err = util::max_abs_err(den_cpu.pix, den_gpu.pix);
    const bool pass = err <= TOLERANCE;

    // ---- Quality metrics (from the CPU reference -> deterministic) ---------
    // PSNR of the NOISY input and of the DENOISED output, both vs the clean
    // ground truth. A working denoiser RAISES PSNR. These are computed from the
    // CPU result so stdout never depends on GPU float ordering.
    const double psnr_noisy = psnr(prob.noisy, prob.clean);
    const double psnr_den   = psnr(den_cpu,    prob.clean);

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    const int N = p.width;
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("Non-Local Means denoise: %dx%d image, patch r=%d, search r=%d, sigma=%.3f, h=%.3f\n",
                p.width, p.height, p.patch_radius, p.search_radius, p.sigma, p.h);
    std::printf("PSNR noisy  vs clean = %.4f dB\n", psnr_noisy);
    std::printf("PSNR denoised vs clean = %.4f dB\n", psnr_den);
    std::printf("PSNR improvement = %.4f dB\n", psnr_den - psnr_noisy);
    // A fixed row profile through the middle of the image: 8 evenly spaced
    // columns of the denoised central row. These exact values are what
    // expected_output.txt captures.
    const int midrow = p.height / 2;
    std::printf("denoised central row (8 samples):");
    for (int s = 0; s < 8; ++s) {
        const int col = (s * (N - 1)) / 7;      // 8 evenly spaced columns 0..N-1
        std::printf(" %.4f", den_cpu.pix[(std::size_t)midrow * N + col]);
    }
    std::printf("\n");
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-04)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%dx%d, sigma=%.3f)\n",
                 source.c_str(), p.width, p.height, p.sigma);
    std::fprintf(stderr, "[timing] CPU denoise: %.3f ms   GPU denoise: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- NLM's O(P*S^2*R^2) cost makes the "
                         "GPU's edge grow fast with image size; a tiny sample is launch-bound.\n");
    std::fprintf(stderr, "[verify] max_abs_err = %.3e  (tolerance %.1e)\n", err, TOLERANCE);

    // Exit code feeds the demo's pass/fail gate.
    return pass ? 0 : 1;
}
