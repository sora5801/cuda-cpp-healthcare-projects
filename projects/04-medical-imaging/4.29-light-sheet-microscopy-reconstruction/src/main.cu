// ===========================================================================
// src/main.cu  --  Entry point: load plane, run CPU + GPU RL, verify, report
// ---------------------------------------------------------------------------
// Project 4.29 : Light-Sheet Microscopy Reconstruction
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the measured (blurry, noisy) plane from data/sample.
//   2. CPU reference: Richardson-Lucy deconvolution via a direct DFT (trusted).
//   3. GPU: the SAME RL deconvolution via cuFFT (kernels.cu) -- the thing taught.
//   4. VERIFY: GPU estimate agrees with the CPU estimate within a documented,
//      floating-point-honest tolerance (PATTERNS.md 4: iterative FFT solver).
//   5. REPORT: deterministic reconstruction statistics to stdout; timing/detail
//      to stderr. stdout is byte-stable so demo/run_demo can diff it.
//
// A NOTE ON THE STATISTIC WE PRINT
//   We reduce each image to (sum, max, L2) -- three order-independent numbers --
//   and print them at fixed precision. We ALSO print how much the reconstruction
//   sharpened the blurry input (peak and L2 ratios): a real, interpretable
//   outcome of deconvolution. The committed sample embeds a known ground truth,
//   so "did RL sharpen it back up?" is a meaningful, verifiable question.
//
//   STDOUT is byte-for-byte deterministic (diffed by the demo); anything that
//   varies run-to-run (timings) goes to STDERR (shown, not diffed).
//
// Code tour: start here, then kernels.cuh -> kernels.cu (the cuFFT loop), then
// reference_cpu.cpp and the shared rl_core.h. The "why" is in ../THEORY.md.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // deconvolve_gpu (GPU path), LsfmData
#include "reference_cpu.h"    // load_lsfm, deconvolve_cpu, image_stats
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "4.29";
static const char* PROJECT_NAME = "Light-Sheet Microscopy Reconstruction";

// -----------------------------------------------------------------------------
// VERIFICATION TOLERANCE (honest about floating point -- PATTERNS.md 4).
//   Both paths run in DOUBLE precision and share the per-pixel RL math (rl_core.h)
//   and the same PSF, so the ONLY difference is that the GPU convolves with the
//   cuFFT FFT while the CPU convolves with a direct DFT. Those two are the same
//   transform but accumulate round-off differently; over `iters` multiplicative
//   RL steps that difference grows slightly. We therefore verify to a small
//   RELATIVE tolerance on the summary statistics rather than bit-equality, and
//   say so plainly. (THEORY.md "Numerical considerations" derives this.)
//
//   In practice the observed error on the committed sample is ~1e-15 (both paths
//   are double precision and share the PSF + rl_core.h), so 1e-9 is a comfortable,
//   honest floor -- roughly a million times looser than we actually achieve, yet
//   far tighter than a single-precision path (R2C/C2R) could ever hit.
// -----------------------------------------------------------------------------
static constexpr double REL_TOL = 1.0e-9;   // relative tolerance on sum/max/L2

// Relative difference |x-y| / (|x| + tiny), guarding against divide-by-zero.
static double rel_diff(double x, double y) {
    return std::fabs(x - y) / (std::fabs(x) + 1e-30);
}

int main(int argc, char** argv) {
    // ---- 1. Load the measured plane ----------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/lsfm_sample.txt";
    LsfmData d;
    try {
        d = load_lsfm(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // Statistics of the INPUT (blurry) image, so we can report the sharpening.
    double in_sum = 0.0, in_max = 0.0, in_l2 = 0.0;
    image_stats(d.measured, in_sum, in_max, in_l2);

    // ---- 2. CPU reference: RL via direct DFT (timed) -----------------------
    std::vector<double> est_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    deconvolve_cpu(d, est_cpu);
    const double cpu_ms = cpu_timer.stop_ms();
    double cpu_sum = 0.0, cpu_max = 0.0, cpu_l2 = 0.0;
    image_stats(est_cpu, cpu_sum, cpu_max, cpu_l2);

    // ---- 3. GPU: RL via cuFFT (kernel time measured inside the wrapper) ----
    std::vector<double> est_gpu;
    float gpu_ms = 0.0f;
    deconvolve_gpu(d, est_gpu, &gpu_ms);
    double gpu_sum = 0.0, gpu_max = 0.0, gpu_l2 = 0.0;
    image_stats(est_gpu, gpu_sum, gpu_max, gpu_l2);

    // ---- 4. Verify (GPU vs CPU summary statistics agree) -------------------
    const double d_sum = rel_diff(cpu_sum, gpu_sum);
    const double d_max = rel_diff(cpu_max, gpu_max);
    const double d_l2  = rel_diff(cpu_l2,  gpu_l2);
    double worst = d_sum;
    if (d_max > worst) worst = d_max;
    if (d_l2  > worst) worst = d_l2;
    const bool pass = worst <= REL_TOL;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    // We print the CPU (reference) statistics so the value is transform-independent
    // and stable; the GPU is validated against them just above.
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("Richardson-Lucy deconvolution (cuFFT, Fourier domain): %dx%d image, "
                "PSF sigma=%.2f px, %d iterations\n", d.H, d.W, d.sigma, d.iters);
    std::printf("input  (blurry)   : sum=%.4f  max=%.6f  L2=%.6f\n", in_sum, in_max, in_l2);
    std::printf("output (deblurred): sum=%.4f  max=%.6f  L2=%.6f\n", cpu_sum, cpu_max, cpu_l2);
    // Sharpening ratios: RL conserves total flux (sum ~ unchanged) while pushing
    // energy back into peaks (max, L2 rise). These ratios are the headline result.
    std::printf("sharpening        : peak x%.4f  L2 x%.4f  (flux ratio %.4f)\n",
                in_max > 0 ? cpu_max / in_max : 0.0,
                in_l2  > 0 ? cpu_l2  / in_l2  : 0.0,
                in_sum != 0 ? cpu_sum / in_sum : 0.0);
    std::printf("RESULT: %s (GPU cuFFT matches CPU DFT within rel tol=1.0e-09)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%d x %d, sigma=%.2f, iters=%d)\n",
                 path.c_str(), d.H, d.W, d.sigma, d.iters);
    std::fprintf(stderr, "[timing] CPU direct-DFT RL: %.3f ms   GPU cuFFT RL: %.3f ms\n",
                 cpu_ms, gpu_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- the CPU here is an O((H*W)^2)-per-"
                         "convolution DFT; the GPU's cuFFT is O(N log N). The gap explodes with "
                         "image size (real LSFM planes are 2048^2+).\n");
    std::fprintf(stderr, "[verify] worst relative stat error (GPU vs CPU) = %.3e  (tol %.1e)\n",
                 worst, REL_TOL);

    return pass ? 0 : 1;
}
