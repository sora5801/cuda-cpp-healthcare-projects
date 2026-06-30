// ===========================================================================
// src/main.cu  --  Entry point: load image, RL-deconvolve (CPU+GPU), verify
// ---------------------------------------------------------------------------
// Project 4.30 : Deconvolution Microscopy
//
// 5-step shape (the repo's standard main):
//   1. Load the blurry observed image (data/sample).
//   2. CPU reference: Richardson-Lucy with direct circular convolution.
//   3. GPU: Richardson-Lucy with cuFFT convolution (kernels.cu).
//   4. VERIFY: GPU deconvolved image agrees with CPU within a documented
//      tolerance (RL is a long iterative double-precision solver; FFT vs direct
//      convolution + FMA reordering diverge by a tiny, physically-negligible
//      amount -- PATTERNS.md section 4).
//   5. REPORT: deterministic per-image scalars (sharpness before/after, a small
//      sample of pixels) to stdout; timings + worst error to stderr.
//
// The PSF used for deconvolution MUST match the PSF the synthetic generator used
// to BLUR the image, or RL would be deconvolving with the wrong kernel. Those
// parameters live in the constants below and in scripts/make_synthetic.py --
// keep them in sync (data/README.md documents the contract).
//
// Code tour: start here, then kernels.cuh -> kernels.cu (cuFFT), then
// reference_cpu.cpp. The science / GPU-mapping is in ../THEORY.md.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"         // deconvolve_rl_gpu, Image, Psf
#include "reference_cpu.h"     // load_image, make_gaussian_psf, richardson_lucy_cpu, sharpness
#include "util/io.hpp"         // util::CpuTimer

static const char* PROJECT_ID   = "4.30";
static const char* PROJECT_NAME = "Deconvolution Microscopy (Richardson-Lucy, cuFFT)";

// ---- Fixed experiment parameters (shared contract with make_synthetic.py) --
//   PSF_RADIUS / PSF_SIGMA  : the Gaussian blur the microscope applies AND the
//                             PSF we deconvolve with (we assume a known PSF).
//   RL_ITERS                : Richardson-Lucy iteration count. Enough to sharpen
//                             clearly; small enough that the demo is instant.
static constexpr int    PSF_RADIUS = 4;
static constexpr double PSF_SIGMA  = 1.5;
static constexpr int    RL_ITERS   = 30;

// Verification tolerance. RL is iterated 30x in double precision; the GPU's FFT
// convolution and fused-multiply-add ordering differ from the CPU's direct
// convolution, so the two estimates match to ~1e-9 per pixel, not bit-exactly.
// We assert a max absolute pixel error below a physically-negligible 1e-6 on
// images whose intensities are O(1..100). Documented; honest (PATTERNS.md section 4).
static constexpr double ATOL = 1.0e-6;

int main(int argc, char** argv) {
    // ---- 1. Load ----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/blurred_image.txt";
    Image observed;
    try {
        observed = load_image(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // The known PSF (same as the blur applied to make the sample).
    const Psf psf = make_gaussian_psf(PSF_RADIUS, PSF_SIGMA);

    // ---- 2. CPU reference: RL with direct convolution (timed) ------------
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    const Image decon_cpu = richardson_lucy_cpu(observed, psf, RL_ITERS);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU: RL with cuFFT convolution (timed inside) ----------------
    Image decon_gpu;
    float gpu_ms = 0.0f;
    deconvolve_rl_gpu(observed, psf, RL_ITERS, decon_gpu, &gpu_ms);

    // ---- 4. Verify (max absolute per-pixel error) ------------------------
    double worst = 0.0;
    const int n = observed.size();
    for (int i = 0; i < n; ++i) {
        const double diff = std::fabs(decon_cpu.pix[i] - decon_gpu.pix[i]);
        if (diff > worst) worst = diff;
    }
    const bool pass = (worst <= ATOL);

    // ---- 5a. Deterministic report -> STDOUT ------------------------------
    // We report the GPU result (which the verify step just proved matches the
    // CPU). Scalars are rounded to a fixed number of decimals so the bytes are
    // identical run to run despite ~1e-9 FFT/FMA noise in the low digits.
    const double sharp_blurry = sharpness(observed);
    const double sharp_decon  = sharpness(decon_gpu);

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("image: %dx%d pixels   PSF: Gaussian r=%d sigma=%.1f   RL iters: %d\n",
                observed.w, observed.h, PSF_RADIUS, PSF_SIGMA, RL_ITERS);
    std::printf("sharpness (mean sq gradient):  blurry=%.4f  deconvolved=%.4f  (x%.2f sharper)\n",
                sharp_blurry, sharp_decon,
                (sharp_blurry > 0.0) ? sharp_decon / sharp_blurry : 0.0);

    // A small deterministic fingerprint of the restored image: the value at a
    // few fixed pixels along the main diagonal. Rounded to 3 decimals so the
    // last (noisy) digits never flip the stdout bytes.
    std::printf("restored pixels along diagonal (x=y):");
    for (int t = 0; t < 5; ++t) {
        const int x = (observed.w * (t + 1)) / 6;     // 5 evenly-spaced samples
        const int y = (observed.h * (t + 1)) / 6;
        const double v = decon_gpu.pix[static_cast<std::size_t>(y) * observed.w + x];
        std::printf(" (%d,%d)=%.3f", x, y, v);
    }
    std::printf("\n");

    // Total restored intensity should be conserved (the PSF sums to 1, RL is
    // intensity-preserving up to the boundary). Report it as a sanity scalar.
    double sum_obs = 0.0, sum_dec = 0.0;
    for (int i = 0; i < n; ++i) { sum_obs += observed.pix[i]; sum_dec += decon_gpu.pix[i]; }
    std::printf("total intensity:  observed=%.2f  deconvolved=%.2f\n", sum_obs, sum_dec);
    std::printf("RESULT: %s (GPU cuFFT deconvolution matches CPU reference within atol=1e-6)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR ------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d x %d image)\n", path.c_str(), observed.w, observed.h);
    std::fprintf(stderr, "[timing] CPU RL (direct conv): %.3f ms   GPU RL (cuFFT): %.3f ms\n",
                 cpu_ms, gpu_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- direct convolution is O(N*K) per iter; cuFFT is "
                         "O(N log N). The GPU's edge grows with image size and PSF width.\n");
    std::fprintf(stderr, "[verify] worst absolute per-pixel error (CPU vs GPU) = %.3e\n", worst);

    return pass ? 0 : 1;
}
