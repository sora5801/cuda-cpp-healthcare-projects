// ===========================================================================
// src/main.cu  --  Entry point: load image, degrade, super-resolve, verify
// ---------------------------------------------------------------------------
// Project 4.24 : CT/MRI Super-Resolution   (reduced-scope teaching version)
//
// 6-step shape:
//   1. Load a ground-truth HIGH-RES image (data/sample).
//   2. Degrade it R x -> a LOW-RES image (downsample_avg): the SR input.
//   3. Build the fixed synthetic network weights (make_sr_weights).
//   4. Run BOTH the CPU reference and the GPU kernel to super-resolve the LR
//      image back up to HR, and time each.
//   5. VERIFY: GPU HR output matches CPU HR output within a tiny tolerance;
//      also report PSNR vs. the ground truth (and vs. a naive baseline) so the
//      learner sees that the network beats plain nearest-neighbour upscaling.
//   6. REPORT: a DETERMINISTIC result block to stdout; timing to stderr.
//
// Code tour: start here, then sr_core.h (the per-pixel network math), then
// kernels.cu (the one-thread-per-HR-pixel launch), then reference_cpu.cpp.
// The science/GPU-mapping/derivation is in ../THEORY.md.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // super_resolve_gpu, Image
#include "reference_cpu.h"    // load_image, make_sr_weights, downsample_avg, ...
#include "sr_core.h"          // SR_SCALE, SR_C_FEAT (for the report header)
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "4.24";
static const char* PROJECT_NAME = "CT/MRI Super-Resolution";

// GPU/CPU agreement tolerance. Both sides evaluate sr_hr_pixel() -- the SAME
// __host__ __device__ function -- so for R=2 the float ops are identical and the
// results agree to the last bit; we allow a hair (1e-6) only to absorb any
// compiler FMA-contraction difference on the host vs. device. (PATTERNS.md §4.)
static constexpr double TOLERANCE = 1.0e-6;

// ---------------------------------------------------------------------------
// nearest_upsample: the naive baseline SR method -- repeat each LR pixel R x R.
//   Used only to CONTRAST against the learned network in the report (so PSNR
//   improvement is meaningful, not compared to nothing). Not on the GPU path.
// ---------------------------------------------------------------------------
static Image nearest_upsample(const Image& lr, int scale) {
    Image hr;
    hr.w = lr.w * scale; hr.h = lr.h * scale;
    hr.pix.assign(static_cast<size_t>(hr.w) * hr.h, 0.0f);
    for (int hy = 0; hy < hr.h; ++hy)
        for (int hx = 0; hx < hr.w; ++hx)
            hr.pix[(size_t)hy * hr.w + hx] =
                lr.pix[(size_t)(hy / scale) * lr.w + (hx / scale)];
    return hr;
}

int main(int argc, char** argv) {
    // ---- 1. Load ground-truth HR image ------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/phantom_hr.txt";
    Image hr_truth;
    try {
        hr_truth = load_image(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. Degrade -> LR input -------------------------------------------
    const Image lr = downsample_avg(hr_truth, SR_SCALE);

    // ---- 3. Weights -------------------------------------------------------
    const SrWeights W = make_sr_weights();

    // ---- 4. CPU reference (timed) -----------------------------------------
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    const Image hr_cpu = super_resolve_cpu(lr, W, SR_SCALE);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 4b. GPU (kernel timed) -------------------------------------------
    Image hr_gpu;
    float gpu_kernel_ms = 0.0f;
    super_resolve_gpu(lr, W, SR_SCALE, hr_gpu, &gpu_kernel_ms);

    // ---- 5. Verify GPU == CPU ---------------------------------------------
    const double err = util::max_abs_err(hr_cpu.pix, hr_gpu.pix);
    const bool pass = err <= TOLERANCE;

    // Quality metrics (computed from the GPU output; identical to CPU within tol).
    const Image hr_nn = nearest_upsample(lr, SR_SCALE);   // naive baseline
    const double psnr_nn  = psnr(hr_truth, hr_nn);        // baseline vs. truth
    const double psnr_sr  = psnr(hr_truth, hr_gpu);       // network vs. truth

    // ---- 6a. Deterministic report -> STDOUT --------------------------------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("scale R=%d  |  LR %dx%d -> HR %dx%d  |  net: %d feat ch, 3x3 conv + subpixel\n",
                SR_SCALE, lr.w, lr.h, hr_gpu.w, hr_gpu.h, SR_C_FEAT);
    std::printf("PSNR nearest-neighbour vs truth = %.4f dB\n", psnr_nn);
    std::printf("PSNR super-resolved   vs truth = %.4f dB\n", psnr_sr);
    std::printf("PSNR improvement over baseline = %.4f dB\n", psnr_sr - psnr_nn);
    // A handful of deterministic HR pixels (evenly spaced along the raster) so
    // the diff in run_demo catches any change to the numerics.
    std::printf("HR samples (8 evenly spaced):");
    const size_t hn = hr_gpu.pix.size();
    for (int s = 0; s < 8; ++s) {
        const size_t i = (static_cast<size_t>(s) * (hn - 1)) / 7;
        std::printf(" %.6f", hr_gpu.pix[i]);
    }
    std::printf("\n");
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-06)\n", pass ? "PASS" : "FAIL");

    // ---- 6b. Run-varying detail -> STDERR ----------------------------------
    std::fprintf(stderr, "[data]   source: %s  (ground-truth HR %dx%d)\n",
                 path.c_str(), hr_truth.w, hr_truth.h);
    std::fprintf(stderr, "[timing] CPU SR: %.3f ms   GPU SR kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- one thread per HR pixel; the GPU's edge "
                         "grows with image size and with 3-D volumes / batches of slices.\n");
    std::fprintf(stderr, "[verify] max_abs_err(GPU,CPU) = %.3e  (tolerance %.1e)\n", err, TOLERANCE);

    return pass ? 0 : 1;
}
