// ===========================================================================
// src/main.cu  --  Entry point: load sinogram, run CPU + GPU SIRT, verify, report
// ---------------------------------------------------------------------------
// Project 4.2 : Iterative / Model-Based CT Reconstruction
//
// THE 5-STEP SHAPE (every project in this repo follows it)
//   1. Load the measured sinogram + geometry (data/sample).
//   2. Precompute the shared trig tables and SIRT normalization weights.
//   3a. CPU reference reconstruction (reference_cpu.cpp)      -> trusted answer.
//   3b. GPU reconstruction           (kernels.cu)             -> the thing taught.
//   4. VERIFY: GPU image matches CPU image within tolerance   -> correctness.
//      (plus a scientific check: RMS error of the reconstruction vs. truth.)
//   5. REPORT: deterministic image samples to stdout; timing to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Timings (which vary run-to-run) go to STDERR,
//   which the demo shows but does not diff (PATTERNS.md §3).
//
// READ THIS FIRST in the code tour, then kernels.cuh -> kernels.cu, and
// reference_cpu.cpp for the baseline. See ../THEORY.md for the "why".
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // sirt_gpu (GPU path)
#include "reference_cpu.h"    // load_ct, compute_trig, SIRT reference, rms_error
#include "util/io.hpp"        // util::CpuTimer, util::max_abs_err

static const char* PROJECT_ID   = "4.2";
static const char* PROJECT_NAME = "Iterative / Model-Based CT Reconstruction";

// Correctness tolerance for GPU-vs-CPU agreement. SIRT runs many iterations of
// float projections; the GPU's fused-multiply-add contracts differently from the
// host compiler, so the two images drift by ~1e-4 over the run even though each
// step's math is "the same" (PATTERNS.md §4, the long-iterative case). We verify
// to a physically-negligible absolute tolerance and say so -- we do NOT pretend
// the images are bit-identical.
static constexpr double TOLERANCE = 2.0e-3;

int main(int argc, char** argv) {
    // ---- 1. Load ------------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/sinogram_sample.txt";
    CTProblem ct;
    try {
        ct = load_ct(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. Shared precomputation (trig + SIRT weights) --------------------
    std::vector<float> cosv, sinv, row_scale, col_scale;
    compute_trig(ct.n_angles, cosv, sinv);
    compute_sirt_weights(ct, cosv, sinv, row_scale, col_scale);

    // ---- 3a. CPU reference SIRT (timed) ------------------------------------
    std::vector<float> img_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    reconstruct_sirt_cpu(ct, cosv, sinv, row_scale, col_scale, img_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3b. GPU SIRT (device time summed over all iterations) -------------
    std::vector<float> img_gpu;
    float gpu_kernel_ms = 0.0f;
    sirt_gpu(ct, cosv, sinv, row_scale, col_scale, img_gpu, &gpu_kernel_ms);

    // ---- 4. Verify GPU vs CPU ----------------------------------------------
    const double err  = util::max_abs_err(img_cpu, img_gpu);
    const bool   pass = err <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    const int N = ct.img;
    const int cpix = (N / 2) * N + (N / 2);        // center pixel
    // Peak of the reconstruction (a stable, order-independent summary).
    float vmax = img_gpu[0]; int amax = 0;
    for (std::size_t i = 1; i < img_gpu.size(); ++i)
        if (img_gpu[i] > vmax) { vmax = img_gpu[i]; amax = static_cast<int>(i); }

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("SIRT%s: %d angles x %d detectors -> %dx%d image, %d iterations\n",
                (ct.tv_weight > 0.0f) ? "+TV" : "", ct.n_angles, ct.n_det, N, N, ct.iters);
    std::printf("lambda = %.3f  tv_weight = %.4f\n", ct.lambda, ct.tv_weight);
    std::printf("center pixel value = %.4f\n", img_gpu[cpix]);
    std::printf("max reconstructed value = %.4f at (px,py)=(%d,%d)\n",
                vmax, amax % N, amax / N);
    std::printf("central row profile (8 samples):");
    for (int s = 0; s < 8; ++s) {
        const int px = (s * (N - 1)) / 7;          // 8 evenly spaced columns
        std::printf(" %.4f", img_gpu[(N / 2) * N + px]);
    }
    std::printf("\n");
    // A scientific check when the sample ships ground truth: how close is the
    // reconstruction to the object we actually scanned? (Reported to 4 dp so it
    // stays deterministic across the tiny CPU/GPU drift.)
    if (!ct.truth.empty()) {
        const double rmse = rms_error(img_gpu, ct.truth);
        std::printf("reconstruction RMSE vs truth = %.4f\n", rmse);
    }
    std::printf("RESULT: %s (GPU matches CPU within tol=2.0e-03)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d x %d sinogram -> %dx%d image)\n",
                 path.c_str(), ct.n_angles, ct.n_det, N, N);
    std::fprintf(stderr, "[timing] CPU SIRT: %.3f ms   GPU SIRT: %.3f ms   (%d iterations)\n",
                 cpu_ms, gpu_kernel_ms, ct.iters);
    std::fprintf(stderr, "[timing] teaching artifact -- the GPU's edge grows with image "
                         "size, view count, and iteration budget (clinical MBIR is far larger).\n");
    std::fprintf(stderr, "[verify] max_abs_err(GPU,CPU) = %.3e  (tolerance %.1e)\n", err, TOLERANCE);

    return pass ? 0 : 1;
}
