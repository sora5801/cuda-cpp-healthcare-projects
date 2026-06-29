// ===========================================================================
// src/main.cu  --  Entry point: load sinogram, filter, backproject, verify
// ---------------------------------------------------------------------------
// Project 4.01 : CT Reconstruction (Filtered Backprojection)
//
// 5-step shape:
//   1. Load the sinogram + geometry (data/sample).
//   2. Ramp-filter on the host (shared by both reconstructions).
//   3a. CPU reference backprojection (reference_cpu.cpp).
//   3b. GPU backprojection (kernels.cu).
//   4. VERIFY: GPU image matches CPU image within tolerance.
//   5. REPORT: deterministic image samples to stdout; timing to stderr.
//
// The ramp filter runs once on the host because it is cheap (1-D per row) and we
// want BOTH reconstructions to start from the identical filtered data -- the GPU
// teaching point is the BACKPROJECTION gather.
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // backproject_gpu, CTProblem
#include "reference_cpu.h"    // load_ct, compute_trig, ramp_filter, backproject_cpu
#include "util/io.hpp"        // util::CpuTimer, util::max_abs_err

static const char* PROJECT_ID   = "4.1";
static const char* PROJECT_NAME = "CT Reconstruction (Filtered Backprojection)";

// Backprojection sums many filtered samples; CPU and GPU differ only by float
// rounding / FMA contraction, so a small absolute tolerance is appropriate.
static constexpr double TOLERANCE = 1.0e-3;

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/sinogram_sample.txt";
    CTProblem ct;
    try {
        ct = load_ct(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. Ramp filter (host, shared) ------------------------------------
    std::vector<float> cosv, sinv, filtered;
    compute_trig(ct.n_angles, cosv, sinv);
    ramp_filter(ct, filtered);

    // ---- 3a. CPU reference (timed) ----------------------------------------
    std::vector<float> img_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    backproject_cpu(ct, filtered, cosv, sinv, img_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3b. GPU reconstruction (kernel timed) ----------------------------
    std::vector<float> img_gpu;
    float gpu_kernel_ms = 0.0f;
    backproject_gpu(ct, filtered, cosv, sinv, img_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    const double err = util::max_abs_err(img_cpu, img_gpu);
    const bool pass = err <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    const int N = ct.img;
    const int cpix = (N / 2) * N + (N / 2);          // center pixel index
    float vmax = img_gpu[0]; int amax = 0;
    for (std::size_t i = 1; i < img_gpu.size(); ++i)
        if (img_gpu[i] > vmax) { vmax = img_gpu[i]; amax = static_cast<int>(i); }

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("Parallel-beam FBP: %d projections x %d detectors -> %dx%d image\n",
                ct.n_angles, ct.n_det, N, N);
    std::printf("ramp filter: Ram-Lak (spatial)\n");
    std::printf("center pixel value = %.4f\n", img_gpu[cpix]);
    std::printf("max reconstructed value = %.4f at (px,py)=(%d,%d)\n",
                vmax, amax % N, amax / N);
    std::printf("central row profile (8 samples):");
    for (int s = 0; s < 8; ++s) {
        const int px = (s * (N - 1)) / 7;            // 8 evenly spaced columns
        std::printf(" %.4f", img_gpu[(N / 2) * N + px]);
    }
    std::printf("\n");
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-03)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d x %d sinogram -> %dx%d image)\n",
                 path.c_str(), ct.n_angles, ct.n_det, N, N);
    std::fprintf(stderr, "[timing] CPU backproject: %.3f ms   GPU backproject: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- the GPU's edge grows with image size "
                         "and projection count (clinical volumes are far larger).\n");
    std::fprintf(stderr, "[verify] max_abs_err = %.3e  (tolerance %.1e)\n", err, TOLERANCE);

    return pass ? 0 : 1;
}
