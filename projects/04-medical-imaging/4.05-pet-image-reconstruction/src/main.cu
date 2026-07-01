// ===========================================================================
// src/main.cu  --  Entry point: load counts, run MLEM on CPU + GPU, verify
// ---------------------------------------------------------------------------
// Project 4.5 : PET Image Reconstruction (MLEM / OS-EM)
//
// 5-step shape (the house style, cf. 4.01 / 12.01):
//   1. Load the measured sinogram + geometry (data/sample).
//   2. Precompute trig + the sensitivity image A^T 1 (shared by both solvers).
//   3a. CPU reference MLEM (reference_cpu.cpp) -- the trusted baseline.
//   3b. GPU MLEM (kernels.cu) -- forward/back projection gathers, no atomics.
//   4. VERIFY: the GPU image matches the CPU image within tolerance.
//   5. REPORT: deterministic image samples -> stdout; timing -> stderr.
//
// WHY THE SENSITIVITY IS COMPUTED ONCE, ON THE HOST
//   s_j = A^T 1 does not depend on the current image, so it is computed a single
//   time before iterating and reused by every MLEM step on both sides. Doing it
//   once (and identically for CPU and GPU) keeps the two solvers using the exact
//   same normalizer -- one less source of divergence.
//
// STDOUT is kept byte-for-byte deterministic so demo/run_demo can diff it against
// demo/expected_output.txt; timings (run-to-run varying) go to STDERR.
//
// Code tour: start here, then pet_geometry.h (shared math), reference_cpu.*,
// then kernels.cuh -> kernels.cu. See ../THEORY.md for the "why".
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // mlem_gpu, PetProblem, PetGeom
#include "reference_cpu.h"    // load_pet, compute_trig, sensitivity_cpu, mlem_cpu
#include "util/io.hpp"        // util::CpuTimer, util::max_abs_err

static const char* PROJECT_ID   = "4.5";
static const char* PROJECT_NAME = "PET Image Reconstruction (MLEM)";

// MLEM sums many interpolated projection samples over many iterations. The CPU
// (pixel-driven scatter) and GPU (LOR-parallel gather) accumulate the SAME terms
// in a different order, so they differ only by float rounding / FMA contraction
// that compounds slowly over iterations. A small absolute tolerance is the honest
// choice (docs/PATTERNS.md §4, same reasoning as 4.01 / 10.02).
static constexpr double TOLERANCE = 1.0e-3;

// Default iteration count if the sample file's header advises none (>0 wins).
static constexpr int DEFAULT_ITERS = 30;

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/sinogram_sample.txt";
    PetProblem p;
    int file_iters = 0;
    try {
        p = load_pet(path, file_iters);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int iters = (file_iters > 0) ? file_iters : DEFAULT_ITERS;

    // ---- 2. Shared setup: trig + sensitivity image ------------------------
    compute_trig(p);
    std::vector<float> sens;
    sensitivity_cpu(p, sens);

    // ---- 3a. CPU reference MLEM (timed) -----------------------------------
    std::vector<float> img_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    mlem_cpu(p, sens, iters, img_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3b. GPU MLEM (kernel time summed over iterations) ----------------
    std::vector<float> img_gpu;
    float gpu_kernel_ms = 0.0f;
    mlem_gpu(p, sens, iters, img_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    const double err = util::max_abs_err(img_cpu, img_gpu);
    const bool pass = err <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    const int N = p.geom.N;
    const int cpix = (N / 2) * N + (N / 2);       // center pixel (the hot disc)
    // Peak of the reconstruction and where it sits (ties -> lowest index).
    float vmax = img_gpu[0]; int amax = 0;
    for (std::size_t i = 1; i < img_gpu.size(); ++i)
        if (img_gpu[i] > vmax) { vmax = img_gpu[i]; amax = static_cast<int>(i); }
    // Total activity = sum of the reconstructed image (roughly conserved by MLEM).
    double total = 0.0;
    for (float v : img_gpu) total += v;

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("MLEM: %d iterations, %d angles x %d detectors -> %dx%d image\n",
                iters, p.geom.K, p.geom.D, N, N);
    std::printf("center pixel activity = %.4f\n", img_gpu[cpix]);
    std::printf("peak activity = %.4f at (px,py)=(%d,%d)\n", vmax, amax % N, amax / N);
    std::printf("total reconstructed activity = %.4f\n", total);
    std::printf("central row profile (8 samples):");
    for (int s = 0; s < 8; ++s) {
        const int px = (s * (N - 1)) / 7;         // 8 evenly spaced columns
        std::printf(" %.4f", img_gpu[(N / 2) * N + px]);
    }
    std::printf("\n");
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-03)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d x %d sinogram -> %dx%d image, %d iters)\n",
                 path.c_str(), p.geom.K, p.geom.D, N, N, iters);
    std::fprintf(stderr, "[timing] CPU MLEM: %.3f ms   GPU MLEM: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- per-iteration projection cost, and thus the "
                         "GPU's edge, grows with image size, angle count, and iterations.\n");
    std::fprintf(stderr, "[verify] max_abs_err = %.3e  (tolerance %.1e)\n", err, TOLERANCE);

    return pass ? 0 : 1;
}
