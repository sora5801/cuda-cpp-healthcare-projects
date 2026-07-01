// ===========================================================================
// src/main.cu  --  Entry point: load projections, run SART on CPU + GPU, verify
// ---------------------------------------------------------------------------
// Project 4.14 : Digital Breast Tomosynthesis
//
// WHAT THIS FILE DOES  (the 5-step shape EVERY project in this repo follows)
//   1. LOAD the limited-angle projection problem (data/sample or a CLI path).
//   2. Precompute the angle (cos/sin) table shared by both reconstructions.
//   3a. CPU reference SART reconstruction  (reference_cpu.cpp)  -> trusted answer.
//   3b. GPU SART reconstruction            (kernels.cu)          -> the thing taught.
//   4. VERIFY: assert the GPU image matches the CPU image within tolerance.
//   5. REPORT: deterministic reconstruction samples to stdout; timing to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Timings (which vary run-to-run) go to STDERR, shown
//   but never diffed (docs/PATTERNS.md §3).
//
// WHY THE OUTPUT IS INTERPRETABLE
//   The committed sample encodes a synthetic compressed-breast slice: a soft
//   ellipse of fibroglandular tissue with two small dense "lesion" discs at
//   KNOWN locations. A correct reconstruction recovers elevated attenuation at
//   those spots, so we report the reconstructed value at each planted lesion and
//   the location of the global peak -- a result you can sanity-check by eye.
//
// READ THIS FIRST in the code tour, then kernels.cuh -> kernels.cu, and
// reference_cpu.cpp for the baseline. See ../THEORY.md for the "why".
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // reconstruct_sart_gpu (GPU path), DBTProblem
#include "reference_cpu.h"    // load_dbt, compute_angles, reconstruct_sart_cpu
#include "util/io.hpp"        // util::CpuTimer, util::max_abs_err

static const char* PROJECT_ID   = "4.14";
static const char* PROJECT_NAME = "Digital Breast Tomosynthesis";

// Verification tolerance. SART forward/backprojection sum many bilinearly
// interpolated float samples over several iterations; the GPU's fused
// multiply-add and the host compiler diverge by only float rounding, so a small
// ABSOLUTE tolerance is the honest bar (docs/PATTERNS.md §4, same class as 4.01).
static constexpr double TOLERANCE = 1.0e-3;

int main(int argc, char** argv) {
    // ---- 1. Load the limited-angle problem ---------------------------------
    const std::string path = (argc > 1) ? argv[1]
                                        : "data/sample/dbt_sample.txt";
    DBTProblem p;
    try {
        p = load_dbt(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. Angle table (shared by CPU and GPU for bit-identical trig) -----
    std::vector<float> cosv, sinv;
    compute_angles(p, cosv, sinv);

    // ---- 3a. CPU reference SART (timed) ------------------------------------
    std::vector<float> img_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    reconstruct_sart_cpu(p, cosv, sinv, img_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3b. GPU SART (kernel time accumulated inside the driver) ----------
    std::vector<float> img_gpu;
    float gpu_kernel_ms = 0.0f;
    reconstruct_sart_gpu(p, cosv, sinv, img_gpu, &gpu_kernel_ms);

    // ---- 4. Verify GPU vs CPU ----------------------------------------------
    const double err  = util::max_abs_err(img_cpu, img_gpu);
    const bool   pass = err <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    const int N = p.img;

    // Locate the global peak of the reconstruction (should sit on a lesion).
    float vmax = img_gpu[0];
    int   amax = 0;
    for (std::size_t i = 1; i < img_gpu.size(); ++i)
        if (img_gpu[i] > vmax) { vmax = img_gpu[i]; amax = static_cast<int>(i); }

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("Limited-angle SART: %d projections over +/-%.1f deg, %d detectors -> %dx%d image\n",
                p.n_angles, static_cast<double>(p.half_span) * 180.0 / 3.14159265358979323846,
                p.n_det, N, N);
    std::printf("SART: %d iterations, relaxation lambda = %.2f\n", p.n_iters, p.relax);
    std::printf("center pixel value = %.4f\n", img_gpu[(N / 2) * N + (N / 2)]);
    std::printf("peak value = %.4f at (px,py)=(%d,%d)\n", vmax, amax % N, amax / N);
    // Central-row profile: 8 evenly spaced columns across the reconstruction.
    // With two planted lesions on the central row this profile shows two humps.
    std::printf("central row profile (8 samples):");
    for (int s = 0; s < 8; ++s) {
        const int px = (s * (N - 1)) / 7;
        std::printf(" %.4f", img_gpu[(N / 2) * N + px]);
    }
    std::printf("\n");
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-03)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%d angles x %d det -> %dx%d image, %d SART iters)\n",
                 path.c_str(), p.n_angles, p.n_det, N, N, p.n_iters);
    std::fprintf(stderr, "[timing] CPU SART: %.3f ms   GPU SART (all kernels): %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- the GPU's edge grows with image size, "
                         "detector count, and iteration count (clinical DBT volumes are far larger).\n");
    std::fprintf(stderr, "[verify] max_abs_err = %.3e  (tolerance %.1e)\n", err, TOLERANCE);

    return pass ? 0 : 1;
}
