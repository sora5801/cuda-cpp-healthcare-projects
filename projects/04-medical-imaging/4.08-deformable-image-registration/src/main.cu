// ===========================================================================
// src/main.cu  --  Entry point: run Demons DIR on CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 4.8 : Deformable Image Registration (reduced-scope teaching version)
//
// 5-step shape (the shape EVERY project in this repo follows):
//   1. Load the fixed/moving image pair (data/sample).
//   2. CPU reference Demons (reference_cpu.cpp)  -> trusted displacement field.
//   3. GPU Demons (kernels.cu)                    -> identical per-pixel physics.
//   4. VERIFY: the GPU field matches the CPU field within a documented tolerance.
//   5. REPORT: deterministic registration quality (SSD before/after) to stdout;
//              timing to stderr.
//
// WHY THE STDOUT IS DETERMINISTIC
//   Every number printed to stdout is derived from the CPU field (a serial
//   double-precision computation that is bit-identical every run) and printed at
//   FIXED precision. The GPU field is only used for the PASS/FAIL verdict (a
//   thresholded comparison), never printed as a raw float -- so demo/run_demo
//   can diff stdout against expected_output.txt. Run-varying numbers (timings,
//   the exact GPU-vs-CPU difference) go to stderr, which is shown but not diffed.
//
// WHAT SUCCESS LOOKS LIKE
//   The moving image is a deformed copy of the fixed image (a shifted/warped
//   bright disk, see data/README.md). A correct DIR drives the SSD between the
//   WARPED moving image and the fixed image far below the initial SSD -- i.e.
//   the disk "snaps" onto the target. We report both, and the percent reduction.
//
// Code tour: start here, then demons.h (the per-pixel physics), kernels.cu, and
// reference_cpu.cpp for the serial baseline. See ../THEORY.md for the "why".
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // register_gpu, DemonsParams, DirImages
#include "reference_cpu.h"    // load_images, register_cpu, warp_image, ssd
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "4.8";
static const char* PROJECT_NAME = "Deformable Image Registration";

// Correctness tolerance on the DISPLACEMENT field (pixels). Demons is a long
// iterative solver: over hundreds of iterations the GPU's fused-multiply-add and
// the host compiler's arithmetic diverge by ~1e-5 even in double precision
// (PATTERNS.md §4). Displacements here are O(1-5) pixels, so 1e-3 pixel is far
// below anything visible -- "the same deformation" -- yet honest about the FP
// drift. We do NOT claim bit-identical fields.
static constexpr double TOLERANCE = 1.0e-3;

// The Demons run parameters. Chosen so the demo converges visibly in a fraction
// of a second on the tiny sample while exercising all three kernels many times.
static DemonsParams make_params(int nx, int ny) {
    DemonsParams P;
    P.nx      = nx;
    P.ny      = ny;
    P.iters   = 120;      // outer iterations (enough to register the sample)
    P.sigma   = 1.5;      // Gaussian regularization width, in pixels
    P.radius  = 5;        // kernel half-width (>= ceil(3*sigma)=5): captures ~99%
    P.epsilon = 1.0e-6;   // force-denominator floor (avoids 0/0 in flat regions)
    return P;
}

int main(int argc, char** argv) {
    // ---- 1. Load the image pair -------------------------------------------
    const std::string path =
        (argc > 1) ? argv[1] : "data/sample/dir_pair.txt";
    DirImages im;
    try {
        im = load_images(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const DemonsParams P = make_params(im.nx, im.ny);

    // Initial dissimilarity: SSD between the UNREGISTERED moving image and the
    // fixed image. This is the "before" number the method must beat.
    const double ssd_before = ssd(im.fixed, im.moving);

    // ---- 2. CPU reference Demons (timed) ----------------------------------
    std::vector<double> ux_cpu, uy_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    register_cpu(im, P, ux_cpu, uy_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // Warp the moving image by the CPU field and measure SSD-after (the number
    // we report -- deterministic because it comes from the serial CPU field).
    std::vector<double> warped_cpu;
    warp_image(im, ux_cpu, uy_cpu, warped_cpu);
    const double ssd_after = ssd(im.fixed, warped_cpu);

    // ---- 3. GPU Demons (loop timed inside the wrapper) --------------------
    std::vector<double> ux_gpu, uy_gpu;
    float gpu_kernel_ms = 0.0f;
    register_gpu(im, P, ux_gpu, uy_gpu, &gpu_kernel_ms);

    // ---- 4. Verify (GPU displacement field agrees with CPU) ---------------
    double worst = 0.0;   // largest per-component displacement difference (px)
    for (std::size_t i = 0; i < ux_cpu.size(); ++i) {
        worst = std::fmax(worst, std::fabs(ux_cpu[i] - ux_gpu[i]));
        worst = std::fmax(worst, std::fabs(uy_cpu[i] - uy_gpu[i]));
    }
    const bool pass = worst <= TOLERANCE;

    // Mean displacement magnitude of the (CPU) field -- a deterministic summary
    // of "how much the image had to move". Reported to stdout.
    double disp_sum = 0.0;
    for (std::size_t i = 0; i < ux_cpu.size(); ++i)
        disp_sum += std::sqrt(ux_cpu[i] * ux_cpu[i] + uy_cpu[i] * uy_cpu[i]);
    const double mean_disp = disp_sum / static_cast<double>(ux_cpu.size());

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) ----------
    const double pct = 100.0 * (ssd_before - ssd_after) / ssd_before;
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("Demons DIR: %dx%d image, %d iters, sigma=%.2f px (radius=%d)\n",
                im.nx, im.ny, P.iters, P.sigma, P.radius);
    std::printf("SSD before = %.4f\n", ssd_before);
    std::printf("SSD after  = %.4f  (%.2f%% reduction)\n", ssd_after, pct);
    std::printf("mean |displacement| = %.4f px\n", mean_disp);
    // A short deterministic profile: the x-displacement sampled along the middle
    // row lets the learner see the field's shape (it should peak where the disk
    // moved most). Printed at fixed precision from the CPU field.
    std::printf("u_x along center row (8 samples):");
    const int cy = im.ny / 2;
    for (int s = 0; s < 8; ++s) {
        const int x = (s * (im.nx - 1)) / 7;
        std::printf(" %.4f", ux_cpu[cy * im.nx + x]);
    }
    std::printf("\n");
    std::printf("RESULT: %s (GPU field matches CPU within tol=1.0e-03 px)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) -----------------
    std::fprintf(stderr, "[data]   source: %s  (%d x %d pixels)\n",
                 path.c_str(), im.nx, im.ny);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- the GPU's edge grows with "
                         "image/volume size; a real 256^3 DIR is ~10^7 voxels x "
                         "hundreds of iters, where the GPU is essential.\n");
    std::fprintf(stderr, "[verify] worst displacement diff = %.3e px  (tolerance %.1e)\n",
                 worst, TOLERANCE);

    return pass ? 0 : 1;
}
