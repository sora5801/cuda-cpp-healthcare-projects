// ===========================================================================
// src/main.cu  --  Entry point: load 4D-CT sinogram, reconstruct, verify, report
// ---------------------------------------------------------------------------
// Project 4.19 : Motion-Compensated 4D-CT Reconstruction (2-D teaching version)
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the phase-binned sinogram + geometry (data/sample).
//   2. Ramp-filter every projection row once on the host (shared by both recons).
//   3. Reconstruct FOUR images:
//        - NAIVE 4D-FBP        on CPU  (ignores motion -> blurred)   [reference]
//        - NAIVE 4D-FBP        on GPU  (must match the CPU naive)
//        - MOTION-COMPENSATED  on CPU  (DVF warp -> sharp)           [reference]
//        - MOTION-COMPENSATED  on GPU  (must match the CPU MCR)
//   4. VERIFY: each GPU image matches its CPU reference within tolerance.
//   5. REPORT: deterministic image samples + sharpness gain -> stdout; timing ->
//      stderr. The key teaching number is the SHARPNESS RATIO: motion
//      compensation makes the reconstruction crisper, which we quantify.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Timings (which vary run-to-run) go to STDERR.
//
// READ THIS FIRST in the code tour, then mc4dct.h (the physics) ->
// kernels.cuh/kernels.cu (GPU) and reference_cpu.cpp (CPU). See ../THEORY.md.
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // reconstruct_gpu (GPU path)
#include "reference_cpu.h"    // load_4dct, ramp_filter, reconstruct_cpu, sharpness
#include "util/io.hpp"        // util::CpuTimer, util::max_abs_err

static const char* PROJECT_ID   = "4.19";
static const char* PROJECT_NAME = "Motion-Compensated 4D-CT Reconstruction";

// Backprojection sums many ramp-filtered samples and (for MCR) evaluates the DVF
// per pixel; CPU and GPU run the SAME mc_pixel() so they differ only by float
// rounding / FMA contraction. A small absolute tolerance is appropriate and, in
// practice, the agreement is near machine precision (see THEORY.md numerics).
static constexpr double TOLERANCE = 1.0e-3;

int main(int argc, char** argv) {
    // ---- 1. Load the phase-binned sinogram ---------------------------------
    const std::string path =
        (argc > 1) ? argv[1] : "data/sample/sinogram4d_sample.txt";
    FourDCTProblem prob;
    try {
        prob = load_4dct(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const Geom& g = prob.geom;

    // ---- 2. Ramp filter (host, shared by both reconstructions) -------------
    std::vector<float> cosv, sinv, filtered;
    compute_trig(prob, cosv, sinv);
    ramp_filter(prob, filtered);

    // ---- 3a. NAIVE reconstruction: CPU reference (timed) -------------------
    std::vector<float> naive_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    reconstruct_cpu(prob, filtered, cosv, sinv, /*motion_comp=*/0, naive_cpu);
    const double cpu_naive_ms = cpu_timer.stop_ms();

    // ---- 3b. NAIVE reconstruction: GPU -------------------------------------
    std::vector<float> naive_gpu;
    float gpu_naive_ms = 0.0f;
    reconstruct_gpu(prob, filtered, cosv, sinv, /*motion_comp=*/0, naive_gpu, &gpu_naive_ms);

    // ---- 3c. MOTION-COMPENSATED reconstruction: CPU reference (timed) ------
    std::vector<float> mc_cpu;
    cpu_timer.start();
    reconstruct_cpu(prob, filtered, cosv, sinv, /*motion_comp=*/1, mc_cpu);
    const double cpu_mc_ms = cpu_timer.stop_ms();

    // ---- 3d. MOTION-COMPENSATED reconstruction: GPU ------------------------
    std::vector<float> mc_gpu;
    float gpu_mc_ms = 0.0f;
    reconstruct_gpu(prob, filtered, cosv, sinv, /*motion_comp=*/1, mc_gpu, &gpu_mc_ms);

    // ---- 4. Verify BOTH GPU images against their CPU references ------------
    const double err_naive = util::max_abs_err(naive_cpu, naive_gpu);
    const double err_mc     = util::max_abs_err(mc_cpu, mc_gpu);
    const bool pass = (err_naive <= TOLERANCE) && (err_mc <= TOLERANCE);

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    // HEADLINE metric: PEAK RECOVERY of the moving nodule. Motion smears its
    // energy across every phase's position, so the NAIVE peak sits below its
    // true density (1.0); motion compensation re-aligns the phases and the peak
    // climbs back toward 1.0. We report both peaks, the recovery ratio, and how
    // close the MCR peak lands to the known truth (validates the science, not
    // just CPU==GPU agreement -- PATTERNS.md section 4).
    const int N = g.img;
    int naive_px = 0, naive_py = 0, mc_px = 0, mc_py = 0;
    const float peak_naive = image_peak(naive_gpu, N, &naive_px, &naive_py);
    const float peak_mc     = image_peak(mc_gpu, N, &mc_px, &mc_py);
    const double peak_ratio = (peak_naive > 0.0f) ? (double)peak_mc / peak_naive : 0.0;
    // Secondary metric: whole-image sharpness (mean squared gradient).
    const double sharp_naive = image_sharpness(naive_gpu, N);
    const double sharp_mc     = image_sharpness(mc_gpu, N);

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("2-D 4D-CT: %d phases x %d angles/phase = %d projections, "
                "%d detectors -> %dx%d image\n",
                g.n_phases, g.n_ang_phase, prob.total_angles(), g.n_det, N, N);
    std::printf("breathing DVF amplitude = %.4f world units (phase 0 = reference)\n", g.amp);
    std::printf("naive 4D-FBP  peak = %.4f at (%d,%d)   sharpness = %.6f\n",
                peak_naive, naive_px, naive_py, sharp_naive);
    std::printf("motion-comp   peak = %.4f at (%d,%d)   sharpness = %.6f\n",
                peak_mc, mc_px, mc_py, sharp_mc);
    std::printf("peak recovery (MCR / naive) = %.4fx   (true nodule density = 1.0)\n",
                peak_ratio);
    // A short central-column profile through the MOVING nodule's x-position, so
    // the deterministic stdout captures its reconstructed shape (naive vs MCR).
    std::printf("motion-comp column profile at peak x (8 samples top->bottom):");
    for (int s = 0; s < 8; ++s) {
        const int py = (s * (N - 1)) / 7;      // 8 evenly spaced rows
        std::printf(" %.4f", mc_gpu[static_cast<std::size_t>(py) * N + mc_px]);
    }
    std::printf("\n");
    // PASS requires: (a) GPU matches CPU on BOTH reconstructions, and (b) motion
    // compensation actually improves the peak (the physics worked).
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-03; MCR recovers the moving nodule)\n",
                (pass && peak_ratio > 1.0) ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d proj x %d det -> %dx%d image)\n",
                 path.c_str(), prob.total_angles(), g.n_det, N, N);
    std::fprintf(stderr, "[timing] naive : CPU %.3f ms   GPU %.3f ms\n",
                 cpu_naive_ms, gpu_naive_ms);
    std::fprintf(stderr, "[timing] MCR   : CPU %.3f ms   GPU %.3f ms\n",
                 cpu_mc_ms, gpu_mc_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- the GPU's edge grows with image size, "
                         "projection count, and (in real MCR) iterated DVF estimation.\n");
    std::fprintf(stderr, "[verify] max_abs_err naive = %.3e   MCR = %.3e   (tolerance %.1e)\n",
                 err_naive, err_mc, TOLERANCE);

    return (pass && peak_ratio > 1.0) ? 0 : 1;
}
