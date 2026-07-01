// ===========================================================================
// src/main.cu  --  Entry point: load volume, run CPU + GPU vesselness, verify
// ---------------------------------------------------------------------------
// Project 4.26 : Vessel Segmentation & Centerline Extraction
//
// 5-step shape (every project in this repo follows it):
//   1. Load the 3-D volume + Frangi params (data/sample).
//   2. Gaussian-smooth on the host (both paths start from the same data).
//   3a. CPU reference Frangi vesselness  (reference_cpu.cpp) -> trusted answer.
//   3b. GPU Frangi vesselness            (kernels.cu)        -> the thing taught.
//   4. VERIFY: the GPU score field matches the CPU field within tolerance.
//   5. REPORT: deterministic segmentation summary to stdout; timing to stderr.
//
//   The synthetic volume embeds a bright tube along the x-axis, so the vesselness
//   response should PEAK on the vessel centerline and a threshold should recover
//   roughly the tube's voxel count -- an interpretable, verifiable result.
//
//   STDOUT is byte-for-byte deterministic (diffed by demo/run_demo); timings and
//   run-varying numbers go to STDERR (shown, not diffed).
//
// Code tour: start here, then frangi.h (the per-voxel math), reference_cpu.cpp
// (the serial pipeline), kernels.cuh -> kernels.cu (the GPU twin). See ../THEORY.md.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // vesselness_gpu (GPU path)
#include "reference_cpu.h"    // load_volume, gaussian_smooth, vesselness_cpu, summarize
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "4.26";
static const char* PROJECT_NAME = "Vessel Segmentation & Centerline Extraction";

// CPU and GPU run the SAME closed-form eigen + Frangi math on the SAME smoothed
// (host) volume; they differ only by FMA/rounding order in a handful of flops,
// so scores agree to ~1e-9. We verify to a comfortably strict 1e-6. (THEORY 5.)
static constexpr double TOLERANCE = 1.0e-6;

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/vessel_volume.txt";
    VesselJob job;
    try {
        job = load_volume(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. Gaussian smooth on the host (shared by both paths) -------------
    Volume smoothed;
    gaussian_smooth(job.vol, job.fp.sigma, smoothed);

    // ---- 3a. CPU reference (timed) -----------------------------------------
    std::vector<float> v_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    vesselness_cpu(smoothed, job.fp, v_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3b. GPU vesselness (kernel timed inside the wrapper) --------------
    std::vector<float> v_gpu;
    float gpu_kernel_ms = 0.0f;
    vesselness_gpu(smoothed, job.fp, v_gpu, &gpu_kernel_ms);

    // ---- 4. Verify (score fields agree) ------------------------------------
    double err = 0.0;
    for (std::size_t k = 0; k < v_cpu.size(); ++k) {
        const double d = std::fabs((double)v_cpu[k] - (double)v_gpu[k]);
        if (d > err) err = d;
    }
    const bool pass = err <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    // Summaries come from the GPU field (verified equal to the CPU field). The
    // sum is scaled to an integer-like checksum so it prints reproducibly.
    long long n_vessel = 0; double vsum = 0.0, pmax = 0.0;
    int px = 0, py = 0, pz = 0;
    summarize(job.vol, v_gpu, job.mask_threshold, n_vessel, vsum, px, py, pz, pmax);

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("volume: %dx%dx%d voxels, sigma=%.2f, alpha=%.2f beta=%.2f c=%.1f, %s\n",
                job.vol.nx, job.vol.ny, job.vol.nz, job.fp.sigma,
                job.fp.alpha, job.fp.beta, job.fp.c,
                job.fp.bright_vessels ? "bright-on-dark" : "dark-on-bright");
    std::printf("Frangi vesselness: peak = %.6f at voxel (%d,%d,%d)\n",
                pmax, px, py, pz);
    std::printf("segmentation: %lld voxels with vesselness >= %.2f\n",
                n_vessel, job.mask_threshold);
    std::printf("vesselness checksum (sum*1000, rounded) = %lld\n",
                (long long)std::llround(vsum * 1000.0));
    // A short across-vessel profile through the peak voxel (fixed x=peak, vary y
    // at z=peak): a clean single-peak ridge -> the filter localizes the vessel.
    std::printf("across-vessel vesselness profile v(y) at x=%d z=%d:\n", px, pz);
    for (int y = 0; y < job.vol.ny; ++y)
        std::printf(" %.4f", v_gpu[vox_idx(px, y, pz, job.vol.nx, job.vol.ny)]);
    std::printf("\n");
    std::printf("RESULT: %s (GPU vesselness matches CPU within tol=1.0e-06)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d x %d x %d voxels)\n",
                 path.c_str(), job.vol.nx, job.vol.ny, job.vol.nz);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- one thread per voxel; the "
                         "GPU's edge grows with clinical-size (~10^6-10^8 voxel) volumes.\n");
    std::fprintf(stderr, "[verify] max score diff = %.3e  (tolerance %.1e)\n", err, TOLERANCE);

    return pass ? 0 : 1;
}
