// ===========================================================================
// src/main.cu  --  Entry point: load CT volume, render DRR (CPU+GPU), verify
// ---------------------------------------------------------------------------
// Project 4.28 : GPU-Accelerated DRR Generation for 2D/3D Registration
//
// 5-step shape (the same skeleton every project in this repo uses):
//   1. Load the CT volume (data/sample) and build a fixed DRR geometry.
//   2a. CPU reference DRR  (reference_cpu.cpp, render_drr_cpu).
//   2b. GPU DRR            (kernels.cu, render_drr_gpu).
//   3. VERIFY: the GPU image matches the CPU image within a documented tolerance.
//   4. REPORT: deterministic DRR samples to stdout; timing/detail to stderr.
//
// WHY CPU AND GPU AGREE TIGHTLY: both call the identical integrate_ray() from the
// shared drr_core.h, so the only difference is float rounding / FMA contraction
// over the per-ray sum -> a small absolute tolerance suffices (see TOLERANCE).
//
// READ THIS AFTER: drr_core.h (the physics), reference_cpu.h (CPU side),
// kernels.cuh (GPU side).
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // render_drr_gpu
#include "reference_cpu.h"    // load_volume, make_demo_geometry, render_drr_cpu
#include "util/io.hpp"        // util::CpuTimer, util::max_abs_err

static const char* PROJECT_ID   = "4.28";
static const char* PROJECT_NAME = "GPU-Accelerated DRR Generation for 2D/3D Registration";

// DRR detector panel size (pixels). Kept modest so the committed demo renders in
// well under a second; clinical DRRs are 256x256..512x512 and the GPU's edge over
// the CPU only grows from here (the work is width*height*n_steps).
static constexpr int DRR_WIDTH  = 128;
static constexpr int DRR_HEIGHT = 128;

// Ray-march sampling step in mm. Smaller = more accurate quadrature, more steps.
// 1.0 mm is a sensible teaching default for a volume with ~1-2 mm voxels.
static constexpr float STEP_MM = 1.0f;

// Each DRR pixel is a sum of ~hundreds of mu*step terms; CPU and GPU differ only
// by float rounding / fused-multiply-add contraction over that sum, so a small
// ABSOLUTE tolerance is the honest check (PATTERNS.md section 4). Typical pixel
// values here are O(1-10), so 1e-3 is well below any visible difference.
static constexpr double TOLERANCE = 1.0e-3;

int main(int argc, char** argv) {
    // ---- 1. Load volume + build geometry -----------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/ct_volume_sample.txt";
    CtVolume vol;
    try {
        vol = load_volume(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const DrrGeometry geo = make_demo_geometry(vol.desc, DRR_WIDTH, DRR_HEIGHT, STEP_MM);

    // ---- 2a. CPU reference DRR (timed) -------------------------------------
    std::vector<float> img_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    render_drr_cpu(vol, geo, img_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 2b. GPU DRR (kernel timed) ----------------------------------------
    std::vector<float> img_gpu;
    float gpu_kernel_ms = 0.0f;
    render_drr_gpu(vol, geo, img_gpu, &gpu_kernel_ms);

    // ---- 3. Verify ---------------------------------------------------------
    const double err = util::max_abs_err(img_cpu, img_gpu);
    const bool pass = err <= TOLERANCE;

    // ---- 4a. Deterministic report -> STDOUT --------------------------------
    // Find the brightest (most attenuating) pixel from the GPU image -- for our
    // synthetic phantom this should land on the dense "bone" sphere, a result the
    // learner can sanity-check against the geometry.
    const int W = geo.width, H = geo.height;
    float vmax = img_gpu[0]; int amax = 0;
    for (std::size_t i = 1; i < img_gpu.size(); ++i)
        if (img_gpu[i] > vmax) { vmax = img_gpu[i]; amax = static_cast<int>(i); }
    const int cu = W / 2, cv = H / 2;                 // center pixel coords
    const int cidx = cv * W + cu;                     // center pixel index

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("CT volume: %dx%dx%d voxels, spacing %.2fx%.2fx%.2f mm\n",
                vol.desc.nx, vol.desc.ny, vol.desc.nz,
                vol.desc.sx, vol.desc.sy, vol.desc.sz);
    std::printf("DRR detector: %dx%d pixels, ray step %.2f mm (cone-beam, lateral view)\n",
                W, H, geo.step_mm);
    std::printf("center pixel attenuation = %.4f\n", img_gpu[cidx]);
    std::printf("max attenuation = %.4f at (u,v)=(%d,%d)\n", vmax, amax % W, amax / W);
    std::printf("central row profile (8 samples):");
    for (int s = 0; s < 8; ++s) {
        const int u = (s * (W - 1)) / 7;              // 8 evenly spaced columns
        std::printf(" %.4f", img_gpu[cv * W + u]);
    }
    std::printf("\n");
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-03)\n", pass ? "PASS" : "FAIL");

    // ---- 4b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%dx%dx%d volume -> %dx%d DRR)\n",
                 path.c_str(), vol.desc.nx, vol.desc.ny, vol.desc.nz, W, H);
    std::fprintf(stderr, "[timing] CPU render: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- the GPU's edge grows with panel size, "
                         "ray length, and the 50-200 DRRs per registration iteration.\n");
    std::fprintf(stderr, "[verify] max_abs_err = %.3e  (tolerance %.1e)\n", err, TOLERANCE);

    return pass ? 0 : 1;
}
