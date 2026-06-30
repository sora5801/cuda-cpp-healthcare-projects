// ===========================================================================
// src/main.cu  --  Entry point: load volume, render CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 4.31 : Virtual Colonoscopy & CT Colonography
//
// 5-step shape (same skeleton as every project in this repo):
//   1. Load the synthetic CT volume + camera (data/sample).
//   2a. CPU reference render (reference_cpu.cpp)  -> img_cpu.
//   2b. GPU render (kernels.cu)                   -> img_gpu.
//   3. VERIFY: the GPU image matches the CPU image within tolerance.
//   4. REPORT (deterministic) -> STDOUT: image stats, an ASCII preview of the
//      fly-through frame, and the brightness over the known POLYP region (the
//      "known answer" the synthetic scene embeds).
//   5. REPORT (varying) -> STDERR: timings and the numeric max error.
//
// stdout is byte-identical every run (it is diffed against expected_output.txt);
// timings, which vary, go to stderr (PATTERNS.md §3).
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // render_gpu, Scene, Camera
#include "reference_cpu.h"    // load_scene, render_cpu
#include "util/io.hpp"        // util::CpuTimer, util::max_abs_err

static const char* PROJECT_ID   = "4.31";
static const char* PROJECT_NAME = "Virtual Colonoscopy & CT Colonography";

// The CPU and GPU run the IDENTICAL shared cast_ray() math in the same order, so
// they differ only by the compiler's freedom to fuse multiply-adds (FMA) and
// reorder associative float ops between host and device. Over a ray of a few
// hundred FP32 samples that drift stays tiny; 1e-3 in [0,1] shading units is a
// safe, honest bound (PATTERNS.md §4). We also confirm it is usually far smaller
// via the stderr max-error line.
static constexpr double TOLERANCE = 1.0e-3;

// ---------------------------------------------------------------------------
// ascii_shade(): map a [0,1] intensity to one of a few ASCII "brightness"
//   characters so we can print a tiny, deterministic PREVIEW of the rendered
//   fly-through frame to stdout. Dark (background / far wall) -> ' '/'.'; bright
//   (lit wall facing the headlamp) -> '#'. Purely for human legibility; the
//   numeric checks below are what actually verify correctness.
// ---------------------------------------------------------------------------
static char ascii_shade(float v) {
    const char* ramp = " .:-=+*#%@";   // 10 levels, dark -> bright
    int idx = (int)(v * 9.0f + 0.5f);  // round to nearest level
    if (idx < 0) idx = 0;
    if (idx > 9) idx = 9;
    return ramp[idx];
}

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/colon_volume_sample.txt";
    Scene scene;
    try {
        scene = load_scene(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int W = scene.width, H = scene.height;

    // ---- 2a. CPU reference render (timed) ----------------------------------
    std::vector<float> img_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    render_cpu(scene, img_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 2b. GPU render (kernel timed) -------------------------------------
    std::vector<float> img_gpu;
    float gpu_kernel_ms = 0.0f;
    render_gpu(scene, img_gpu, &gpu_kernel_ms);

    // ---- 3. Verify ---------------------------------------------------------
    const double err = util::max_abs_err(img_cpu, img_gpu);
    const bool pass = err <= TOLERANCE;

    // ---- 4. Deterministic report -> STDOUT ---------------------------------
    // Image statistics computed on the GPU image (which equals the CPU image
    // within tolerance). All integers / rounded floats -> byte-stable output.
    int   wall_hits = 0;          // pixels that hit a wall (intensity > 0)
    double sum = 0.0;             // for the mean intensity
    float vmax = img_gpu[0]; int amax = 0;
    for (size_t i = 0; i < img_gpu.size(); ++i) {
        float v = img_gpu[i];
        sum += v;
        if (v > 0.0f) ++wall_hits;
        if (v > vmax) { vmax = v; amax = (int)i; }
    }
    const double wall_frac = (double)wall_hits / img_gpu.size();
    const double mean_int  = sum / img_gpu.size();

    // The synthetic scene places a polyp bump on the -y lumen wall. With the
    // camera's up = +y, that wall projects into the UPPER-CENTER of the frame.
    // We report the mean brightness of that fixed window: the polyp's convex
    // surface catches the headlamp and reads ~2x BRIGHTER than the smooth wall
    // there (measured: ~0.73 with polyp vs ~0.36 without) -- the recoverable
    // "known answer" the synthetic scene embeds (THEORY §1, §6).
    const int rx0 = (int)(W * 0.42f), rx1 = (int)(W * 0.62f);  // center columns
    const int ry0 = (int)(H * 0.25f), ry1 = (int)(H * 0.45f);  // upper-middle rows
    double polyp_sum = 0.0; int polyp_n = 0;
    for (int y = ry0; y < ry1; ++y)
        for (int x = rx0; x < rx1; ++x) { polyp_sum += img_gpu[(size_t)y * W + x]; ++polyp_n; }
    const double polyp_mean = polyp_n ? polyp_sum / polyp_n : 0.0;

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("CT colonography fly-through: volume %dx%dx%d -> frame %dx%d (iso=%.2f)\n",
                scene.nx, scene.ny, scene.nz, W, H, scene.iso);
    std::printf("wall-hit pixels = %d / %d (%.3f)\n", wall_hits, (int)img_gpu.size(), wall_frac);
    std::printf("mean intensity = %.4f\n", mean_int);
    std::printf("max intensity = %.4f at (px,py)=(%d,%d)\n", vmax, amax % W, amax / W);
    std::printf("polyp-region mean brightness = %.4f\n", polyp_mean);

    // A small deterministic ASCII preview: downsample the frame to ~24 columns so
    // the learner SEES the lumen (the round wall around a dark center, with the
    // polyp brightening the lower lip). Rows/cols are integer-strided -> stable.
    const int PW = 24, PH = 12;
    std::printf("ascii preview (%dx%d, '@'=bright wall, ' '=dark lumen/background):\n", PW, PH);
    for (int r = 0; r < PH; ++r) {
        int y = (r * (H - 1)) / (PH - 1);
        std::printf("  |");
        for (int c = 0; c < PW; ++c) {
            int x = (c * (W - 1)) / (PW - 1);
            std::putchar(ascii_shade(img_gpu[(size_t)y * W + x]));
        }
        std::printf("|\n");
    }
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-03)\n", pass ? "PASS" : "FAIL");

    // ---- 5. Varying detail -> STDERR ---------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%dx%dx%d volume -> %dx%d frame)\n",
                 path.c_str(), scene.nx, scene.ny, scene.nz, W, H);
    std::fprintf(stderr, "[timing] CPU render: %.3f ms   GPU render (kernel): %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- the GPU's edge grows with frame size, "
                         "volume size and frame count (clinical fly-throughs are 512^3 at 60 FPS).\n");
    std::fprintf(stderr, "[verify] max_abs_err = %.3e  (tolerance %.1e)\n", err, TOLERANCE);

    return pass ? 0 : 1;
}
