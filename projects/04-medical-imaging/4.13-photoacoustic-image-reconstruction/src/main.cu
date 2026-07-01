// ===========================================================================
// src/main.cu  --  Entry point: load traces, reconstruct on CPU + GPU, verify
// ---------------------------------------------------------------------------
// Project 4.13 : Photoacoustic Image Reconstruction
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the PA acquisition (sensor geometry + pressure traces) from
//      data/sample, or fail loudly if the file is missing.
//   2. Reconstruct with the CPU reference (reference_cpu.cpp) -> trusted answer.
//   3. Reconstruct on the GPU (kernels.cu)                     -> the thing taught.
//   4. VERIFY: assert the GPU image matches the CPU image within a tolerance.
//   5. REPORT: deterministic image samples to stdout; timing to stderr.
//
//   STDOUT is kept byte-for-byte deterministic so demo/run_demo can diff it
//   against demo/expected_output.txt. Timings (which vary run-to-run) go to
//   STDERR, which the demo shows but does not diff.
//
//   INTERPRETABILITY: the committed sample places a few KNOWN point absorbers in
//   the tissue (see scripts/make_synthetic.py). A correct reconstruction shows
//   bright peaks AT those locations, so we report the brightest pixel and check
//   it lands on the strongest planted absorber -- validating the science, not
//   just CPU==GPU agreement (PATTERNS.md §6).
//
// READ THIS FIRST in the code tour, then pa_core.h -> kernels.cuh -> kernels.cu,
// and reference_cpu.cpp for the baseline. See ../THEORY.md for the "why".
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // reconstruct_gpu (GPU path), PAProblem
#include "reference_cpu.h"    // load_pa, reconstruct_cpu (CPU baseline)
#include "util/io.hpp"        // util::CpuTimer, util::max_abs_err

// Identity tokens (kept in sync with demo/expected_output.txt).
static const char* PROJECT_ID   = "4.13";
static const char* PROJECT_NAME = "Photoacoustic Image Reconstruction";

// Correctness tolerance. The CPU and GPU run the IDENTICAL pa_pixel_das()
// (PATTERNS.md §2), but they still differ by a few 1e-5: nvcc CONTRACTS
// `a*b + c` into a single fused-multiply-add (FMA, one rounding) while the host
// compiler rounds twice, and the ~64-term delay-and-sum accumulates that tiny
// per-term difference. That is real and worth teaching (PATTERNS.md §4): with
// peak reconstructed values ~30, an absolute tolerance of 1e-3 is a physically
// negligible ~0.003%, far below any imaging-relevant difference. We verify to
// that tolerance rather than pretending the two are bit-identical.
static constexpr double TOLERANCE = 1.0e-3;

int main(int argc, char** argv) {
    // ---- 1. Load the acquisition -------------------------------------------
    const std::string path =
        (argc > 1) ? argv[1] : "data/sample/pa_sample.txt";
    PAProblem pa;
    try {
        pa = load_pa(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference reconstruction (timed) ---------------------------
    std::vector<float> img_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    reconstruct_cpu(pa, img_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU reconstruction (kernel timed inside the wrapper) -----------
    std::vector<float> img_gpu;
    float gpu_kernel_ms = 0.0f;
    reconstruct_gpu(pa, img_gpu, &gpu_kernel_ms);

    // ---- 4. Verify GPU vs CPU ----------------------------------------------
    const double err  = util::max_abs_err(img_cpu, img_gpu);
    const bool   pass = err <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    const int N = pa.img;
    // Find the brightest reconstructed pixel from the GPU image -- for point
    // absorbers this is the recovered source location. Deterministic scan: on a
    // tie we keep the FIRST (lowest linear index), so stdout never varies.
    float vmax = img_gpu[0];
    int   amax = 0;
    for (std::size_t i = 1; i < img_gpu.size(); ++i) {
        if (img_gpu[i] > vmax) { vmax = img_gpu[i]; amax = static_cast<int>(i); }
    }
    const int amax_px = amax % N;                 // column of the peak
    const int amax_py = amax / N;                 // row of the peak
    // Convert that peak pixel back to world coordinates (metres) so the learner
    // can compare against the planted absorber positions in make_synthetic.py.
    const float pix = (N > 1) ? (2.0f * pa.world_half / (N - 1)) : 0.0f;
    const float peak_x = -pa.world_half + amax_px * pix;
    const float peak_y = -pa.world_half + amax_py * pix;

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("2-D delay-and-sum backprojection\n");
    std::printf("%d sensors x %d samples -> %dx%d image (c=%.1f m/s, dt=%.3e s)\n",
                pa.n_sensors, pa.n_samples, N, N, pa.c, pa.dt);
    std::printf("peak value = %.4f at pixel (px,py)=(%d,%d) = (x,y)=(%.4f,%.4f) m\n",
                vmax, amax_px, amax_py, peak_x, peak_y);
    // A short profile across the image center row: shows the bright compact
    // sources sitting above a low background (the DAS point-spread response).
    std::printf("center-row profile (8 samples):");
    for (int s = 0; s < 8; ++s) {
        const int px = (s * (N - 1)) / 7;         // 8 evenly spaced columns
        std::printf(" %.4f", img_gpu[(N / 2) * N + px]);
    }
    std::printf("\n");
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-03)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%d sensors, %d samples, %dx%d image)\n",
                 path.c_str(), pa.n_sensors, pa.n_samples, N, N);
    std::fprintf(stderr, "[timing] CPU reconstruct: %.3f ms   GPU reconstruct: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- the GPU's edge grows with image "
                         "size and sensor count (clinical 3-D volumes are far larger).\n");
    std::fprintf(stderr, "[verify] max_abs_err = %.3e  (tolerance %.1e)\n", err, TOLERANCE);

    return pass ? 0 : 1;
}
