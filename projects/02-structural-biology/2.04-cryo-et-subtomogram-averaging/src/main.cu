// ===========================================================================
// src/main.cu  --  Entry point: load subtomograms, align (CPU+GPU), verify, report
// ---------------------------------------------------------------------------
// Project 2.4 : Cryo-ET Subtomogram Averaging  (reduced-scope teaching version)
//
// THE 5-STEP SHAPE every project in this repo follows:
//   1. Load the problem (a reference cube + N candidate cubes from data/sample).
//   2. CPU reference  (reference_cpu.cpp) : direct zero-shift NCC per angle, the
//      best angle per candidate, and the refined average -> trusted answers.
//   3. GPU search     (kernels.cu)        : cuFFT cross-correlation over ALL
//      shifts, peak + zero-shift NCC per (candidate, angle) -> the thing taught.
//   4. VERIFY: the GPU's zero-shift NCC matches the CPU's direct one (this is the
//      cross-correlation theorem, demonstrated numerically) within tolerance.
//   5. REPORT: deterministic per-candidate best angle + average summary -> stdout;
//      timing + verification error -> stderr.
//
//   STDOUT is byte-for-byte deterministic (diffed by demo/run_demo against
//   demo/expected_output.txt); run-to-run timings go to STDERR (PATTERNS.md §3).
//
// Code tour: start here, then kernels.cuh -> kernels.cu (the cuFFT pipeline),
// then reference_cpu.*. The science/GPU-mapping lives in ../THEORY.md.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // align_gpu, SubtomogramSet
#include "reference_cpu.h"    // load_subtomograms, correlate_cpu, build_average_cpu
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "2.4";
static const char* PROJECT_NAME = "Cryo-ET Subtomogram Averaging";

// Tolerance for "GPU zero-shift NCC == CPU zero-shift NCC". Both compute the
// same correlation sum, but by DIFFERENT routes: the CPU sums products in single
// precision directly; the GPU goes through a single-precision cuFFT round trip
// (forward R2C, complex multiply, inverse C2R, scaled by 1/V). The two therefore
// differ by floating-point rounding -- measured at only ~3e-7 on these 16^3
// cubes (the FFT's error grows ~sqrt(log V), very slowly), but we keep a roomy
// 1e-3 ceiling so the test stays robust across GPUs/arches (PATTERNS.md §4). The
// gap is a real, documented numerical effect, not a bug.
static constexpr double TOLERANCE = 1.0e-3;

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path =
        (argc > 1) ? argv[1] : "data/sample/subtomograms_sample.txt";
    SubtomogramSet set;
    try {
        set = load_subtomograms(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<double> ncc_cpu;     // [n_sub*n_angles] zero-shift NCC
    std::vector<int>    best_cpu;    // [n_sub] best angle per candidate
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    correlate_cpu(set, ncc_cpu, best_cpu);
    const double cpu_ms = cpu_timer.stop_ms();
    std::vector<float> avg_cpu;
    const double core_cpu = build_average_cpu(set, best_cpu, avg_cpu);

    // ---- 3. GPU search (timed inside the wrapper) -------------------------
    std::vector<double> ncc_zero_gpu, ncc_peak_gpu;
    std::vector<int>    best_gpu;
    float gpu_kernel_ms = 0.0f;
    align_gpu(set, ncc_zero_gpu, ncc_peak_gpu, best_gpu, &gpu_kernel_ms);
    // Build the GPU's refined average from ITS chosen angles (same routine).
    std::vector<float> avg_gpu;
    const double core_gpu = build_average_cpu(set, best_gpu, avg_gpu);

    // ---- 4. Verify ---------------------------------------------------------
    // (a) zero-shift NCC agrees element-by-element (the FFT identity), and
    // (b) both paths recover the same best angle per candidate.
    double worst = 0.0;
    for (std::size_t i = 0; i < ncc_cpu.size(); ++i) {
        const double diff = std::fabs(ncc_cpu[i] - ncc_zero_gpu[i]);
        if (diff > worst) worst = diff;
    }
    bool angles_match = (best_cpu == best_gpu);
    const bool pass = (worst <= TOLERANCE) && angles_match;

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("Subtomogram averaging: 1 reference vs %d candidates, "
                "%d^3 voxels, %d trial angles (cuFFT cross-correlation)\n",
                set.n_sub, set.d, set.n_angles);
    std::printf("per-candidate alignment (best angle index : peak NCC):\n");
    for (int s = 0; s < set.n_sub; ++s) {
        const int k = best_gpu[static_cast<std::size_t>(s)];
        const double peak = ncc_peak_gpu[static_cast<std::size_t>(s) * set.n_angles + k];
        const double deg = 360.0 * static_cast<double>(k) / static_cast<double>(set.n_angles);
        std::printf("  cand[%d]  angle[%d] = %6.1f deg   peak NCC = %.4f\n",
                    s, k, deg, peak);
    }
    std::printf("refined average core intensity (mean|voxel|) = %.6f\n", core_gpu);
    std::printf("RESULT: %s (GPU cuFFT correlation matches CPU direct, same poses)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d candidates, %d^3 voxels, %d angles)\n",
                 path.c_str(), set.n_sub, set.d, set.n_angles);
    std::fprintf(stderr, "[timing] CPU direct correlation: %.3f ms   GPU cuFFT pipeline: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- the CPU does O(V) per shift at ZERO shift "
                         "only; the GPU does ALL %d shifts via O(V log V) cuFFT. The GPU's edge "
                         "grows with cube size and shift count.\n", set.vol());
    std::fprintf(stderr, "[verify] worst |NCC_gpu - NCC_cpu| (zero shift) = %.3e  (tol %.1e)\n",
                 worst, TOLERANCE);
    std::fprintf(stderr, "[verify] best angles match: %s   CPU avg core = %.6f (GPU %.6f)\n",
                 angles_match ? "yes" : "NO", core_cpu, core_gpu);

    return pass ? 0 : 1;
}
