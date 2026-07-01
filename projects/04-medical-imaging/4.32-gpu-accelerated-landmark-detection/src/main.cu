// ===========================================================================
// src/main.cu  --  Entry point: load heatmaps, decode on CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 4.32 : GPU-Accelerated Landmark Detection
//
// THE 5-STEP SHAPE (every project in this repo follows it)
//   1. Load the predicted heatmaps (data/sample), or fail loudly.
//   2. CPU reference decode (reference_cpu.cpp)        -> trusted landmarks.
//   3. GPU decode           (kernels.cu)               -> the thing taught.
//   4. VERIFY: the GPU landmarks match the CPU ones (integer peaks EXACTLY;
//      sub-voxel centroids within a tiny tolerance from one double division).
//   5. REPORT: deterministic recovered coordinates -> stdout; timing -> stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Timings (which vary run-to-run) go to STDERR,
//   which the demo shows but does not diff.
//
// WHAT THE NUMBERS MEAN
//   Each heatmap was BUILT around a known ground-truth point (a Gaussian blob,
//   see scripts/make_synthetic.py). So besides "does the GPU match the CPU?" we
//   can also ask "did the decoder recover the planted point?" -- the reported
//   decode error validates the SCIENCE, not just CPU==GPU agreement.
//
// Code tour: read this first, then landmark.h (the shared math), kernels.cuh ->
// kernels.cu (the GPU decode), and reference_cpu.cpp (the baseline).
// ===========================================================================
#include <cmath>     // std::fabs, std::sqrt
#include <cstdio>    // std::printf, std::fprintf
#include <string>
#include <vector>

#include "kernels.cuh"        // decode_gpu, GpuLandmark
#include "reference_cpu.h"    // load_heatmaps, decode_cpu, HeatmapSet
#include "landmark.h"         // Landmark, VolumeDims
#include "util/io.hpp"        // util::CpuTimer

// Identify the program. Kept in sync with demo/expected_output.txt.
static const char* PROJECT_ID   = "4.32";
static const char* PROJECT_NAME = "GPU-Accelerated Landmark Detection";

// Verification tolerance for the SUB-VOXEL centroid. The integer argmax and the
// integer weight sums are bit-identical CPU vs GPU; only the final
// double-precision division (num/den) can differ, and only in its last ulps.
// 1e-9 is far tighter than any voxel spacing yet safely above that ulp noise.
// (docs/PATTERNS.md section 4: honest, documented tolerance.)
static constexpr double TOLERANCE = 1.0e-9;

int main(int argc, char** argv) {
    // ---- 1. Load ----------------------------------------------------------
    const std::string path =
        (argc > 1) ? argv[1] : "data/sample/heatmaps_sample.txt";
    HeatmapSet hs;
    try {
        hs = load_heatmaps(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference decode (timed) ----------------------------------
    std::vector<Landmark> lm_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    decode_cpu(hs, lm_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU decode (kernel timed inside the wrapper) ------------------
    std::vector<Landmark> lm_gpu;
    float gpu_kernel_ms = 0.0f;
    decode_gpu(hs, lm_gpu, &gpu_kernel_ms);

    // ---- 4. Verify --------------------------------------------------------
    // (a) integer argmax peaks must match EXACTLY (same tie-break both sides).
    // (b) sub-voxel centroids must match within TOLERANCE.
    int   peak_mismatch = 0;
    double max_coord_diff = 0.0;
    for (int l = 0; l < hs.num_landmarks; ++l) {
        if (lm_cpu[l].px != lm_gpu[l].px ||
            lm_cpu[l].py != lm_gpu[l].py ||
            lm_cpu[l].pz != lm_gpu[l].pz) ++peak_mismatch;
        max_coord_diff = std::fmax(max_coord_diff, std::fabs(lm_cpu[l].x - lm_gpu[l].x));
        max_coord_diff = std::fmax(max_coord_diff, std::fabs(lm_cpu[l].y - lm_gpu[l].y));
        max_coord_diff = std::fmax(max_coord_diff, std::fabs(lm_cpu[l].z - lm_gpu[l].z));
    }
    const bool pass = (peak_mismatch == 0) && (max_coord_diff <= TOLERANCE);

    // Recovery error vs the planted ground truth (science check, from the GPU
    // result). We report the WORST landmark's Euclidean error, in voxels.
    double worst_recovery = 0.0;
    const bool have_truth = !hs.truth_x.empty();
    if (have_truth) {
        for (int l = 0; l < hs.num_landmarks; ++l) {
            double dx = lm_gpu[l].x - hs.truth_x[l];
            double dy = lm_gpu[l].y - hs.truth_y[l];
            double dz = lm_gpu[l].z - hs.truth_z[l];
            worst_recovery = std::fmax(worst_recovery, std::sqrt(dx*dx + dy*dy + dz*dz));
        }
    }

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) ----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("heatmap decode: %d landmarks over a %dx%dx%d voxel grid "
                "(argmax + soft-argmax, radius %d)\n",
                hs.num_landmarks, hs.dims.nx, hs.dims.ny, hs.dims.nz,
                SOFTARGMAX_RADIUS);
    for (int l = 0; l < hs.num_landmarks; ++l) {
        // Print the GPU landmark: integer peak voxel, sub-voxel coordinate, and
        // (if known) the recovery error vs the planted point. %.4f is stable
        // across platforms for these small magnitudes.
        std::printf("  L%02d: peak(%3d,%3d,%3d) val=%.4f  ->  coord=(%8.4f,%8.4f,%8.4f)",
                    l, lm_gpu[l].px, lm_gpu[l].py, lm_gpu[l].pz, lm_gpu[l].peak,
                    lm_gpu[l].x, lm_gpu[l].y, lm_gpu[l].z);
        if (have_truth) {
            double dx = lm_gpu[l].x - hs.truth_x[l];
            double dy = lm_gpu[l].y - hs.truth_y[l];
            double dz = lm_gpu[l].z - hs.truth_z[l];
            std::printf("  err=%.4f", std::sqrt(dx*dx + dy*dy + dz*dz));
        }
        std::printf("\n");
    }
    if (have_truth)
        std::printf("worst recovery error = %.4f voxels\n", worst_recovery);
    std::printf("RESULT: %s (GPU landmarks match CPU: peaks exact, "
                "coords within tol=1.0e-09)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) -----------------
    std::fprintf(stderr, "[data]   source: %s  (%d landmarks, %dx%dx%d grid, "
                         "%lld voxels/heatmap)\n",
                 path.c_str(), hs.num_landmarks, hs.dims.nx, hs.dims.ny, hs.dims.nz,
                 static_cast<long long>(volume_voxels(hs.dims)));
    std::fprintf(stderr, "[timing] CPU decode: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this sample is tiny; the "
                         "GPU's edge grows with voxels/heatmap (real 512^3 ~ 1.3e8 each).\n");
    std::fprintf(stderr, "[verify] peak mismatches = %d, max coord diff = %.3e "
                         "(tolerance %.1e)\n", peak_mismatch, max_coord_diff, TOLERANCE);

    // Exit code feeds the demo's pass/fail gate.
    return pass ? 0 : 1;
}
