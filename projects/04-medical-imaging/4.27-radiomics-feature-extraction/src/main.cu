// ===========================================================================
// src/main.cu  --  Entry point: load ROI, run CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 4.27 : Radiomics Feature Extraction
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the ROI volume (data/sample): nx*ny*nz intensities + a 0/1 mask.
//   2. CPU reference feature extraction (reference_cpu.cpp)   -> trusted answer.
//   3. GPU feature extraction (kernels.cu, atomic GLCM)       -> the thing taught.
//   4. VERIFY: GLCM total matches EXACTLY (integers) and every feature agrees
//      within a tiny numeric tolerance.
//   5. REPORT: deterministic feature vector to stdout; timing to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Timings (which vary run to run) go to STDERR,
//   which the demo shows but does not diff.
//
// Code tour: start here, then radiomics.h (per-voxel math), kernels.cuh ->
//   kernels.cu (the atomic GLCM), and reference_cpu.cpp for the serial baseline.
//   See ../THEORY.md for the "why".
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>

#include "kernels.cuh"        // extract_features_gpu (GPU path)
#include "reference_cpu.h"    // Volume, Features, extract_features_cpu (CPU baseline)
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "4.27";
static const char* PROJECT_NAME = "Radiomics Feature Extraction";

// Correctness tolerance for the DERIVED features. The GLCM/histogram COUNTS are
// integers and must match exactly; the features involve log2/sqrt/divisions of
// those identical integers, so any difference is pure floating-point rounding of
// the same operations -> ~1e-12. We allow 1e-9 as comfortable slack.
// (docs/PATTERNS.md section 4: exact where integer, ~machine-eps where derived.)
static constexpr double TOLERANCE = 1.0e-9;

// Compare two feature bundles: the integer GLCM total must be identical, and
// every real-valued feature must agree within TOLERANCE. Returns the worst
// absolute feature difference via `worst` for the stderr diagnostic.
static bool features_agree(const Features& a, const Features& b, double& worst) {
    worst = 0.0;
    const double diffs[] = {
        a.mean - b.mean, a.variance - b.variance, a.energy - b.energy, a.entropy - b.entropy,
        a.glcm_contrast - b.glcm_contrast, a.glcm_energy - b.glcm_energy,
        a.glcm_homogeneity - b.glcm_homogeneity, a.glcm_correlation - b.glcm_correlation,
        a.glcm_entropy - b.glcm_entropy
    };
    for (double d : diffs) worst = std::fmax(worst, std::fabs(d));
    return (a.glcm_total == b.glcm_total) && (worst <= TOLERANCE);
}

int main(int argc, char** argv) {
    // ---- 1. Load the ROI volume --------------------------------------------
    const std::string path =
        (argc > 1) ? argv[1] : "data/sample/radiomics_sample.txt";
    Volume vol;
    try {
        vol = load_volume(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) ------------------------------------------
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    const Features f_cpu = extract_features_cpu(vol);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU result (kernels timed inside the wrapper) ------------------
    float gpu_kernel_ms = 0.0f;
    const Features f_gpu = extract_features_gpu(vol, &gpu_kernel_ms);

    // ---- 4. Verify ----------------------------------------------------------
    double worst = 0.0;
    const bool pass = features_agree(f_cpu, f_gpu, worst);

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    // We print the GPU features (identical to CPU on PASS) at fixed precision so
    // the output is byte-stable across runs and machines.
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("ROI: %d x %d x %d grid, %d masked voxels, %d gray levels\n",
                vol.nx, vol.ny, vol.nz, vol.nroi, vol.Ng);
    std::printf("intensity range [%.4f, %.4f]\n", vol.vmin, vol.vmax);
    std::printf("first-order:\n");
    std::printf("  mean        = %.6f\n", f_gpu.mean);
    std::printf("  variance    = %.6f\n", f_gpu.variance);
    std::printf("  energy      = %.6f\n", f_gpu.energy);
    std::printf("  entropy     = %.6f bits\n", f_gpu.entropy);
    std::printf("GLCM texture (13 directions, symmetric, %lld pairs):\n", f_gpu.glcm_total);
    std::printf("  contrast    = %.6f\n", f_gpu.glcm_contrast);
    std::printf("  energy(ASM) = %.6f\n", f_gpu.glcm_energy);
    std::printf("  homogeneity = %.6f\n", f_gpu.glcm_homogeneity);
    std::printf("  correlation = %.6f\n", f_gpu.glcm_correlation);
    std::printf("  entropy     = %.6f bits\n", f_gpu.glcm_entropy);
    std::printf("RESULT: %s (GPU features match CPU; GLCM counts identical)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s\n", path.c_str());
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernels: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny ROI is dominated by "
                         "launch/copy overhead; the GPU's edge grows with ~10^6-voxel ROIs.\n");
    std::fprintf(stderr, "[verify] GLCM total cpu/gpu = %lld / %lld ; worst feature diff = %.3e "
                         "(tolerance %.1e)\n",
                 f_cpu.glcm_total, f_gpu.glcm_total, worst, TOLERANCE);

    return pass ? 0 : 1;
}
