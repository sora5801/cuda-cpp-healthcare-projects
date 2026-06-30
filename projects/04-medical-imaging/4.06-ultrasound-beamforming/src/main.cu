// ===========================================================================
// src/main.cu  --  Entry point: load RF data, beamform CPU+GPU, verify, report
// ---------------------------------------------------------------------------
// Project 4.6 : Ultrasound Beamforming (Delay-and-Sum)
//
// 5-step shape (the same skeleton every project in this repo uses):
//   1. Load the RF echo data + geometry (data/sample).
//   2a. CPU reference beamforming (reference_cpu.cpp -> beamform.h).
//   2b. GPU beamforming (kernels.cu -> the SAME beamform.h core).
//   3. VERIFY: the GPU image matches the CPU image within tolerance.
//   4. ENVELOPE + REPORT: take |coherent sum|, find the focal spot, print a
//      DETERMINISTIC summary to stdout; timing to stderr.
//
// The synthetic sample embeds ONE point scatterer at a known (x,z). A correct
// beamformer focuses all element echoes there, so the brightest pixel must land
// on that scatterer -- that recovered location is our human-readable proof the
// beamforming worked (not just that CPU==GPU). See data/README.md.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // beamform_gpu, BeamformGeom
#include "reference_cpu.h"    // load_beamform, beamform_cpu, BeamformProblem
#include "util/io.hpp"        // util::CpuTimer, util::max_abs_err

static const char* PROJECT_ID   = "4.6";
static const char* PROJECT_NAME = "Ultrasound Beamforming (Delay-and-Sum)";

// CPU and GPU run the IDENTICAL beamform.h formula, element by element, in the
// same order. The only possible divergence is the GPU contracting a*b+c into a
// fused multiply-add (FMA) where the host emits two rounded ops. Over a sum of
// up to a few hundred elements that is a tiny absolute difference, so we verify
// to a small absolute tolerance rather than claiming bit-identity (PATTERNS.md
// §4: "same exact operations" -> near machine precision).
static constexpr double TOLERANCE = 1.0e-3;

int main(int argc, char** argv) {
    // ---- 1. Load ----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/rf_sample.txt";
    BeamformProblem prob;
    try {
        prob = load_beamform(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const BeamformGeom& g = prob.geom;

    // ---- 2a. CPU reference (timed) ---------------------------------------
    std::vector<float> img_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    beamform_cpu(prob, img_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 2b. GPU beamforming (kernel timed) ------------------------------
    std::vector<float> img_gpu;
    float gpu_kernel_ms = 0.0f;
    beamform_gpu(prob, img_gpu, &gpu_kernel_ms);

    // ---- 3. Verify (on the raw signed coherent sums) ----------------------
    const double err  = util::max_abs_err(img_cpu, img_gpu);
    const bool   pass = err <= TOLERANCE;

    // ---- 4. Envelope + deterministic report -> STDOUT --------------------
    // B-mode brightness is the ENVELOPE (magnitude) of the coherent sum. We take
    // |.| of the GPU image and locate the brightest pixel: for a single point
    // scatterer that pixel is the focal spot and should sit on the scatterer.
    const int nx = g.nx, nz = g.nz;
    float vmax = 0.0f; int amax = 0;
    for (std::size_t i = 0; i < img_gpu.size(); ++i) {
        const float env = std::fabs(img_gpu[i]);
        if (env > vmax) { vmax = env; amax = static_cast<int>(i); }
    }
    const int max_ix = amax % nx;            // brightest pixel column (lateral)
    const int max_iz = amax / nx;            // brightest pixel row (depth)
    // World coordinates of the focal spot, in millimetres for readability.
    const float max_x_mm = 1000.0f * (g.x_min + max_ix * g.dx);
    const float max_z_mm = 1000.0f * (g.z_min + max_iz * g.dz);

    // A LATERAL profile of the envelope across the focal ROW (the depth where
    // the scatterer sits): 8 evenly spaced columns. This is the classic "beam
    // plot" -- a clear main lobe peaking at the scatterer column and falling off
    // to the sides, which is exactly what focusing buys you. (A depth profile
    // would be less illustrative because the *signed* coherent sum oscillates.)
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("DAS beamform: %d elements x %d RF samples -> %dx%d image (x by z)\n",
                g.n_elements, g.n_samples, nx, nz);
    std::printf("focal spot (brightest pixel): (ix,iz)=(%d,%d)  =  (x,z)=(%.1f,%.1f) mm\n",
                max_ix, max_iz, max_x_mm, max_z_mm);
    std::printf("peak envelope value = %.4f\n", vmax);
    std::printf("center pixel envelope = %.4f\n",
                std::fabs(img_gpu[(std::size_t)(nz / 2) * nx + (nx / 2)]));
    std::printf("lateral profile across focal row (8 samples):");
    for (int s = 0; s < 8; ++s) {
        const int ix = (s * (nx - 1)) / 7;                 // 8 evenly spaced cols
        const float env = std::fabs(img_gpu[(std::size_t)max_iz * nx + ix]);
        std::printf(" %.4f", env);
    }
    std::printf("\n");
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-03)\n",
                pass ? "PASS" : "FAIL");

    // ---- 4b. Varying detail -> STDERR ------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d elements x %d samples -> %dx%d image)\n",
                 path.c_str(), g.n_elements, g.n_samples, nx, nz);
    std::fprintf(stderr, "[geom]   fs=%.3g Hz  c=%.1f m/s  pitch=%.3g m  "
                         "grid origin=(%.4g,%.4g) m  dx=%.3g dz=%.3g m\n",
                 g.fs, g.c, g.pitch, g.x_min, g.z_min, g.dx, g.dz);
    std::fprintf(stderr, "[timing] CPU beamform: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- on this tiny grid the GPU may not win; "
                         "its edge grows with image size, element count, and frame rate.\n");
    std::fprintf(stderr, "[verify] max_abs_err = %.3e  (tolerance %.1e)\n", err, TOLERANCE);

    return pass ? 0 : 1;
}
