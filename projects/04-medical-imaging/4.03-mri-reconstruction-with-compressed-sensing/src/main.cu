// ===========================================================================
// src/main.cu  --  Entry point: load k-space, run CPU + GPU FISTA, verify, report
// ---------------------------------------------------------------------------
// Project 4.3 : MRI Reconstruction with Compressed Sensing
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the under-sampled k-space problem (data/sample).
//   2. CPU reference: zero-filled baseline + FISTA reconstruction (reference_cpu.cpp).
//   3. GPU: the same FISTA using cuFFT (kernels.cu).
//   4. VERIFY: assert the GPU image agrees with the CPU image within tolerance,
//      AND (the real science) that CS reduces error vs the zero-filled baseline.
//   5. REPORT: deterministic summary to stdout; timing to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Every printed number is derived from the CPU path
//   (which is fully deterministic); run-to-run timings go to STDERR (shown, not
//   diffed). See PATTERNS.md section 3 on the stdout/stderr split.
//
// READ THIS FIRST in the code tour, then kernels.cuh -> kernels.cu (the cuFFT
// FISTA), and reference_cpu.cpp for the baseline. See ../THEORY.md for the "why".
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // reconstruct_gpu (GPU path)
#include "reference_cpu.h"    // load_kspace, reconstruct_cpu, zero_filled_magnitude
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "4.3";
static const char* PROJECT_NAME = "MRI Reconstruction with Compressed Sensing";

// Correctness tolerance for the GPU-vs-CPU MAGNITUDE image comparison.
//   The GPU uses cuFFT and the CPU uses our radix-2 FFT; both are single precision
//   and both run the SAME FISTA arithmetic (cs_core.h). Over a few dozen iterations
//   the two FFT libraries' rounding diverges by a small, physically-negligible
//   amount, so we verify to a relative-scaled absolute tolerance rather than
//   pretending the results are bit-identical (PATTERNS.md section 4, iterative case).
static constexpr double TOL_ABS = 2.0e-3;   // absolute floor
static constexpr double TOL_REL = 1.0e-3;   // fraction of the image's peak magnitude

// rms: root-mean-square value of an image (a stable, order-independent scalar).
static double rms(const std::vector<float>& v) {
    double s = 0.0;
    for (float x : v) s += static_cast<double>(x) * static_cast<double>(x);
    return std::sqrt(s / static_cast<double>(v.size()));
}

// rms_diff: RMS of the pixelwise difference of two equal-size images.
static double rms_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double dd = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        s += dd * dd;
    }
    return std::sqrt(s / static_cast<double>(a.size()));
}

// max_abs: peak magnitude of an image (for the relative tolerance and reporting).
static double max_abs(const std::vector<float>& v) {
    double m = 0.0;
    for (float x : v) { const double a = std::fabs(x); if (a > m) m = a; }
    return m;
}

int main(int argc, char** argv) {
    // ---- 1. Load the under-sampled k-space problem ------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/kspace_sample.txt";
    KSpaceData d;
    try {
        d = load_kspace(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int total = d.n * d.n;
    int n_sampled = 0;
    for (int m : d.mask) n_sampled += m;
    const double sample_frac = static_cast<double>(n_sampled) / static_cast<double>(total);

    // ---- 2. CPU reference: baseline + FISTA (timed) -----------------------
    std::vector<float> zerofill_cpu, recon_cpu;
    zero_filled_magnitude(d, zerofill_cpu);          // naive "before" image
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    reconstruct_cpu(d, recon_cpu);                   // CS "after" image
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU: the same FISTA via cuFFT (loop timed inside the wrapper) --
    std::vector<float> recon_gpu;
    float gpu_ms = 0.0f;
    reconstruct_gpu(d, recon_gpu, &gpu_ms);

    // ---- 4. Verify --------------------------------------------------------
    // (a) GPU agrees with CPU (the portability/correctness check).
    const double peak = max_abs(recon_cpu);
    const double gpu_cpu_rms = rms_diff(recon_cpu, recon_gpu);
    const double tol = TOL_ABS + TOL_REL * peak;
    const bool gpu_ok = gpu_cpu_rms <= tol;

    // (b) CS actually helped: reconstruction error vs. ground truth is SMALLER
    //     than the zero-filled baseline's error (the science, if truth is present).
    bool cs_ok = true;
    double err_zf = 0.0, err_cs = 0.0;
    if (d.has_truth) {
        err_zf = rms_diff(zerofill_cpu, d.truth);    // aliased baseline error
        err_cs = rms_diff(recon_cpu,   d.truth);     // CS-reconstructed error
        cs_ok = err_cs < err_zf;                     // CS must beat zero-filling
    }
    const bool pass = gpu_ok && cs_ok;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) ----------
    // Every value below comes from the DETERMINISTIC CPU path (recon_cpu / truth),
    // so stdout is byte-identical every run regardless of GPU thread ordering.
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("under-sampled Cartesian CS-MRI (single slice, single coil), FISTA + cuFFT\n");
    std::printf("image: %dx%d   sampled k-space: %d/%d (%.1f%%)   lambda=%.4f   iters=%d\n",
                d.n, d.n, n_sampled, total, 100.0 * sample_frac, d.lambda, d.iters);
    std::printf("recon image RMS (CPU): %.6f   peak: %.6f\n", rms(recon_cpu), peak);
    if (d.has_truth) {
        std::printf("error vs truth (RMS): zero-filled=%.6f  CS-reconstructed=%.6f\n",
                    err_zf, err_cs);
        std::printf("CS improvement: %.2fx lower error than zero-filling\n",
                    err_cs > 0.0 ? err_zf / err_cs : 0.0);
    }
    std::printf("RESULT: %s (GPU cuFFT recon matches CPU FISTA within tol; CS beats zero-fill)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) -----------------
    std::fprintf(stderr, "[data]   source: %s  (%dx%d, %.1f%% sampled)\n",
                 path.c_str(), d.n, d.n, 100.0 * sample_frac);
    std::fprintf(stderr, "[timing] CPU FISTA: %.3f ms   GPU FISTA (cuFFT): %.3f ms\n",
                 cpu_ms, gpu_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- on this tiny slice the two per-iteration "
                         "FFTs are launch-bound; the GPU's edge grows with image size and coil count.\n");
    std::fprintf(stderr, "[verify] GPU-vs-CPU image RMS diff = %.6e  (tolerance %.6e)\n",
                 gpu_cpu_rms, tol);

    // Exit code feeds the demo's pass/fail gate.
    return pass ? 0 : 1;
}
