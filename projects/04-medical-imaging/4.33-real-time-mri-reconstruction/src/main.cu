// ===========================================================================
// src/main.cu  --  Entry point: load radial k-space, run CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 4.33 : Real-Time MRI Reconstruction
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the golden-angle radial acquisition (data/sample).
//   2. CPU reference: reconstruct every sliding-window frame (reference_cpu.cpp).
//   3. GPU: reconstruct the same frames with the gridding scatter + cuFFT (kernels.cu).
//   4. VERIFY: assert the GPU movie agrees with the CPU movie within tolerance,
//      AND (the real science) that gridding recovers the synthetic ground truth.
//   5. REPORT: a deterministic summary to stdout; timing to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Every printed number is derived from the CPU path
//   (which is fully deterministic); run-to-run timings go to STDERR (shown, not
//   diffed). See PATTERNS.md section 3 on the stdout/stderr split.
//
// READ THIS FIRST in the code tour, then kernels.cuh -> kernels.cu (the gridding
// scatter + cuFFT), reference_cpu.cpp (the CPU twin), grid_core.h (the shared math).
// See ../THEORY.md for the "why".
// ===========================================================================
#include <algorithm>  // std::copy (assembling the CPU movie frame by frame)
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // reconstruct_frames_gpu (GPU path)
#include "reference_cpu.h"    // load_radial, reconstruct_frame_cpu
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "4.33";
static const char* PROJECT_NAME = "Real-Time MRI Reconstruction";

// Correctness tolerance for the GPU-vs-CPU MAGNITUDE image comparison.
//   The gridding scatter is FIXED-POINT (integer, associative), so the gridded grid
//   is bit-identical on both sides. The remaining difference is our radix-2 FFT vs
//   cuFFT (both FP32) plus the reciprocal-deapodization rounding; over one inverse
//   FFT these diverge by a small, physically-negligible amount, so we verify to a
//   relative-scaled absolute tolerance rather than pretending bit-identity
//   (PATTERNS.md section 4, the "single FFT" case).
static constexpr double TOL_ABS = 1.0e-4;   // absolute floor
static constexpr double TOL_REL = 1.0e-4;   // fraction of the movie's peak magnitude

// rms_diff: RMS of the pixelwise difference of two equal-size vectors.
static double rms_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double dd = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        s += dd * dd;
    }
    return std::sqrt(s / static_cast<double>(a.size()));
}

// max_abs: peak magnitude of a vector (for the relative tolerance and reporting).
static double max_abs(const std::vector<float>& v) {
    double m = 0.0;
    for (float x : v) { const double a = std::fabs(x); if (a > m) m = a; }
    return m;
}

// normalized_correlation: cosine similarity between two images (both flattened).
//   1.0 = identical up to scale. We use it to score the reconstruction against the
//   synthetic ground truth WITHOUT caring about the (arbitrary) overall brightness
//   scale that gridding + density-comp leave undetermined. This is the "did we
//   recover the anatomy?" metric, robust and deterministic.
static double normalized_correlation(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        na  += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        nb  += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    const double den = std::sqrt(na) * std::sqrt(nb);
    return den > 0.0 ? dot / den : 0.0;
}

int main(int argc, char** argv) {
    // ---- 1. Load the radial acquisition -----------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/radial_sample.txt";
    RadialData d;
    try {
        d = load_radial(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int total = d.n * d.n;

    // ---- 2. CPU reference: reconstruct every sliding-window frame (timed) --
    std::vector<float> movie_cpu(static_cast<std::size_t>(d.n_frames) * total);
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    for (int f = 0; f < d.n_frames; ++f) {
        std::vector<float> frame;
        reconstruct_frame_cpu(d, f * d.stride, frame);
        std::copy(frame.begin(), frame.end(),
                  movie_cpu.begin() + static_cast<std::size_t>(f) * total);
    }
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU: the same movie via the gridding scatter + cuFFT ----------
    std::vector<float> movie_gpu;
    float gpu_ms = 0.0f;
    reconstruct_frames_gpu(d, movie_gpu, &gpu_ms);

    // ---- 4. Verify --------------------------------------------------------
    // (a) GPU agrees with CPU across the whole movie (portability/correctness).
    const double peak = max_abs(movie_cpu);
    const double gpu_cpu_rms = rms_diff(movie_cpu, movie_gpu);
    const double tol = TOL_ABS + TOL_REL * peak;
    const bool gpu_ok = gpu_cpu_rms <= tol;

    // (b) The science: the LAST frame's reconstruction correlates with the truth
    //     image (gridding actually recovered the anatomy from sparse radial data).
    bool recon_ok = true;
    double corr = 0.0;
    if (d.has_truth) {
        std::vector<float> last_frame(movie_cpu.end() - total, movie_cpu.end());
        corr = normalized_correlation(last_frame, d.truth);
        // Threshold 0.85: this reduced-scope gridding recon (no oversampling, simple
        // |k| density comp, ~21 spokes) reaches ~0.96 correlation with the phantom;
        // 0.85 leaves honest margin for the streaking/time-mixing this tiny setup has
        // (THEORY "verify correctness" / "real world"). It confirms the anatomy is
        // recovered without pretending the image is artifact-free.
        recon_ok = corr > 0.85;
    }
    const bool pass = gpu_ok && recon_ok;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) ----------
    // Every value below comes from the DETERMINISTIC CPU path (movie_cpu / truth),
    // so stdout is byte-identical every run regardless of GPU thread ordering.
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("golden-angle radial NUFFT recon (single slice, single coil), gridding + cuFFT\n");
    std::printf("grid: %dx%d   spokes: %d x %d readout   KB width: %d\n",
                d.n, d.n, d.n_spokes, d.n_ro, d.kb_w);
    std::printf("sliding window: %d spokes/frame, stride %d, %d frames\n",
                d.win, d.stride, d.n_frames);
    // A deterministic per-frame fingerprint. We report each frame's PEAK location and
    // its brightness NORMALIZED to the movie's peak (so the numbers are O(1) and easy
    // to read regardless of the arbitrary overall gain that gridding + density
    // compensation leave undetermined). As the sliding window advances over spokes
    // acquired at different times, the moving blob shifts, so the peak LOCATION
    // changes frame to frame -- proving the reconstruction is a genuine dynamic movie.
    // Every value is from the deterministic CPU path, so stdout is byte-identical run
    // to run (timings, which vary, go to stderr -- PATTERNS.md section 3).
    const double gain = peak > 0.0 ? 1.0 / peak : 1.0;   // normalize display to peak=1
    std::printf("recon movie: normalized to peak=1.000 (raw peak %.3e, arbitrary MR units)\n", peak);
    std::printf("per-frame peak (normalized) @ (row,col):\n");
    const int show = d.n_frames < 6 ? d.n_frames : 6;
    for (int f = 0; f < show; ++f) {
        const float* fr = movie_cpu.data() + static_cast<std::size_t>(f) * total;
        float vmax = fr[0]; int amax = 0;
        for (int i = 1; i < total; ++i) if (fr[i] > vmax) { vmax = fr[i]; amax = i; }
        std::printf("  frame %d: %.4f @ (%d,%d)\n", f, vmax * gain, amax / d.n, amax % d.n);
    }
    if (d.has_truth)
        std::printf("last-frame vs truth: normalized correlation = %.4f\n", corr);
    std::printf("RESULT: %s (GPU gridding+cuFFT matches CPU within tol; recon recovers truth)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) -----------------
    std::fprintf(stderr, "[data]   source: %s  (%dx%d grid, %d spokes, %d frames)\n",
                 path.c_str(), d.n, d.n, d.n_spokes, d.n_frames);
    std::fprintf(stderr, "[timing] CPU movie: %.3f ms   GPU movie (gridding+cuFFT): %.3f ms\n",
                 cpu_ms, gpu_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- on this tiny slice the per-frame kernels "
                         "are launch-bound; the GPU's edge grows with grid size, spokes, and coils, "
                         "and a real system pipelines frames with acquisition via CUDA streams.\n");
    std::fprintf(stderr, "[verify] GPU-vs-CPU movie RMS diff = %.6e  (tolerance %.6e)\n",
                 gpu_cpu_rms, tol);

    // Exit code feeds the demo's pass/fail gate.
    return pass ? 0 : 1;
}
