// ===========================================================================
// src/main.cu  --  Entry point: align -> ramp-filter -> back-project -> verify
// ---------------------------------------------------------------------------
// Project 2.31 : Cryo-EM Tilt-Series Alignment & Tomogram Reconstruction
//
// THE 5-STEP SHAPE every project in this repo follows, specialized to cryo-ET:
//   1. Load the tilt series + geometry (data/sample).
//   2. ALIGN the tilt series: estimate per-projection integer shifts by cross-
//      correlation to the reference, then shift-correct (reference_cpu.cpp).
//   3a. RAMP FILTER the aligned projections two ways:
//        - GPU via cuFFT (ramp_filter_gpu)  <- the cuFFT teaching point,
//        - CPU via spatial convolution (ramp_filter_cpu) <- the baseline.
//       We VERIFY the two filtered sinograms agree on the field of view.
//   3b. BACK-PROJECT the (GPU-)filtered sinogram two ways:
//        - CPU reference (backproject_cpu),
//        - GPU gather kernel (backproject_gpu).
//       We VERIFY the two reconstructions agree (the headline correctness gate).
//   4. REPORT deterministic results -> stdout; timings + run-varying detail
//      -> stderr (so demo/run_demo can diff stdout against expected_output.txt).
//
//   WHY both back-projections consume the SAME (GPU) filtered sinogram: the FFT
//   ramp (cuFFT, periodic convolution) and the spatial ramp (CPU, linear
//   convolution) differ slightly near the projection EDGES. Feeding both gathers
//   the identical filtered data makes the back-projection comparison a TIGHT
//   CPU-vs-GPU check of the gather alone; the ramp filters are compared
//   separately with their own, documented, looser tolerance.
//
// READ THIS FIRST in the code tour, then reference_cpu.h (science) ->
// wbp_core.h (shared math) -> kernels.cuh -> kernels.cu. See ../THEORY.md.
// ===========================================================================
#include <cstdio>
#include <cmath>        // std::fabs (interior ramp comparison)
#include <string>
#include <vector>

#include "kernels.cuh"        // ramp_filter_gpu, backproject_gpu
#include "reference_cpu.h"    // load_tilt_series, estimate_shifts, apply_shifts,
                              // ramp_filter_cpu, compute_trig, backproject_cpu
#include "util/io.hpp"        // util::CpuTimer, util::max_abs_err

static const char* PROJECT_ID   = "2.31";
static const char* PROJECT_NAME = "Cryo-EM Tilt-Series Alignment & Tomogram Reconstruction";

// Back-projection sums many filtered samples; with the SAME filtered input the
// CPU and GPU differ only by float rounding / FMA contraction order, so a small
// absolute tolerance is appropriate (PATTERNS.md sec.4, "long iterative-ish").
static constexpr double RECON_TOL = 1.0e-3;
// The two ramp filters (FFT periodic vs spatial linear convolution) agree in the
// interior but diverge near row edges; we verify on the INTERIOR detector bins
// and allow a looser tolerance there. Both numbers are documented in THEORY.md.
static constexpr double RAMP_TOL  = 5.0e-2;
// Alignment search window (+- bins) for the cross-correlation shift estimate.
static constexpr int    SEARCH    = 8;

// Compare two ramp-filtered sinograms ONLY over the interior detector bins
// (drop the outermost `margin` bins of every row, where FFT wrap-around vs
// zero-padded convolution legitimately disagree). Returns the worst |diff|.
static double interior_ramp_err(const TiltSeries& ts,
                                const std::vector<float>& a,
                                const std::vector<float>& b, int margin) {
    const int n = ts.n_det;
    double worst = 0.0;
    for (int k = 0; k < ts.n_tilts; ++k) {
        for (int j = margin; j < n - margin; ++j) {
            const std::size_t idx = static_cast<std::size_t>(k) * n + j;
            const double d = std::fabs(static_cast<double>(a[idx]) -
                                       static_cast<double>(b[idx]));
            if (d > worst) worst = d;
        }
    }
    return worst;
}

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1]
                                        : "data/sample/tilt_series_sample.txt";
    TiltSeries ts;
    try {
        ts = load_tilt_series(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. Tilt-series alignment -----------------------------------------
    std::vector<int> shift;
    const int ref = estimate_shifts(ts, SEARCH, shift);
    std::vector<float> aligned;
    apply_shifts(ts, shift, aligned);

    // ---- 3a. Ramp filter: GPU (cuFFT) and CPU (spatial), then verify ------
    std::vector<float> filt_gpu, filt_cpu;
    float ramp_ms = 0.0f;
    ramp_filter_gpu(ts, aligned, filt_gpu, &ramp_ms);
    ramp_filter_cpu(ts, aligned, filt_cpu);
    // Compare on the interior (margin = 1/8 of the detector width, min 2 bins).
    const int margin = (ts.n_det / 8 > 2) ? ts.n_det / 8 : 2;
    const double ramp_err = interior_ramp_err(ts, filt_gpu, filt_cpu, margin);
    const bool ramp_ok = ramp_err <= RAMP_TOL;

    // ---- 3b. Back-project the GPU-filtered sinogram: CPU + GPU, verify ----
    std::vector<float> cosv, sinv;
    compute_trig(ts, cosv, sinv);

    std::vector<float> slice_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    backproject_cpu(ts, filt_gpu, cosv, sinv, slice_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    std::vector<float> slice_gpu;
    float bp_ms = 0.0f;
    backproject_gpu(ts, filt_gpu, cosv, sinv, slice_gpu, &bp_ms);

    const double recon_err = util::max_abs_err(slice_cpu, slice_gpu);
    const bool recon_ok = recon_err <= RECON_TOL;
    const bool pass = recon_ok && ramp_ok;

    // ---- 4a. Deterministic report -> STDOUT (diffed by the demo) ----------
    const int N = ts.img;
    const int cpix = (N / 2) * N + (N / 2);          // center pixel index
    // Brightest reconstructed pixel: recovers the embedded dense feature (the
    // synthetic sample places a bright disc at the slice center -- see data/).
    float vmax = slice_gpu[0]; int amax = 0;
    for (std::size_t i = 1; i < slice_gpu.size(); ++i)
        if (slice_gpu[i] > vmax) { vmax = slice_gpu[i]; amax = static_cast<int>(i); }

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("tilt series: %d projections x %d detector bins -> %dx%d slice\n",
                ts.n_tilts, ts.n_det, N, N);
    std::printf("tilt range: %.1f deg to %.1f deg (reference proj #%d at %.1f deg)\n",
                ts.tilt.front(), ts.tilt.back(), ref, ts.tilt[ref]);
    // Alignment shifts are INTEGER bins -> exact + reproducible on stdout.
    std::printf("estimated shifts (bins):");
    for (int k = 0; k < ts.n_tilts; ++k) std::printf(" %+d", shift[k]);
    std::printf("\n");
    std::printf("ramp filter: cuFFT (R2C -> |f| ramp -> C2R), Hann-apodized\n");
    std::printf("center pixel value = %.4f\n", slice_gpu[cpix]);
    std::printf("max reconstructed value = %.4f at (px,py)=(%d,%d)\n",
                vmax, amax % N, amax / N);
    std::printf("central row profile (8 samples):");
    for (int s = 0; s < 8; ++s) {
        const int px = (s * (N - 1)) / 7;            // 8 evenly spaced columns
        std::printf(" %.4f", slice_gpu[(N / 2) * N + px]);
    }
    std::printf("\n");
    std::printf("RESULT: %s (GPU back-projection matches CPU within tol=1.0e-03;\n",
                pass ? "PASS" : "FAIL");
    std::printf("        cuFFT ramp matches CPU ramp on interior within tol=5.0e-02)\n");

    // ---- 4b. Varying detail -> STDERR (shown, not diffed) -----------------
    std::fprintf(stderr, "[data]   source: %s  (%d tilts x %d bins -> %dx%d slice)\n",
                 path.c_str(), ts.n_tilts, ts.n_det, N, N);
    std::fprintf(stderr, "[align]  reference proj #%d (tilt %.1f deg); search window +-%d bins\n",
                 ref, ts.tilt[ref], SEARCH);
    std::fprintf(stderr, "[timing] cuFFT ramp filter: %.3f ms\n", ramp_ms);
    std::fprintf(stderr, "[timing] CPU back-project: %.3f ms   GPU back-project: %.3f ms\n",
                 cpu_ms, bp_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- the GPU's edge grows with slice size "
                         "and tilt count (real tomograms are far larger; 3-D = a stack of these).\n");
    std::fprintf(stderr, "[verify] recon max_abs_err = %.3e (tol %.1e) | "
                         "ramp interior err = %.3e (tol %.1e)\n",
                 recon_err, RECON_TOL, ramp_err, RAMP_TOL);

    return pass ? 0 : 1;
}
