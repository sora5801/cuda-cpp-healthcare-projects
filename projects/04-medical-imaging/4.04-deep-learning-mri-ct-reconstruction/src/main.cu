// ===========================================================================
// src/main.cu  --  Entry point: load scan, reconstruct (CPU+GPU), verify, report
// ---------------------------------------------------------------------------
// Project 4.4 : Deep-Learning MRI/CT Reconstruction  (REDUCED-SCOPE TEACHING VERSION)
//
// WHAT THIS FILE DOES  (the 5-step shape every project follows)
//   1. Load the acquisition (data/sample file, or the built-in synthetic phantom):
//      the under-sampled k-space, the sampling mask, and (for scoring) the truth.
//   2. CPU reference reconstruction (reference_cpu.cpp)      -> trusted answer.
//   3. GPU unrolled reconstruction (kernels.cu)              -> the thing taught.
//   4. VERIFY: GPU image matches the CPU image within tolerance -> correctness.
//   5. REPORT: deterministic result to stdout; timing to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Timings (which vary run to run) go to STDERR.
//
//   IMPORTANT (honesty): this is a REDUCED-SCOPE teaching version. The "network"
//   uses FIXED operators (a Gaussian denoiser prior + k-space data consistency),
//   NOT trained weights, and a direct DFT instead of cuFFT. It teaches the
//   UNROLLED-RECONSTRUCTION STRUCTURE that real learned methods (E2E-VarNet on
//   fastMRI) share. See ../THEORY.md "Where this sits in the real world".
//
// Code tour: start here, then kernels.cuh (types) -> recon_core.h + dft_core.h
// (the shared math) -> kernels.cu (GPU) and reference_cpu.cpp (CPU baseline).
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // Acquisition, ReconParams, recon_gpu
#include "reference_cpu.h"    // make_synthetic_acquisition, load_acquisition,
                              //   recon_cpu, rms_error
#include "util/io.hpp"        // util::CpuTimer, util::max_abs_err

static const char* PROJECT_ID   = "4.4";
static const char* PROJECT_NAME = "Deep-Learning MRI/CT Reconstruction";

// Default synthetic image size when no sample file is supplied. Kept small so the
// O(N^2) teaching DFT runs in well under a second on both CPU and GPU.
static constexpr int DEF_NY = 24;
static constexpr int DEF_NX = 24;

// Reconstruction hyper-parameters (fixed; a trained net would learn these).
// lambda=0.4 with 12 unrolled stages was chosen because it visibly improves the
// zero-filled image on the synthetic phantom (~11% lower RMS) while staying
// stable; see the parameter discussion in THEORY.md and the Exercises.
static constexpr int   RECON_STAGES = 12;
static constexpr float RECON_LAMBDA = 0.4f;

// Verification tolerance. The recon is a LONG iterative float pipeline (stages x
// two DFTs each), so the GPU's fused-multiply-add and the host compiler diverge
// by ~1e-5 accumulated over the unroll -- real, and worth teaching (PATTERNS.md
// section 4). We verify to a physically-negligible 1e-3 on pixel values in [0,1].
static constexpr double TOLERANCE = 1.0e-3;

int main(int argc, char** argv) {
    // ---- 1. Load / synthesize the acquisition ------------------------------
    Acquisition acq;
    const char* source = "synthetic phantom (built-in)";
    if (argc > 1 && load_acquisition(argv[1], acq)) {
        source = argv[1];
    } else {
        acq = make_synthetic_acquisition(DEF_NY, DEF_NX);
    }
    ReconParams params;
    params.stages = RECON_STAGES;
    params.lambda = RECON_LAMBDA;

    // Count how many k-space samples we actually kept (the "acceleration").
    int sampled = 0;
    for (int m : acq.mask) sampled += m;
    const double frac = 100.0 * sampled / acq.n();

    // ---- 2. CPU reference reconstruction (timed) ---------------------------
    std::vector<float> recon_cpu_img;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    recon_cpu(acq, params, recon_cpu_img);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU reconstruction (stage kernels timed) -----------------------
    std::vector<float> recon_gpu_img;
    float gpu_kernel_ms = 0.0f;
    recon_gpu(acq, params, recon_gpu_img, &gpu_kernel_ms);

    // ---- 4. Verify GPU vs CPU ----------------------------------------------
    const double err = util::max_abs_err(recon_cpu_img, recon_gpu_img);
    const bool pass = err <= TOLERANCE;

    // ---- Science score: reconstruction quality vs the ground truth ---------
    // A 0-stage recon is the ZERO-FILLED image (measured k-space straight through
    // the inverse transform) -- the aliased starting point. Running 0 stages of
    // the SAME pipeline gives it deterministically.
    ReconParams zero_params; zero_params.stages = 0; zero_params.lambda = RECON_LAMBDA;
    std::vector<float> zero_filled;
    recon_cpu(acq, zero_params, zero_filled);
    const double rms_zf = rms_error(zero_filled,   acq.truth);   // before
    const double rms_rc = rms_error(recon_gpu_img, acq.truth);   // after

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("[REDUCED-SCOPE teaching demo: fixed denoiser prior + k-space data consistency,\n");
    std::printf(" unrolled %d stages. Not a trained network -- see THEORY.md.]\n", params.stages);
    std::printf("image = %d x %d, k-space samples kept = %d / %d (%.1f%%)\n",
                acq.ny, acq.nx, sampled, acq.n(), frac);
    std::printf("RMS error vs truth : zero-filled = %.6f  ->  reconstructed = %.6f\n",
                rms_zf, rms_rc);
    std::printf("recon improved zero-filled by %.1f%%\n",
                100.0 * (rms_zf - rms_rc) / rms_zf);
    // A deterministic fingerprint of the reconstruction: 8 evenly-spaced pixels
    // along the image's main diagonal (row i, col i scaled to the width). We
    // print 4 decimals: that is stable across Debug/Release AND CPU/GPU here (the
    // last ~2 digits of a 6-decimal print wobble with the compiler's FMA choices
    // -- exactly the numerics THEORY.md "Numerical considerations" discusses).
    std::printf("recon diagonal samples (8):");
    for (int s8 = 0; s8 < 8; ++s8) {
        const int y = (s8 * (acq.ny - 1)) / 7;
        const int x = (s8 * (acq.nx - 1)) / 7;
        std::printf(" %.4f", recon_gpu_img[static_cast<std::size_t>(y) * acq.nx + x]);
    }
    std::printf("\n");
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-03)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%d x %d image)\n", source, acq.ny, acq.nx);
    std::fprintf(stderr, "[timing] CPU recon: %.3f ms   GPU stage kernels: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- the direct O(N^2) DFT dominates; a real "
                         "pipeline uses cuFFT (O(N log N)) and a trained CNN, where the GPU's edge is large.\n");
    std::fprintf(stderr, "[verify] max_abs_err(GPU,CPU) = %.3e  (tolerance %.1e)\n", err, TOLERANCE);

    return pass ? 0 : 1;
}
