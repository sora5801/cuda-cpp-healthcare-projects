// ===========================================================================
// src/main.cu  --  Entry point: load fundus image, run CPU + GPU CNN, verify
// ---------------------------------------------------------------------------
// Project 7.18 : Retinal Fundus AI Screening
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load a fundus image (data/sample, or a synthetic fallback).
//   2. Build the fixed teaching CNN and run the CPU reference forward pass.
//   3. Run the GPU forward pass (kernels.cu).
//   4. VERIFY: GPU logits/probs/CAM agree with the CPU within tolerance, and
//      both predict the same DR grade.
//   5. REPORT: deterministic grade + probabilities + CAM summary to stdout;
//      timing to stderr.
//
//   STDOUT is kept byte-for-byte deterministic so demo/run_demo can diff it
//   against demo/expected_output.txt. Anything run-to-run (timings) goes to
//   STDERR, which the demo shows but does not diff.
//
//   NOT FOR CLINICAL USE. Fixed (untrained) weights on synthetic data -- a
//   teaching model of the CNN inference pipeline, not a diagnostic tool.
//
// READ THIS FIRST in the code tour, then kernels.cuh -> kernels.cu, and
// reference_cpu.cpp / cnn_core.h for the baseline. See ../THEORY.md for the "why".
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // forward_gpu (GPU path), TILE
#include "reference_cpu.h"    // FundusImage, CnnModel, forward_cpu, load_fundus
#include "util/io.hpp"        // util::CpuTimer, util::max_abs_err

static const char* PROJECT_ID   = "7.18";
static const char* PROJECT_NAME = "Retinal Fundus AI Screening";

// Correctness tolerance for the CPU-vs-GPU comparison. This is a multi-stage
// float pipeline (two conv layers + pooling + a float tree-reduction global
// average pool whose summation ORDER differs from the CPU's serial double sum).
// Per PATTERNS.md section 4, that legitimately diverges by ~1e-5..1e-4, so we
// verify to a small physical tolerance and say so -- NOT bit-exact. THEORY.md
// section 5 explains the numerics.
static constexpr double TOLERANCE = 1.0e-3;

// Human-readable DR severity labels for the 5 grades (clinical convention).
static const char* GRADE_NAME[CNN_NUM_CLASSES] = {
    "0 no DR", "1 mild", "2 moderate", "3 severe", "4 proliferative"};

// ---------------------------------------------------------------------------
// make_synthetic_image: a tiny built-in fallback used when no data file is
//   given. A 16x16 RGB image with a bright greenish "optic-disc"-like blob and
//   a few dark red "microaneurysm"-like spots -- purely illustrative, clearly
//   synthetic (never implies a real retina). The committed sample in
//   data/sample/ is the same idea at 32x32 (see scripts/make_synthetic.py).
// ---------------------------------------------------------------------------
static FundusImage make_synthetic_image() {
    FundusImage img;
    img.C = CNN_C_IN; img.H = 16; img.W = 16; img.label = -1;
    img.data.assign((std::size_t)img.C * img.H * img.W, 0.0f);
    auto at = [&](int c, int y, int x) -> float& {
        return img.data[(std::size_t)c * img.H * img.W + y * img.W + x];
    };
    for (int y = 0; y < img.H; ++y)
        for (int x = 0; x < img.W; ++x) {
            // Warm reddish background (fundus is orange-red).
            at(0, y, x) = 0.55f; at(1, y, x) = 0.25f; at(2, y, x) = 0.15f;
            // Bright blob near the centre (optic-disc-like): boost G and R.
            const float d = std::sqrt((float)((y - 8) * (y - 8) + (x - 8) * (x - 8)));
            if (d < 3.0f) { at(0, y, x) = 0.9f; at(1, y, x) = 0.85f; at(2, y, x) = 0.6f; }
        }
    // Two small dark-red spots (microaneurysm-like).
    at(1, 4, 12) = 0.05f; at(2, 4, 12) = 0.05f; at(0, 4, 12) = 0.6f;
    at(1, 12, 5) = 0.05f; at(2, 12, 5) = 0.05f; at(0, 12, 5) = 0.6f;
    return img;
}

int main(int argc, char** argv) {
    // ---- 1. Load the image -------------------------------------------------
    FundusImage img;
    const char* source = "synthetic (built-in 16x16)";
    if (argc > 1) {
        try {
            img = load_fundus(argv[1]);
            source = argv[1];
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[error] %s\n", e.what());
            return 2;
        }
    } else {
        img = make_synthetic_image();
    }

    // The two 2x2 pools require the dimensions to stay >= 1 after halving twice.
    if (img.H < 4 || img.W < 4) {
        std::fprintf(stderr, "[error] image too small (need H,W >= 4)\n");
        return 2;
    }

    const CnnModel model = make_fixed_model();

    // ---- 2. CPU reference forward pass (timed) -----------------------------
    ForwardResult cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    forward_cpu(img, model, cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU forward pass (conv layers timed inside) --------------------
    ForwardResult gpu;
    float conv_ms = 0.0f;
    forward_gpu(img, model, gpu, &conv_ms);

    // ---- 4. Verify ---------------------------------------------------------
    const double err_logits = util::max_abs_err(cpu.logits, gpu.logits);
    const double err_probs  = util::max_abs_err(cpu.probs,  gpu.probs);
    const double err_cam    = util::max_abs_err(cpu.cam,    gpu.cam);
    const bool grade_match  = (cpu.pred_grade == gpu.pred_grade);
    const double worst = std::fmax(err_logits, std::fmax(err_probs, err_cam));
    const bool pass = grade_match && (worst <= TOLERANCE);

    // Locate the CAM peak (the "most suspicious" region) on the GPU result.
    int cam_py = 0, cam_px = 0; float cam_peak = gpu.cam.empty() ? 0.0f : gpu.cam[0];
    for (int y = 0; y < gpu.cam_h; ++y)
        for (int x = 0; x < gpu.cam_w; ++x) {
            const float v = gpu.cam[(std::size_t)y * gpu.cam_w + x];
            if (v > cam_peak) { cam_peak = v; cam_py = y; cam_px = x; }
        }

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("[teaching CNN inference: conv->relu->pool x2 -> GAP -> FC -> softmax]\n");
    std::printf("image: %dx%d RGB  (channel-major, normalized [0,1])\n", img.H, img.W);
    std::printf("predicted DR grade: %s\n", GRADE_NAME[gpu.pred_grade]);
    std::printf("class probabilities:");
    for (int k = 0; k < CNN_NUM_CLASSES; ++k) std::printf(" %.6f", gpu.probs[k]);
    std::printf("\n");
    std::printf("Grad-CAM %dx%d peak = %.6f at (row=%d,col=%d)\n",
                gpu.cam_h, gpu.cam_w, cam_peak, cam_py, cam_px);
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-03; same grade)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (label=%d)\n", source, img.label);
    std::fprintf(stderr, "[model]  fixed (untrained) weights; NOT for clinical use\n");
    std::fprintf(stderr, "[timing] CPU forward: %.3f ms   GPU conv layers: %.3f ms\n",
                 cpu_ms, conv_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- the GPU's edge grows with image "
                         "size (real fundus is 2048x2048) and with batching.\n");
    std::fprintf(stderr, "[verify] max_abs_err logits=%.3e probs=%.3e cam=%.3e  (tol %.1e)\n",
                 err_logits, err_probs, err_cam, TOLERANCE);
    std::fprintf(stderr, "[verify] grade CPU=%d GPU=%d  -> %s\n",
                 cpu.pred_grade, gpu.pred_grade, grade_match ? "match" : "MISMATCH");

    return pass ? 0 : 1;
}
