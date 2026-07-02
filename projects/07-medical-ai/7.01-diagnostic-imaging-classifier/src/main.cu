// ===========================================================================
// src/main.cu  --  Entry point: load model+images, run CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 7.1 : Diagnostic Imaging Classifier   (REDUCED-SCOPE teaching version)
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the problem: model weights + a batch of images (from data/sample,
//      or a built-in synthetic fallback if no file is given).
//   2. Compute the CPU reference (reference_cpu.cpp)  -> trusted logits/preds.
//   3. Compute the GPU result    (kernels.cu)         -> the thing taught.
//   4. VERIFY: assert the GPU logits equal the CPU logits (EXACTLY -- shared
//      __host__ __device__ math), and the predictions match.
//   5. REPORT: deterministic per-image predictions + batch accuracy to stdout;
//      timing to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Anything that varies run-to-run (timings) goes to
//   STDERR, which the demo shows but does not diff (PATTERNS.md section 3).
//
//   NOT FOR CLINICAL USE. Synthetic images, fixed synthetic weights, a teaching
//   forward pass -- see THEORY.md for the honest gap to a real diagnostic model.
//
// READ THIS FIRST in the code tour, then kernels.cuh -> kernels.cu, and
// reference_cpu.h / reference_cpu.cpp for the baseline. See ../THEORY.md.
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // classify_gpu (GPU path)
#include "reference_cpu.h"    // Weights, Dataset, classify_cpu, shared math
#include "util/io.hpp"        // util::CpuTimer

// Identify the program. Kept in sync with demo/expected_output.txt.
static const char* PROJECT_ID   = "7.1";
static const char* PROJECT_NAME = "Diagnostic Imaging Classifier";

// Class names for the human-readable report (index = class id).
static const char* CLASS_NAME[NUM_CLS] = { "normal", "lesion" };

int main(int argc, char** argv) {
    // ---- 1. Load the problem ------------------------------------------------
    Weights w;
    Dataset d;
    const char* source = "synthetic (built-in)";
    if (argc > 1) {
        try {
            load_sample(argv[1], w, d);      // may throw if file bad/short
            source = argv[1];
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[data] could not load '%s' (%s); using built-in.\n",
                         argv[1], e.what());
            make_builtin(w, d);
        }
    } else {
        make_builtin(w, d);
    }

    // ---- 2. CPU reference (timed) ------------------------------------------
    std::vector<float> logits_cpu;
    std::vector<int>   pred_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    classify_cpu(w, d, logits_cpu, pred_cpu);
    double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU result (kernels timed inside the wrapper) ------------------
    std::vector<float> logits_gpu;
    std::vector<int>   pred_gpu;
    float gpu_kernel_ms = 0.0f;
    classify_gpu(w, d, logits_gpu, pred_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ----------------------------------------------------------
    // Because CPU and GPU run the SAME __host__ __device__ arithmetic in the same
    // order, the logits are bit-identical: tolerance is EXACTLY 0. We report the
    // largest absolute logit difference (expected 0) and confirm every predicted
    // class matches too.
    double worst = 0.0;
    bool preds_match = true;
    for (std::size_t k = 0; k < logits_cpu.size(); ++k) {
        double diff = std::fabs(static_cast<double>(logits_cpu[k]) -
                                static_cast<double>(logits_gpu[k]));
        if (diff > worst) worst = diff;
    }
    for (int i = 0; i < d.n; ++i)
        if (pred_cpu[i] != pred_gpu[i]) preds_match = false;
    const double TOLERANCE = 0.0;                 // exact (see comment above)
    bool pass = (worst <= TOLERANCE) && preds_match;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("[reduced-scope teaching CNN inference: conv->relu->maxpool->dense->softmax]\n");
    std::printf("images = %d   geometry = %dx%d, %d filters (%dx%d), %d classes\n",
                d.n, IMG_H, IMG_W, NUM_F, KERNEL, KERNEL, NUM_CLS);

    // Per-image prediction table. P(lesion) is the softmax of the two logits,
    // printed at fixed precision so the text is deterministic. The CPU logits
    // are used for the printed numbers (GPU == CPU exactly, verified above).
    int correct = 0, labeled = 0;
    std::printf("\n idx  pred     P(lesion)  truth   ok\n");
    for (int i = 0; i < d.n; ++i) {
        float l0 = logits_cpu[static_cast<std::size_t>(i) * NUM_CLS + 0];
        float l1 = logits_cpu[static_cast<std::size_t>(i) * NUM_CLS + 1];
        float p1 = softmax_pos1(l0, l1);
        int   pr = pred_cpu[i];
        int   gt = d.labels[i];
        const char* ok = "-";
        if (gt >= 0) {
            ++labeled;
            bool hit = (pr == gt);
            if (hit) ++correct;
            ok = hit ? "yes" : "NO";
        }
        std::printf("%4d  %-6s   %8.4f    %-6s  %s\n",
                    i, CLASS_NAME[pr], p1,
                    (gt >= 0 ? CLASS_NAME[gt] : "?"), ok);
    }

    // Batch accuracy over the labeled images (deterministic integer ratio).
    if (labeled > 0) {
        // Print as an exact fraction plus a fixed-precision percentage so the
        // line is reproducible regardless of platform float printing.
        double acc = 100.0 * static_cast<double>(correct) / labeled;
        std::printf("\naccuracy: %d/%d correct (%.1f%%)\n", correct, labeled, acc);
    }
    std::printf("RESULT: %s (GPU matches CPU exactly; tol=%.1g)\n",
                pass ? "PASS" : "FAIL", TOLERANCE);

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s\n", source);
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernels: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny batch is "
                         "dominated by launch/copy overhead, not compute; the GPU "
                         "edge grows with batch size and image resolution.\n");
    std::fprintf(stderr, "[verify] max |logit_cpu - logit_gpu| = %.3e (tol %.1g); "
                         "predictions match: %s\n",
                 worst, TOLERANCE, preds_match ? "yes" : "no");

    // Exit code feeds the demo's pass/fail gate.
    return pass ? 0 : 1;
}
