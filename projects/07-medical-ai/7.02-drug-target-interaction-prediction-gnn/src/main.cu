// ===========================================================================
// src/main.cu  --  Entry point: score drug x protein pairs, verify, report
// ---------------------------------------------------------------------------
// Project 7.2 : Drug-Target Interaction Prediction (GNN)
//
// 5-step shape:
//   1. Load the batched drug graphs + protein descriptors (data/sample).
//   2. Build the FIXED (untrained, seeded) GNN weights (reference_cpu.cpp).
//   3. Run the forward pass on CPU (reference) and GPU (kernels.cu).
//   4. VERIFY: GPU embeddings + DTI scores match the CPU within a tiny tolerance.
//   5. REPORT: deterministic DTI score matrix + the top-ranked pair to stdout;
//      timing + verification detail to stderr.
//
// STDOUT is byte-for-byte deterministic (fixed weights, fixed-order reductions)
// so demo/run_demo can diff it against demo/expected_output.txt. Timings go to
// STDERR (shown, not diffed).
//
// Code tour: start here, then gnn.h (the math), kernels.cuh, kernels.cu,
// reference_cpu.cpp.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // dti_gpu, Dataset, GnnModel
#include "reference_cpu.h"    // load_dataset, build_model, dti_cpu
#include "util/io.hpp"        // util::CpuTimer, util::max_abs_err

static const char* PROJECT_ID   = "7.2";
static const char* PROJECT_NAME = "Drug-Target Interaction Prediction (GNN)";

// Verification tolerance. The CPU and GPU run identical math (gnn.h); the only
// divergence is the GPU's fused multiply-add (FMA) contracting `a*b + c` into a
// single rounding, vs. the host's two-step rounding. Over these tiny sums that
// difference is ~1e-6, so 1e-4 on a probability in [0,1] is safe and honest
// (PATTERNS.md sec 4 -- "same exact operations, small FP tolerance").
static constexpr double TOLERANCE = 1.0e-4;

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/dti_sample.txt";
    Dataset d;
    try {
        d = load_dataset(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. Fixed (untrained, seeded) model --------------------------------
    const GnnModel model = build_model();

    // ---- 3a. CPU reference (timed) -----------------------------------------
    std::vector<float> emb_cpu, score_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    dti_cpu(d, model, emb_cpu, score_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3b. GPU forward pass (kernels timed inside the wrapper) -----------
    std::vector<float> emb_gpu, score_gpu;
    float gpu_kernel_ms = 0.0f;
    dti_gpu(d, model, emb_gpu, score_gpu, &gpu_kernel_ms);

    // ---- 4. Verify (embeddings + scores) -----------------------------------
    const double emb_err   = util::max_abs_err(emb_cpu, emb_gpu);
    const double score_err = util::max_abs_err(score_cpu, score_gpu);
    const bool pass = (emb_err <= TOLERANCE) && (score_err <= TOLERANCE);

    // Find the top-scoring pair from the GPU matrix (deterministic argmax: ties
    // -> lowest flat index). This is the model's single best DTI prediction and
    // the number the demo highlights.
    int best_j = 0;
    for (int j = 1; j < d.D * d.P; ++j)
        if (score_gpu[j] > score_gpu[best_j]) best_j = j;
    const int best_drug = best_j / d.P;
    const int best_prot = best_j % d.P;

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("[reduced-scope teaching model: FIXED (untrained) message-passing GNN]\n");
    std::printf("batch: %d drugs x %d proteins, %d atoms total, F=%d, T=%d rounds\n",
                d.D, d.P, d.total_nodes, GNN_F, GNN_T);
    std::printf("DTI score matrix (rows = drugs, cols = proteins), probabilities:\n");
    for (int drug = 0; drug < d.D; ++drug) {
        std::printf("  drug %d:", drug);
        for (int p = 0; p < d.P; ++p)
            std::printf(" %.4f", score_gpu[static_cast<std::size_t>(drug) * d.P + p]);
        std::printf("\n");
    }
    std::printf("top interaction: drug %d <-> protein %d  (score %.4f)\n",
                best_drug, best_prot, score_gpu[best_j]);
    std::printf("implanted ground truth: drug %d <-> protein %d  -> %s\n",
                d.true_drug, d.true_prot,
                (best_drug == d.true_drug && best_prot == d.true_prot) ? "RECOVERED" : "not top");
    std::printf("RESULT: %s (GPU embeddings+scores match CPU)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d drugs, %d proteins, %d atoms)\n",
                 path.c_str(), d.D, d.P, d.total_nodes);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernels: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- tiny batch is launch-bound; the GPU's edge "
                         "grows to millions of compounds x thousands of targets.\n");
    std::fprintf(stderr, "[verify] max |emb_cpu-emb_gpu| = %.3e, max |score_cpu-score_gpu| = %.3e "
                         "(tolerance %.1e)\n", emb_err, score_err, TOLERANCE);
    std::fprintf(stderr, "[honesty] weights are UNTRAINED (seeded); scores are illustrative, "
                         "NOT clinical binding predictions.\n");

    return pass ? 0 : 1;
}
