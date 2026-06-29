// ===========================================================================
// src/main.cu  --  Entry point: predict molecular properties, verify, report
// ---------------------------------------------------------------------------
// Project 1.11 : QSAR / Property Prediction
//
// 5-step shape (the shape every project in this repo follows):
//   1. Load a BATCH of molecular graphs (data/sample) + build fixed GCN weights.
//   2. CPU reference: run the 2-layer GCN + readout (reference_cpu.cpp).
//   3. GPU: the same pipeline as three kernels (kernels.cu).
//   4. VERIFY: GPU predictions agree with CPU within a documented fp32 tolerance.
//   5. REPORT: deterministic per-molecule predictions + a ranking to stdout;
//      timings and the measured error to stderr.
//
// Code tour: start here, then gcn.h (the per-node math), reference_cpu.cpp (the
// loader + serial reference), kernels.cu (the GPU twin).
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // gcn_predict_gpu, Graph, Model
#include "reference_cpu.h"    // load_graph, make_model, gcn_predict_cpu
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "1.11";
static const char* PROJECT_NAME = "QSAR / Property Prediction";

// Verification tolerance. The CPU and GPU run the IDENTICAL gcn.h math in the
// IDENTICAL CSR order, so they differ only by the GPU's optional fused-multiply-
// add (FMA) contraction -- a few ulp per layer. 1e-4 is far above that and far
// below the spread of the predictions, so it certifies "same computation" while
// being honest about fp32 (PATTERNS.md §4).
static constexpr double TOLERANCE = 1.0e-4;

int main(int argc, char** argv) {
    // ---- 1. Load the batch + build the (fixed) model ----------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/molecules_sample.txt";
    Graph g;
    try {
        g = load_graph(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const Model model = make_model();

    // ---- 2. CPU reference (timed) ----------------------------------------
    std::vector<float> pred_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    gcn_predict_cpu(g, model, pred_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU pipeline (kernels timed) ---------------------------------
    std::vector<float> pred_gpu;
    float gpu_kernel_ms = 0.0f;
    gcn_predict_gpu(g, model, pred_gpu, &gpu_kernel_ms);

    // ---- 4. Verify (max abs error over all molecules) --------------------
    const double max_err = util::max_abs_err(pred_cpu, pred_gpu);
    const bool pass = (max_err <= TOLERANCE);

    // ---- 5a. Deterministic report -> STDOUT ------------------------------
    // Predictions are printed at fixed precision in molecule order, then the
    // single highest-scoring molecule is named so the demo has a clear "answer".
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("GCN inference: %d molecules, %d atoms total (F_IN=%d, F_HID=%d, F_OUT=%d)\n",
                g.num_mols, g.num_nodes, GCN_F_IN, GCN_F_HID, GCN_F_OUT);
    int best = 0;
    for (int m = 0; m < g.num_mols; ++m) {
        std::printf("  mol %2d (%2d atoms): predicted property = %.6f\n",
                    m, g.atoms_in(m), pred_gpu[m]);
        if (pred_gpu[m] > pred_gpu[best]) best = m;
    }
    std::printf("top-ranked molecule: mol %d (property = %.6f)\n", best, pred_gpu[best]);
    std::printf("RESULT: %s (GPU predictions match CPU within %.0e)\n",
                pass ? "PASS" : "FAIL", TOLERANCE);

    // ---- 5b. Varying detail -> STDERR ------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d molecules, %d atoms, %d edges incl. self-loops)\n",
                 path.c_str(), g.num_mols, g.num_nodes,
                 static_cast<int>(g.col_idx.size()));
    std::fprintf(stderr, "[model]  fixed synthetic weights (seeded LCG) -- NOT a trained model; "
                         "predictions are a synthetic demo, not real ADMET.\n");
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernels: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- this tiny batch is launch-bound; the GPU's "
                         "edge appears at 10^5-10^8 molecules.\n");
    std::fprintf(stderr, "[verify] max |pred_cpu - pred_gpu| = %.3e  (tolerance %.1e)\n",
                 max_err, TOLERANCE);

    return pass ? 0 : 1;
}
