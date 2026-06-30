// ===========================================================================
// src/main.cu  --  Entry point: load protein, run CPU + GPU attention, verify
// ---------------------------------------------------------------------------
// Project 3.18 : Protein Language Model Inference
//
// 5-step shape (the shape EVERY project in this repo follows):
//   1. Load the protein sequence + model shape (data/sample).
//   2. Build the synthetic input embeddings (shared generator).
//   3. Run the CPU reference self-attention (reference_cpu.cpp) -> trusted answer.
//   4. Run the GPU self-attention (kernels.cu) and VERIFY it matches the CPU.
//   5. REPORT a deterministic per-residue summary to stdout; timing to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Run-varying numbers (timings) go to STDERR.
//
// Code tour: start here, then kernels.cuh -> kernels.cu, then reference_cpu.cpp
// and attention_math.h. See ../THEORY.md for the "why".
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // attention_gpu (GPU path)
#include "reference_cpu.h"    // load_protein, build_embeddings, attention_cpu
#include "util/io.hpp"        // util::CpuTimer, util::max_abs_err

static const char* PROJECT_ID   = "3.18";
static const char* PROJECT_NAME = "Protein Language Model Inference";

// Correctness tolerance. The whole block is FP32 with double-accumulated inner
// products on BOTH sides, but the softmax denominator is summed in a different
// ORDER on the GPU (a tree reduction) than on the CPU (left to right), and the
// projections use fused multiply-add differently. Over the multi-stage pipeline
// (project -> score -> softmax -> blend -> project) that yields a divergence of
// ~1e-5 in the output embeddings; we verify to 1e-4, a physically-negligible
// gap, and say so honestly (PATTERNS.md §4, the "long FP32 pipeline" case).
static constexpr double TOLERANCE = 1.0e-4;

int main(int argc, char** argv) {
    // ---- 1. Load the protein + model shape ---------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/protein_sample.txt";
    ProteinInput p;
    try {
        p = load_protein(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const AttnConfig& cfg = p.cfg;

    // ---- 2. Build the input embeddings (identical on CPU and GPU) -----------
    const std::vector<float> X = build_embeddings(p);

    // ---- 3. CPU reference (timed) ------------------------------------------
    AttnResult cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    attention_cpu(X, cfg, cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 4. GPU result (kernels timed inside the wrapper) + verify ----------
    AttnResult gpu;
    float gpu_kernel_ms = 0.0f;
    attention_gpu(X, cfg, gpu, &gpu_kernel_ms);

    // Headline correctness metric: largest mismatch in the OUTPUT embeddings.
    const double err_out  = util::max_abs_err(cpu.out, gpu.out);
    // Also check the head-0 attention map agrees (the interpretable tensor).
    const double err_attn = util::max_abs_err(cpu.attn, gpu.attn);
    // And that the discrete "most-attended residue" readout is identical.
    bool top_match = (cpu.top_attn == gpu.top_attn);
    const bool pass = (err_out <= TOLERANCE) && (err_attn <= TOLERANCE) && top_match;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("one multi-head self-attention block over %d residues\n", cfg.seq_len);
    std::printf("d_model=%d  heads=%d  d_head=%d\n", cfg.d_model, cfg.n_heads, cfg.d_head);
    std::printf("sequence: %s\n", p.sequence.c_str());
    // Per-residue summary: output-embedding norm (6 dp) and the residue that
    // head 0 attends to most. Both are computed by the GPU; the CPU agreement is
    // asserted above. Norms are rounded to 6 dp so the line is reproducible.
    std::printf("per-residue (idx aa  out_norm  head0_top->idx):\n");
    for (int i = 0; i < cfg.seq_len; ++i)
        std::printf("  %2d %c  %.6f  ->%2d\n",
                    i, p.sequence[i], gpu.out_norm[i], gpu.top_attn[i]);
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-04)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (L=%d, d_model=%d, heads=%d)\n",
                 path.c_str(), cfg.seq_len, cfg.d_model, cfg.n_heads);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernels: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- this is ONE block at tiny L; real ESM-2 "
                         "stacks 33 layers x 20 heads over L up to ~1024, where the GPU dominates.\n");
    std::fprintf(stderr, "[verify] max_abs_err(out) = %.3e  max_abs_err(attn) = %.3e  "
                         "top_attn match = %s  (tolerance %.1e)\n",
                 err_out, err_attn, top_match ? "yes" : "NO", TOLERANCE);

    return pass ? 0 : 1;
}
