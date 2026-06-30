// ===========================================================================
// src/main.cu  --  Entry point: load data, run CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 2.1 : Protein Structure Prediction Inference (AlphaFold-class)
//               REDUCED-SCOPE TEACHING VERSION -- ONE Evoformer building block:
//               scaled dot-product self-attention over a protein's residues.
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the problem (Q/K/V matrices from data/sample/, or a built-in
//      synthetic fallback so the program always runs).
//   2. Compute the CPU reference (reference_cpu.cpp)         -> trusted answer.
//   3. Compute the GPU result    (kernels.cu)                -> the thing taught.
//   4. VERIFY: assert GPU agrees with CPU within a tolerance -> correctness.
//   5. REPORT: deterministic result to stdout; timing to stderr.
//
//   STDOUT is kept byte-for-byte deterministic so demo/run_demo can diff it
//   against demo/expected_output.txt. Anything that varies run-to-run (timings)
//   goes to STDERR, which the demo shows but does not diff.
//
// READ THIS FIRST in the code tour, then attention_core.h, kernels.cuh ->
// kernels.cu, and reference_cpu.cpp for the baseline. See ../THEORY.md for "why".
// ===========================================================================
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

#include "kernels.cuh"        // attention_gpu (GPU path), AttentionProblem
#include "reference_cpu.h"    // load_attention, attention_cpu (CPU baseline)
#include "attention_core.h"   // scaled_score, stable_exp, D_MODEL
#include "util/io.hpp"        // util::CpuTimer, util::max_abs_err

static const char* PROJECT_ID   = "2.1";
static const char* PROJECT_NAME = "Protein Structure Prediction Inference (AlphaFold-class)";

// Correctness tolerance: CPU and GPU share the exact per-element math
// (attention_core.h) and accumulate channels in the same j-order, so they agree
// to ~FP32 rounding. 1e-5 is a tight, honest bound for FP32 outputs whose
// channel sums involve ~L double-precision adds reduced to float (THEORY sec 6).
static constexpr double TOLERANCE = 1.0e-5;

// ---------------------------------------------------------------------------
// make_synthetic_problem: a tiny, INTERPRETABLE problem used when no data file
// is given. We engineer a known answer (PATTERNS.md sec 6): each query residue
// i is built to be most similar to one specific key residue, so the "dominant
// attended residue" we print is predictable. See scripts/make_synthetic.py,
// which writes the very same numbers to data/sample/attention_sample.txt.
//
//   Construction: feature channel 0 carries a one-hot-ish "identity" signal.
//   Residue r's Q, K, V all put a large value `PEAK` in channel (r % d) and
//   small ramp values elsewhere. Then Q[i].K[j] is maximised at j == i (the
//   peaks line up), so each residue attends mostly to ITSELF -- the simplest
//   verifiable attention pattern (self-attention's identity-like baseline).
// ---------------------------------------------------------------------------
static AttentionProblem make_synthetic_problem() {
    AttentionProblem prob;
    const int L = 6;                 // 6 residues: small enough to read by hand
    const int d = D_MODEL;           // 32 feature channels
    prob.L = L;
    prob.d = d;
    const std::size_t mat = static_cast<std::size_t>(L) * d;
    prob.q.resize(mat);
    prob.k.resize(mat);
    prob.v.resize(mat);

    const float PEAK = 3.0f;         // dominant signal placed in one channel
    for (int r = 0; r < L; ++r) {
        for (int c = 0; c < d; ++c) {
            // A small, deterministic ramp so vectors are not degenerate.
            const float ramp = 0.01f * static_cast<float>((r * d + c) % 7);
            const float peak = (c == (r % d)) ? PEAK : 0.0f;   // residue r's "identity" channel
            const std::size_t idx = static_cast<std::size_t>(r) * d + c;
            prob.q[idx] = ramp + peak;
            prob.k[idx] = ramp + peak;
            // Values encode the residue index in channel 0 so the mixed output
            // is easy to interpret (out[i] ~ V[i] when i attends to itself).
            prob.v[idx] = (c == 0) ? static_cast<float>(r + 1) : ramp;
        }
    }
    return prob;
}

// ---------------------------------------------------------------------------
// dominant_attended_residue: for query residue i, return the residue j that
// receives the LARGEST softmax weight, plus that weight. Computed deterministic-
// ally on the host using the SAME shared primitives the kernel uses, purely so
// the stdout summary is interpretable (it is not part of the GPU path). Ties
// break to the lower index. O(L*d) per call.
// ---------------------------------------------------------------------------
static void dominant_attended_residue(const AttentionProblem& prob, int i,
                                      int& best_j, double& best_w) {
    const int L = prob.L, d = prob.d;
    const float* q_i = &prob.q[static_cast<std::size_t>(i) * d];

    double row_max = -1.0e308;
    std::vector<double> s(static_cast<std::size_t>(L));
    for (int j = 0; j < L; ++j) {
        const float* k_j = &prob.k[static_cast<std::size_t>(j) * d];
        s[j] = scaled_score(q_i, k_j, d);
        if (s[j] > row_max) row_max = s[j];
    }
    double denom = 0.0;
    for (int j = 0; j < L; ++j) { s[j] = stable_exp(s[j], row_max); denom += s[j]; }

    best_j = 0; best_w = -1.0;
    for (int j = 0; j < L; ++j) {
        const double w = s[j] / denom;
        if (w > best_w) { best_w = w; best_j = j; }   // strict > -> lower idx on ties
    }
}

// L2 norm of one output row (a stable, single-number fingerprint of the row).
static double row_norm(const std::vector<float>& out, int i, int d) {
    double acc = 0.0;
    for (int c = 0; c < d; ++c) {
        const double x = static_cast<double>(out[static_cast<std::size_t>(i) * d + c]);
        acc += x * x;
    }
    return std::sqrt(acc);
}

int main(int argc, char** argv) {
    // ---- 1. Load the problem ------------------------------------------------
    AttentionProblem prob;
    std::string source;
    const std::string path = (argc > 1) ? argv[1]
                                        : "data/sample/attention_sample.txt";
    try {
        prob   = load_attention(path);
        source = path;
    } catch (const std::exception& e) {
        // No file (or a malformed one): fall back to the built-in problem so the
        // program always produces a result. We note the reason on stderr.
        std::fprintf(stderr, "[data]   could not load '%s' (%s); using built-in synthetic.\n",
                     path.c_str(), e.what());
        prob   = make_synthetic_problem();
        source = "synthetic (built-in)";
    }

    // ---- 2. CPU reference (timed) ------------------------------------------
    std::vector<float> out_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    attention_cpu(prob, out_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU result (kernel timed inside the wrapper) -------------------
    std::vector<float> out_gpu;
    float gpu_kernel_ms = 0.0f;
    attention_gpu(prob, out_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ----------------------------------------------------------
    const double err = util::max_abs_err(out_cpu, out_gpu);
    const bool pass = err <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("[reduced-scope: one scaled dot-product self-attention head]\n");
    std::printf("L = %d residues, d = %d feature channels\n", prob.L, prob.d);
    std::printf("per-residue attention (GPU result):\n");
    for (int i = 0; i < prob.L; ++i) {
        int best_j = 0; double best_w = 0.0;
        dominant_attended_residue(prob, i, best_j, best_w);
        // Print: which residue i attends to most, that weight, and the output
        // row's L2 norm. All deterministic (rounded to 6 decimals).
        std::printf("  residue %d -> attends most to residue %d (w=%.6f)  |out|=%.6f\n",
                    i, best_j, best_w, row_norm(out_gpu, i, prob.d));
    }
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-05)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (L=%d, d=%d)\n", source.c_str(), prob.L, prob.d);
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny L is dominated by "
                         "launch/copy overhead; attention's O(L^2 d) cost makes the GPU win at "
                         "real sequence lengths (hundreds-thousands of residues).\n");
    std::fprintf(stderr, "[verify] max_abs_err = %.6e  (tolerance %.1e)\n", err, TOLERANCE);

    // Exit code feeds the demo's pass/fail gate.
    return pass ? 0 : 1;
}
