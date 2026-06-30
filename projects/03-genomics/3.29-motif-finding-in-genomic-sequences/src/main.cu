// ===========================================================================
// src/main.cu  --  Entry point: load sequences, run MEME EM, verify, report
// ---------------------------------------------------------------------------
// Project 3.29 : Motif Finding in Genomic Sequences
//
// WHAT THIS FILE DOES  (the 5-step shape EVERY project in this repo follows)
//   1. Load the problem: a FASTA-like set of DNA sequences sharing a hidden
//      motif of width W (data/sample, or the path given as argv[1]).
//   2. Run the trusted CPU MEME EM (reference_cpu.cpp) to convergence -> the
//      recovered motif + the FINAL-model window scores (the E-step's output).
//   3. Re-run the E-step (the expensive, parallelised step) on the GPU for the
//      SAME final model (kernels.cu) -> the thing being taught.
//   4. VERIFY: assert the GPU window scores match the CPU's EXACTLY (both call
//      the same __host__ __device__ window_score()).
//   5. REPORT: deterministic recovered motif + per-sequence sites to stdout;
//      timing to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Run-to-run timings go to STDERR (shown, not diffed).
//
// READ THIS FIRST in the code tour, then motif_core.h -> kernels.cuh ->
// kernels.cu, and reference_cpu.cpp for the EM baseline. See ../THEORY.md.
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // score_windows_gpu (GPU E-step)
#include "reference_cpu.h"    // load_sequences, run_meme_em_cpu, score_windows_*
#include "util/io.hpp"        // util::max_abs_err

static const char* PROJECT_ID   = "3.29";
static const char* PROJECT_NAME = "Motif Finding in Genomic Sequences";

// Motif width to search for. The committed sample plants an 8 bp motif, so we
// look for width 8. (A real run sweeps a range of widths and keeps the best by
// information content -- see THEORY "real world" and the exercises.)
static constexpr int MOTIF_WIDTH = 8;

// EM controls: a generous iteration cap and a tight log-likelihood tolerance.
static constexpr int    EM_MAX_ITERS = 100;
static constexpr double EM_TOL       = 1.0e-6;

// Correctness tolerance for the GPU-vs-CPU window-score check.
//   The GPU kernel and the CPU reference call the IDENTICAL window_score() with
//   the same fixed summation order, so the float results are bit-for-bit equal
//   -> we verify with an EXACT zero tolerance (PATTERNS.md sec 4: exact when the
//   same operations run on both sides). A tiny epsilon would also pass, but 0
//   states the stronger truth honestly.
static constexpr double TOLERANCE = 0.0;

// ---------------------------------------------------------------------------
// seed_model: build the INITIAL PWM that EM starts from.
//   * Background bg[] is estimated from the OBSERVED base composition of all
//     sequences (the per-window null model). This is data-driven, not assumed.
//   * The initial PWM is a mild, deterministic perturbation of the background:
//     each column slightly favours a different base in a fixed rotation, so EM
//     has a non-degenerate starting point but no hand-planted answer. Because
//     the seed is deterministic, the whole run is reproducible.
// ---------------------------------------------------------------------------
static void seed_model(const SequenceSet& set, MotifModel& model) {
    model.w = set.w;
    model.bg.assign(MOTIF_ALPHABET, 0.0f);
    model.pwm.assign(static_cast<std::size_t>(set.w) * MOTIF_ALPHABET, 0.0f);

    // Background = observed ACGT frequencies over all sequence data.
    long long counts[MOTIF_ALPHABET] = {0, 0, 0, 0};
    long long total = 0;
    for (unsigned char b : set.data)
        if (b < MOTIF_ALPHABET) { counts[b]++; total++; }
    for (int b = 0; b < MOTIF_ALPHABET; ++b)
        model.bg[b] = total ? static_cast<float>(static_cast<double>(counts[b]) / total)
                            : 0.25f;

    // Initial PWM: mostly background, with a small fixed bump toward base
    // (column index mod 4). A weak, deterministic break of symmetry.
    for (int p = 0; p < set.w; ++p) {
        const int fav = p % MOTIF_ALPHABET;
        float z = 0.0f;
        for (int b = 0; b < MOTIF_ALPHABET; ++b) {
            float v = model.bg[b] + (b == fav ? 0.15f : 0.0f);
            model.pwm[p * MOTIF_ALPHABET + b] = v;
            z += v;
        }
        for (int b = 0; b < MOTIF_ALPHABET; ++b)   // renormalise the column
            model.pwm[p * MOTIF_ALPHABET + b] /= z;
    }
}

int main(int argc, char** argv) {
    // ---- 1. Load the sequences ---------------------------------------------
    const std::string path =
        (argc > 1) ? argv[1] : "data/sample/sequences_sample.fasta";
    SequenceSet set;
    try {
        set = load_sequences(path, MOTIF_WIDTH);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU MEME EM (timed) --------------------------------------------
    MotifModel model;
    seed_model(set, model);
    std::vector<float> scores_cpu;     // final-model window scores from EM
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    EMResult em = run_meme_em_cpu(set, model, EM_MAX_ITERS, EM_TOL);
    const double cpu_ms = cpu_timer.stop_ms();
    scores_cpu = em.final_scores;      // the array the GPU must reproduce

    // ---- 3. GPU E-step on the FINAL model (kernel timed inside) ------------
    std::vector<float> scores_gpu;
    float gpu_kernel_ms = 0.0f;
    score_windows_gpu(set, model, scores_gpu, &gpu_kernel_ms);

    // ---- 4. Verify GPU window scores == CPU window scores ------------------
    const double err  = util::max_abs_err(scores_cpu, scores_gpu);
    const bool   pass = err <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("MEME OOPS motif discovery: %d sequences, width W=%d, %d windows scored\n",
                set.n, set.w, set.total_windows());
    std::printf("EM converged in %d iterations\n", em.iters);
    std::printf("recovered motif (consensus): %s\n", em.consensus.c_str());
    std::printf("information content: %.4f bits  (max %.1f)\n",
                em.info_content, 2.0 * set.w);
    std::printf("predicted binding site per sequence (0-based offset):\n");
    for (int s = 0; s < set.n; ++s)
        std::printf("  seq[%d]  site offset = %d\n", s, em.best_site[s]);
    std::printf("RESULT: %s (GPU E-step matches CPU exactly, tol=%.0e)\n",
                pass ? "PASS" : "FAIL", TOLERANCE);

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (n=%d sequences, %zu bases, %d windows)\n",
                 path.c_str(), set.n, set.data.size(), set.total_windows());
    std::fprintf(stderr, "[timing] CPU EM (all iters): %.3f ms   GPU E-step kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny sample is dominated by "
                         "launch/copy overhead; the GPU E-step wins at ChIP-seq scale "
                         "(millions of windows).\n");
    std::fprintf(stderr, "[verify] max_abs_err = %.3e  (tolerance %.0e -> exact)\n",
                 err, TOLERANCE);

    return pass ? 0 : 1;
}
