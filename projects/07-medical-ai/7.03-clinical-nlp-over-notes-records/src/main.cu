// ===========================================================================
// src/main.cu  --  Entry point: load data, run CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 7.3 : Clinical NLP over Notes & Records
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the tokenized clinical-note batch (data/sample/notes_sample.txt).
//   2. CPU reference (reference_cpu.cpp): one self-attention encoder block.
//   3. GPU result  (kernels.cu): the same block via cuBLAS batched DGEMM + a
//      hand-written softmax kernel.
//   4. VERIFY: GPU attention weights AND output embeddings match the CPU within
//      a documented tolerance (the correctness guarantee).
//   5. REPORT: a DETERMINISTIC, human-readable summary to stdout (per-note top
//      attention link, attention entropy, coreference recovery); timing to
//      stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Anything that varies run-to-run (timings) goes to
//   STDERR, which the demo shows but does not diff (PATTERNS.md §3).
//
// THE SCIENCE IN ONE BREATH
//   Clinical NLP models read free-text notes with a transformer. The heart of a
//   transformer is SELF-ATTENTION: every token computes a weighted average of
//   the other tokens, where the weights ("who should I pay attention to?") come
//   from scaled dot products of learned query/key vectors. This is the
//   GEMM-dominated, O(n²)-in-sequence-length bottleneck the catalog names. Our
//   synthetic batch plants a coreference-like link -- the pronoun "he" is built
//   to attend to "patient" -- so a correct attention block RECOVERS that link.
//   That recovery is the headline, human-meaningful result. (Reduced-scope
//   teaching version; see README "Limitations" and THEORY "real world".)
//
// READ THIS FIRST in the code tour, then attn_core.h (shared math), then
// reference_cpu.cpp (baseline), then kernels.cuh -> kernels.cu (the GPU).
// See ../THEORY.md for the full "why".
// ===========================================================================
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // gpu_attention, GpuAttnTimings
#include "reference_cpu.h"    // load_notes, attention_reference, NoteBatch
#include "attn_core.h"        // attn::attn_entropy, special-token ids
#include "util/io.hpp"        // util::CpuTimer

// Identify the program. Must stay in sync with demo/expected_output.txt.
static const char* PROJECT_ID   = "7.3";
static const char* PROJECT_NAME = "Clinical NLP over Notes & Records";

// ---- Verification tolerances (documented; PATTERNS.md §4) ------------------
// Both sides run the SAME double-precision math from attn_core.h, but the GPU's
// DGEMM sums each dot product in a different ORDER than the CPU's serial loop and
// uses fused multiply-add. Floating-point addition is not associative, so the
// numbers agree to ~1e-12, not bit-exactly -- a real, teachable effect. Both
// tolerances are far below any signal we report. (dh is small here, so the
// mismatch is tiny; it would grow with the head dimension.)
static constexpr double WEIGHT_TOL = 1.0e-11;   // per-entry |A_gpu - A_cpu|
static constexpr double OUT_TOL    = 1.0e-11;   // per-entry |O_gpu - O_cpu|

int main(int argc, char** argv) {
    // ---- 1. Load the tokenized note batch ---------------------------------
    const std::string path =
        (argc > 1) ? argv[1] : "data/sample/notes_sample.txt";
    NoteBatch nb;
    try {
        nb = load_notes(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int B = nb.B, S = nb.S, D = nb.D, H = nb.H, dh = nb.dh();

    // ---- 2. CPU reference (timed) -----------------------------------------
    AttnResult cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    attention_reference(nb, cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU result (kernels timed inside the wrapper) -----------------
    AttnResult gpu;
    GpuAttnTimings gt;
    gpu_attention(nb, gpu, &gt);

    // ---- 4. Verify GPU vs CPU ---------------------------------------------
    // (a) attention weights: worst entrywise difference over [B*H*S*S].
    double w_worst = 0.0;
    for (std::size_t i = 0; i < cpu.weights.size(); ++i)
        w_worst = std::fmax(w_worst, std::fabs(cpu.weights[i] - gpu.weights[i]));
    // (b) output embeddings: worst entrywise difference over [B*S*D].
    double o_worst = 0.0;
    for (std::size_t i = 0; i < cpu.out.size(); ++i)
        o_worst = std::fmax(o_worst, std::fabs(cpu.out[i] - gpu.out[i]));
    const bool pass = (w_worst <= WEIGHT_TOL) && (o_worst <= OUT_TOL);

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    // We use the GPU weights (verified == CPU above) for all printed numbers.
    // For each note, find the pronoun "he" and report which token it attends to
    // most strongly in head 0 -- the coreference-like link. Also report the
    // per-token attention entropy (sharp vs. diffuse focus) for the real tokens.
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("batch: B=%d notes, S=%d tokens, D=%d dim, H=%d heads (dh=%d) [SYNTHETIC]\n",
                B, S, D, H, dh);

    // Locate the vocabulary ids we narrate about (by string, robust to reorder).
    auto find_tok = [&](const std::string& name) -> int {
        for (int t = 0; t < nb.V; ++t) if (nb.vocab[t] == name) return t;
        return -1;
    };
    const int id_he      = find_tok("he");
    const int id_patient = find_tok("patient");

    int coref_hits = 0, coref_total = 0;      // how often "he" -> "patient"
    std::printf("per-note attention (head 0): pronoun 'he' attends most to ->\n");
    for (int b = 0; b < B; ++b) {
        // Find the position of "he" in this note (first occurrence).
        int he_pos = -1;
        for (int s = 0; s < nb.valid_len[b]; ++s)
            if (nb.tok(b, s) == id_he) { he_pos = s; break; }
        if (he_pos < 0) {
            std::printf("  note %d: (no pronoun)\n", b);
            continue;
        }
        // Row of attention weights for query = he_pos, head 0.
        std::size_t base = ((static_cast<std::size_t>(b) * H + 0) * S + he_pos) * S;
        int best_kj = 0; double best_w = -1.0;
        for (int kj = 0; kj < nb.valid_len[b]; ++kj) {
            double w = gpu.weights[base + kj];
            // Deterministic argmax: strictly-greater keeps the lowest index on ties.
            if (w > best_w) { best_w = w; best_kj = kj; }
        }
        int best_tok = nb.tok(b, best_kj);
        coref_total++;
        if (best_tok == id_patient) coref_hits++;
        std::printf("  note %d: 'he'@%d -> '%s'@%d  (weight=%.4f)\n",
                    b, he_pos, nb.vocab[best_tok].c_str(), best_kj, best_w);
    }
    std::printf("coreference link 'he'->'patient' recovered in %d / %d notes\n",
                coref_hits, coref_total);

    // Per-note [CLS]-token summary: entropy of its head-0 attention (how broadly
    // the note-summary token reads the note) and the L2 norm of its output
    // embedding (a deterministic scalar fingerprint of the contextual vector).
    std::printf("per-note [CLS] summary (head 0 attention entropy; output L2 norm):\n");
    for (int b = 0; b < B; ++b) {
        std::size_t base = ((static_cast<std::size_t>(b) * H + 0) * S + 0) * S;  // CLS = pos 0
        // Entropy over all S keys (padding weights are ~0 and contribute ~0).
        std::vector<double> row(gpu.weights.begin() + base,
                                gpu.weights.begin() + base + S);
        double ent = attn::attn_entropy(row.data(), S);
        // L2 norm of the CLS output vector O[b, 0, :].
        std::size_t obase = (static_cast<std::size_t>(b) * S + 0) * D;
        double nrm = 0.0;
        for (int d = 0; d < D; ++d) nrm += gpu.out[obase + d] * gpu.out[obase + d];
        nrm = std::sqrt(nrm);
        std::printf("  note %d: entropy=%.4f nats   ||CLS_out||=%.4f\n", b, ent, nrm);
    }

    std::printf("RESULT: %s (GPU attention matches CPU within tol)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (B=%d, S=%d, D=%d, H=%d)\n",
                 path.c_str(), B, S, D, H);
    std::fprintf(stderr, "[timing] CPU reference (full block): %.3f ms\n", cpu_ms);
    std::fprintf(stderr, "[timing] GPU proj DGEMM=%.3f  scores DGEMM=%.3f  "
                         "softmax=%.3f  ctx DGEMM=%.3f  total=%.3f ms\n",
                 gt.proj_ms, gt.score_ms, gt.soft_ms, gt.ctx_ms, gt.total_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this batch is tiny, so the GPU is "
                         "launch/copy bound; attention's O(S^2) cost is what tensor cores "
                         "and Flash Attention accelerate at real sequence lengths.\n");
    std::fprintf(stderr, "[verify] worst attention-weight diff = %.3e  (tol %.1e)\n",
                 w_worst, WEIGHT_TOL);
    std::fprintf(stderr, "[verify] worst output-embed  diff    = %.3e  (tol %.1e)\n",
                 o_worst, OUT_TOL);

    return pass ? 0 : 1;
}
