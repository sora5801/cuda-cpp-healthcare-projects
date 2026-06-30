// ===========================================================================
// src/main.cu  --  Entry point: load reads, align on CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 3.23 : Splice-Aware RNA Alignment   (REDUCED-SCOPE teaching version)
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load a batch: one reference "gene model" + several RNA-seq reads
//      (data/sample/reads_sample.txt).
//   2. CPU reference aligns every read with the spliced DP (reference_cpu.cpp).
//   3. GPU aligns every read with the SAME shared recurrence (kernels.cu).
//   4. VERIFY: GPU score, endpoint, AND every DP-table cell equal the CPU's,
//      to the integer (tolerance is EXACT == 0 -- integer DP, docs/PATTERNS §4).
//   5. REPORT: per-read CIGAR-with-N + a junction summary to stdout (diffed);
//      timings to stderr (shown, not diffed).
//
//   We traceback ONCE on the host, from the GPU's DP tables, to print each
//   read's CIGAR. The GPU teaching point is the parallel batched FILL, not the
//   serial traceback (identical to 3.01's choice).
//
//   Code tour: read this first, then kernels.cuh -> kernels.cu, then
//   reference_cpu.h (the shared recurrence) -> reference_cpu.cpp. See THEORY.md.
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // align_batch_gpu (GPU path) + shared types
#include "reference_cpu.h"    // load_batch, align_batch_cpu, traceback_cigar
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "3.23";
static const char* PROJECT_NAME = "Splice-Aware RNA Alignment";

// Decode a base code 0..3 back to its letter, for printing the reference window.
static char base_char(uint8_t c) { return (c < 4) ? ALPHABET[c] : 'N'; }

int main(int argc, char** argv) {
    // ---- 1. Load the batch --------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/reads_sample.txt";
    ReadBatch batch;
    try {
        batch = load_batch(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) ------------------------------------------
    std::vector<AlignResult> res_cpu;
    std::vector<int> H_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    align_batch_cpu(batch, res_cpu, H_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU batched alignment (kernel timed inside the wrapper) --------
    std::vector<AlignResult> res_gpu;
    std::vector<int> H_gpu;
    float gpu_kernel_ms = 0.0f;
    align_batch_gpu(batch, res_gpu, H_gpu, &gpu_kernel_ms);

    // ---- 4. Verify (EXACT integer agreement: scores, endpoints, ALL cells) -
    int score_mismatches = 0, endpoint_mismatches = 0;
    long long cell_mismatches = 0, max_abs_cell_diff = 0;
    for (int r = 0; r < batch.num_reads; ++r) {
        if (res_cpu[r].score != res_gpu[r].score) ++score_mismatches;
        if (res_cpu[r].end_i != res_gpu[r].end_i ||
            res_cpu[r].end_j != res_gpu[r].end_j) ++endpoint_mismatches;
    }
    for (std::size_t k = 0; k < H_cpu.size(); ++k) {
        const long long d = (long long)H_cpu[k] - H_gpu[k];
        const long long ad = d < 0 ? -d : d;
        if (ad) { ++cell_mismatches; if (ad > max_abs_cell_diff) max_abs_cell_diff = ad; }
    }
    const bool pass = (score_mismatches == 0) && (endpoint_mismatches == 0)
                      && (cell_mismatches == 0);

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("reference gene model: N=%d bases, reads=%d, max read len=%d\n",
                batch.n, batch.num_reads, batch.read_len);
    std::printf("scoring: match=+%d mismatch=%d gap=%d intron_open=%d "
                "canonical(GT-AG)_bonus=+%d\n",
                MATCH, MISMATCH, GAP, INTRON_OPEN, CANON_BONUS);
    std::printf("per-read spliced alignment (CIGAR uses N for intron skips):\n");

    // For each read: print its best score, endpoint, intron count, and CIGAR.
    // We traceback on the GPU tables so the printed CIGAR also proves the GPU
    // table is correct (it must equal the CPU's, checked above).
    int total_introns = 0, reads_with_intron = 0;
    for (int r = 0; r < batch.num_reads; ++r) {
        int introns = 0, matched = 0;
        const std::string cig =
            traceback_cigar(batch, r, H_gpu, res_gpu[r], introns, matched);
        total_introns += introns;
        if (introns > 0) ++reads_with_intron;
        std::printf("  read %2d: len=%3d score=%3d end=(i=%3d,j=%3d) "
                    "introns=%d  CIGAR=%s\n",
                    r, batch.read_lens[r], res_gpu[r].score,
                    res_gpu[r].end_i, res_gpu[r].end_j, introns, cig.c_str());
    }
    std::printf("junction summary: %d/%d reads cross >=1 intron, "
                "%d intron(s) detected total\n",
                reads_with_intron, batch.num_reads, total_introns);

    // Show a short window of the reference around the first detected junction so
    // the GT..AG canonical sites are visible to the learner (deterministic).
    if (batch.n >= 8) {
        std::printf("reference[0:%d] = ", batch.n < 60 ? batch.n : 60);
        const int show = batch.n < 60 ? batch.n : 60;
        for (int j = 0; j < show; ++j) std::printf("%c", base_char(batch.ref[j]));
        std::printf("\n");
    }
    std::printf("RESULT: %s (GPU matches CPU exactly: scores, endpoints, all DP cells)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (N=%d, R=%d, M=%d)\n",
                 path.c_str(), batch.n, batch.num_reads, batch.read_len);
    std::fprintf(stderr, "[timing] CPU align: %.3f ms   GPU batched kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- a few short reads barely fill the GPU; "
                         "the batched-blocks design pays off at millions of reads.\n");
    std::fprintf(stderr, "[verify] score_mismatches=%d endpoint_mismatches=%d "
                         "cell_mismatches=%lld max_abs_cell_diff=%lld\n",
                 score_mismatches, endpoint_mismatches,
                 cell_mismatches, max_abs_cell_diff);

    return pass ? 0 : 1;
}
