// ===========================================================================
// src/main.cu  --  Entry point: load reads, map on CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 3.2 : Short-Read Mapping / Alignment
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the problem (reference + reads) from data/sample.
//   2. Build the reference k-mer index ONCE (shared by CPU and GPU).
//   3. Map all reads on the CPU (reference_cpu.cpp)  -> the trusted answer.
//   4. Map all reads on the GPU (kernels.cu)         -> the thing being taught.
//   5. VERIFY: every read's (pos, score) must match the CPU EXACTLY (integers).
//   6. REPORT: a deterministic per-read table to stdout; timing to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Anything that varies run-to-run (timings) goes to
//   STDERR, which the demo shows but does not diff (PATTERNS.md section 3).
//
//   A short CIGAR-like string ("48=2X" = 48 matches then 2 mismatches... we keep
//   it simpler: "<len>M with <mism> mismatches") is printed per read so the
//   learner sees the mapping, not just a number.
//
// READ THIS FIRST in the code tour, then kernels.cuh -> kernels.cu, and
// reference_cpu.cpp for the baseline. See ../THEORY.md for the "why".
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // map_reads_gpu (GPU path), shared core via header
#include "reference_cpu.h"    // load_problem, build_index, map_reads_cpu
#include "util/io.hpp"        // util::CpuTimer

// Program identity. These print on the first stdout line and must stay in sync
// with demo/expected_output.txt.
static const char* PROJECT_ID   = "3.2";
static const char* PROJECT_NAME = "Short-Read Mapping / Alignment";

// How many per-read rows to print (the sample has few reads, so print them all;
// this cap keeps stdout bounded if a learner points the program at a big file).
static constexpr int MAX_ROWS = 32;

int main(int argc, char** argv) {
    // ---- 1. Load the problem -----------------------------------------------
    // Default to the committed sample; allow an override path as argv[1] so the
    // demo and a curious learner can both drive the same binary.
    const std::string path = (argc > 1) ? argv[1]
                                        : "data/sample/reads_sample.txt";
    MappingProblem prob;
    try {
        prob = load_problem(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. Build the reference k-mer index (once, on the host) ------------
    // Both the CPU reference and the GPU consume THIS exact index, so they seed
    // identically -> their candidate positions (and thus results) match.
    const KmerIndex index = build_index(prob);

    // ---- 3. CPU reference mapping (timed) ----------------------------------
    std::vector<MapResult> res_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    map_reads_cpu(prob, index, res_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 4. GPU mapping (kernel timed inside the wrapper) ------------------
    std::vector<MapResult> res_gpu;
    float gpu_kernel_ms = 0.0f;
    map_reads_gpu(prob, index, res_gpu, &gpu_kernel_ms);

    // ---- 5. Verify: EXACT integer agreement on (pos, score) per read -------
    // Because both sides ran the same integer score_window() over the same
    // candidates, any disagreement is a real bug, so we demand pos AND score
    // to be identical. We also count how many reads mapped (pos != NO_HIT).
    int mismatches = 0;     // reads where CPU and GPU disagree
    int mapped     = 0;     // reads that found a position
    for (int r = 0; r < prob.n_reads; ++r) {
        const MapResult& c = res_cpu[static_cast<std::size_t>(r)];
        const MapResult& g = res_gpu[static_cast<std::size_t>(r)];
        if (c.pos != g.pos || c.score != g.score || c.mism != g.mism) ++mismatches;
        if (c.pos != NO_HIT) ++mapped;
    }
    const bool pass = (mismatches == 0);

    // ---- 6a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("seed-and-extend: %d reads (L=%d) vs reference (L_ref=%d), "
                "seed k=%d, match=+%d mismatch=%d\n",
                prob.n_reads, prob.read_len, prob.ref_len, SEED_K, MATCH, MISMATCH);
    std::printf("index: %d reference %d-mers (sorted)\n", index.n_kmers, SEED_K);
    std::printf("per-read mapping (read -> ref pos, score, edits):\n");
    const int rows = prob.n_reads < MAX_ROWS ? prob.n_reads : MAX_ROWS;
    for (int r = 0; r < rows; ++r) {
        const MapResult& g = res_gpu[static_cast<std::size_t>(r)];
        if (g.pos == NO_HIT) {
            // No seed hit: the read's leading k-mer is absent from the reference.
            std::printf("  read %2d -> UNMAPPED (no seed hit)\n", r);
        } else {
            // "<L>M" is a CIGAR-style "L aligned columns, no gaps"; we annotate
            // the mismatch count so the learner reads it as an edit summary.
            std::printf("  read %2d -> pos %4d  score %3d  %dM (%d mismatch%s)\n",
                        r, g.pos, g.score, prob.read_len, g.mism,
                        g.mism == 1 ? "" : "es");
        }
    }
    std::printf("summary: %d/%d reads mapped\n", mapped, prob.n_reads);
    std::printf("RESULT: %s (GPU matches CPU exactly on every read)\n",
                pass ? "PASS" : "FAIL");

    // ---- 6b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (R=%d reads, L=%d, L_ref=%d)\n",
                 path.c_str(), prob.n_reads, prob.read_len, prob.ref_len);
    std::fprintf(stderr, "[timing] CPU mapping: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny batch is "
                         "dominated by launch/copy overhead; the GPU's "
                         "one-thread-per-read parallelism wins at millions of reads.\n");
    std::fprintf(stderr, "[verify] read-level mismatches CPU vs GPU = %d "
                         "(exact integer comparison)\n", mismatches);

    // Exit code feeds the demo's pass/fail gate.
    return pass ? 0 : 1;
}
