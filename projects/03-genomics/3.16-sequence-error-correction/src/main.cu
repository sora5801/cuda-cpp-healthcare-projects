// ===========================================================================
// src/main.cu  --  Entry point: load reads, run CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 3.16 : Sequence Error Correction  (k-mer spectrum / trusted-k-mer)
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the read set (data/sample, FASTA-like; synthetic + labelled).
//   2. CPU reference (reference_cpu.cpp): build spectrum, correct reads.
//   3. GPU result    (kernels.cu)       : same two phases on the device.
//   4. VERIFY: the GPU spectrum and corrected bytes EQUAL the CPU's exactly
//      (every operation is integer/byte work, so we demand bit-identity, ==).
//   5. REPORT: a deterministic summary to stdout; timings to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Timings (which vary run-to-run) go to STDERR.
//
//   A note on the science metric we print: the synthetic sample carries the
//   error-free "truth" for each read, so we can report errors-BEFORE vs
//   errors-AFTER correction -- the headline that shows the method actually works,
//   not just that CPU==GPU.
//
// READ THIS FIRST in the code tour, then kernels.cuh -> kernels.cu, and
// reference_cpu.cpp for the baseline. See ../THEORY.md for the "why".
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // correct_reads_gpu (GPU path), ReadSet, KMER_*
#include "reference_cpu.h"    // load_reads, build_spectrum_cpu, correct_reads_cpu
#include "util/io.hpp"        // util::CpuTimer

// Self-identification; MUST stay in sync with demo/expected_output.txt.
static const char* PROJECT_ID   = "3.16";
static const char* PROJECT_NAME = "Sequence Error Correction";

// Trust threshold T: a k-mer with spectrum count >= T is "trusted" (real). For
// the synthetic sample (coverage ~12x over a short genome) a true 9-mer recurs
// many times while an error 9-mer appears once or twice, so T=3 cleanly
// separates them. This is the single tunable knob of the method (THEORY sec 2).
static constexpr uint32_t TRUST_THRESHOLD = 3;

// How many distinct k-mers exist in the spectrum (count > 0)? A small, stable
// summary number that helps the learner see the spectrum's shape.
static long distinct_kmers(const std::vector<uint32_t>& counts) {
    long d = 0;
    for (uint32_t c : counts) if (c > 0) ++d;
    return d;
}

// Sum of all per-read substitution counts (total bases the corrector changed).
static long total_changes(const std::vector<int>& changes) {
    long s = 0;
    for (int c : changes) s += c;
    return s;
}

// Count exact mismatches between two equal-length byte arrays (spectrum or read
// bytes). Returns -1 on a length mismatch so a shape bug cannot read as "match".
static long byte_mismatches(const std::vector<char>& a, const std::vector<char>& b) {
    if (a.size() != b.size()) return -1;
    long m = 0;
    for (std::size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) ++m;
    return m;
}
static long u32_mismatches(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
    if (a.size() != b.size()) return -1;
    long m = 0;
    for (std::size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) ++m;
    return m;
}

int main(int argc, char** argv) {
    // ---- 1. Load the read set ----------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/reads_sample.txt";
    ReadSet reads;
    try {
        reads = load_reads(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (both phases, timed) -----------------------------
    std::vector<uint32_t> counts_cpu;
    std::vector<char>     corrected_cpu;
    std::vector<int>      changes_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    build_spectrum_cpu(reads, counts_cpu);
    correct_reads_cpu(reads, counts_cpu, TRUST_THRESHOLD, corrected_cpu, changes_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU result (both phases, each kernel timed in the wrapper) -----
    std::vector<uint32_t> counts_gpu;
    std::vector<char>     corrected_gpu;
    std::vector<int>      changes_gpu;
    float count_ms = 0.0f, correct_ms = 0.0f;
    correct_reads_gpu(reads, TRUST_THRESHOLD, counts_gpu, corrected_gpu,
                      changes_gpu, &count_ms, &correct_ms);

    // ---- 4. Verify: GPU must EQUAL CPU exactly (integer/byte work) ---------
    const long spectrum_diff  = u32_mismatches(counts_cpu, counts_gpu);
    const long corrected_diff = byte_mismatches(corrected_cpu, corrected_gpu);
    const bool pass = (spectrum_diff == 0) && (corrected_diff == 0);

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    // Science metric: how many erroneous bases existed before vs after.
    // errors_before = corrected==truth? No -- compare the RAW reads to truth.
    long errors_before = -1, errors_after = -1;
    if (reads.has_truth) {
        errors_before = count_residual_errors(reads, reads.bases);     // raw vs truth
        errors_after  = count_residual_errors(reads, corrected_gpu);   // fixed vs truth
    }

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("k-mer spectrum error correction (k=%d, trust T=%u)\n",
                KMER_K, TRUST_THRESHOLD);
    std::printf("reads: %d   total bases: %zu\n",
                reads.n, reads.bases.size());
    std::printf("spectrum: %ld distinct %d-mers observed\n",
                distinct_kmers(counts_gpu), KMER_K);
    std::printf("corrections applied: %ld base(s) over %d read(s)\n",
                total_changes(changes_gpu), reads.n);
    if (reads.has_truth) {
        std::printf("errors vs truth:  before = %ld   after = %ld   (removed %ld)\n",
                    errors_before, errors_after, errors_before - errors_after);
    }
    std::printf("verify: spectrum_mismatch=%ld  corrected_mismatch=%ld\n",
                spectrum_diff, corrected_diff);
    std::printf("RESULT: %s (GPU matches CPU exactly: spectrum + corrected reads)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (n=%d reads, has_truth=%d)\n",
                 path.c_str(), reads.n, reads.has_truth ? 1 : 0);
    std::fprintf(stderr, "[timing] CPU (both phases): %.3f ms\n", cpu_ms);
    std::fprintf(stderr, "[timing] GPU phase1 count: %.3f ms   phase2 correct: %.3f ms\n",
                 count_ms, correct_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny sample is dominated "
                         "by launch/copy overhead; the GPU wins at real scale (millions of reads).\n");
    std::fprintf(stderr, "[verify] spectrum slots compared: %u   corrected bytes compared: %zu\n",
                 KMER_TABLE_N, corrected_gpu.size());

    return pass ? 0 : 1;
}
