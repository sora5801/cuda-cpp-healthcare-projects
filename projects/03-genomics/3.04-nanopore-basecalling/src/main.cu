// ===========================================================================
// src/main.cu  --  Entry point: load posteriors, decode, verify, report
// ---------------------------------------------------------------------------
// Project 3.4 : Nanopore Basecalling  (REDUCED-SCOPE: CTC greedy decode)
//
// The 5-step shape every project in this repo follows:
//   1. Load the problem (a batch of reads' posterior matrices from data/sample).
//   2. CPU reference  (reference_cpu.cpp) -> trusted decoded sequences.
//   3. GPU decode     (kernels.cu)        -> the thing being taught.
//   4. VERIFY: GPU base strings + checksums match the CPU EXACTLY (tol == 0,
//      because both call the identical ctc_core.h decode -- PATTERNS.md sec 4).
//   5. REPORT: deterministic per-read summary to stdout; timing to stderr.
//
// STDOUT is byte-for-byte deterministic (diffed by demo/run_demo against
// demo/expected_output.txt); run-to-run timings go to STDERR.
//
// Code tour: start here, then ctc_core.h (the shared decode), kernels.cuh ->
// kernels.cu (the GPU path), then reference_cpu.* (the baseline + loader).
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // basecall_gpu, ReadSet, DecodedRead
#include "reference_cpu.h"    // load_reads, basecall_cpu
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "3.4";
static const char* PROJECT_NAME = "Nanopore Basecalling (CTC greedy decode)";

// How many reads' decoded sequences to print in full in the stdout report. The
// rest are summarized by count so the output stays compact and deterministic.
static constexpr int SHOW_READS = 4;

// Compare CPU vs GPU decoded results read-by-read. Returns the number of reads
// whose (length, base string, checksum) ALL match. Because both sides run the
// identical integer-only ctc_core decode, a correct run matches exactly --
// there is no floating-point tolerance to tune (PATTERNS.md sec 4: exact case).
static int count_matches(const std::vector<DecodedRead>& a,
                         const std::vector<DecodedRead>& b) {
    if (a.size() != b.size()) return -1;          // shape bug -> sentinel
    int matches = 0;
    for (std::size_t r = 0; r < a.size(); ++r) {
        const bool same = a[r].length   == b[r].length
                       && a[r].checksum == b[r].checksum
                       && a[r].base_seq == b[r].base_seq;
        if (same) ++matches;
    }
    return matches;
}

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1]
                                        : "data/sample/reads_sample.txt";
    ReadSet rs;
    try {
        rs = load_reads(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<DecodedRead> dec_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    basecall_cpu(rs, dec_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU decode (kernel timed inside the wrapper) ------------------
    std::vector<DecodedRead> dec_gpu;
    float gpu_kernel_ms = 0.0f;
    basecall_gpu(rs, dec_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    const int matches = count_matches(dec_cpu, dec_gpu);
    const bool pass    = (matches == rs.n_reads);

    // Total called bases across the batch (a deterministic integer summary).
    long long total_bases = 0;
    for (const auto& d : dec_gpu) total_bases += d.length;

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("decoded %d reads from posterior matrices (C=%d classes: -ACGT)\n",
                rs.n_reads, CTC_NUM_CLASSES);
    const int show = rs.n_reads < SHOW_READS ? rs.n_reads : SHOW_READS;
    for (int r = 0; r < show; ++r) {
        const DecodedRead& d = dec_gpu[static_cast<std::size_t>(r)];
        std::printf("  read %d: T=%d  len=%d  checksum=%08x  seq=%s\n",
                    r, rs.T[static_cast<std::size_t>(r)], d.length, d.checksum,
                    d.base_seq.c_str());
    }
    if (rs.n_reads > show)
        std::printf("  ... (%d more reads not shown)\n", rs.n_reads - show);
    std::printf("total called bases (all reads): %lld\n", total_bases);
    std::printf("CPU/GPU agreement: %d/%d reads identical\n", matches, rs.n_reads);
    std::printf("RESULT: %s (GPU matches CPU exactly; tol = 0, integer decode)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (n_reads=%d, max_T=%d, total_steps=%d)\n",
                 path.c_str(), rs.n_reads, rs.max_T,
                 rs.offset.empty() ? 0 : rs.offset.back());
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny batch is dominated "
                         "by launch/copy overhead; the GPU wins at run scale (millions of "
                         "reads decoded concurrently).\n");
    std::fprintf(stderr, "[note]   the neural network that PRODUCES these posteriors is "
                         "out of scope here (research-grade); we decode its output. See THEORY.\n");

    return pass ? 0 : 1;
}
