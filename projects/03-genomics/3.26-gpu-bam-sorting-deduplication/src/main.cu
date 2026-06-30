// ===========================================================================
// src/main.cu  --  Entry point: load reads, run CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 3.26 : GPU BAM Sorting & Deduplication
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the aligned reads (data/sample/reads_sample.txt).
//   2. Compute the CPU reference  (reference_cpu.cpp)  -> trusted answer.
//   3. Compute the GPU result     (kernels.cu)         -> the thing taught.
//   4. VERIFY: assert the GPU sort + dedup EQUAL the CPU's, exactly (integers).
//   5. REPORT: deterministic result to stdout; timing to stderr.
//
//   STDOUT is kept byte-for-byte deterministic so demo/run_demo can diff it
//   against demo/expected_output.txt. Anything that varies run-to-run (timings)
//   goes to STDERR, which the demo shows but does not diff.
//
//   This project's verification needs NO tolerance: every comparison is on
//   integers and every order is total (tie-broken on the unique read id), so the
//   GPU and CPU produce byte-identical results. We assert exact equality.
//
// READ THIS FIRST in the code tour, then bam.h -> reference_cpu.h ->
// kernels.cuh -> kernels.cu, and reference_cpu.cpp for the baseline. See
// ../THEORY.md for the "why".
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // sort_gpu, markdup_gpu (GPU path)
#include "reference_cpu.h"    // load_readset, sort_cpu, markdup_cpu (CPU baseline)
#include "util/io.hpp"        // util::CpuTimer

// These two tokens identify the program; they MUST stay in sync with
// demo/expected_output.txt (the first stdout line).
static const char* PROJECT_ID   = "3.26";
static const char* PROJECT_NAME = "GPU BAM Sorting & Deduplication";

// ---------------------------------------------------------------------------
// fnv1a_order -- a tiny, deterministic 64-bit fingerprint of a sorted read
//   sequence. We print this digest (in hex) instead of dumping thousands of
//   reads: it changes if ANY read moves, so it is a compact, reproducible proof
//   that the GPU sort order equals the CPU sort order. FNV-1a is a classic,
//   well-distributed hash; we fold each record's fields in input order.
//   (This is for verification display only -- not a cryptographic hash.)
// ---------------------------------------------------------------------------
static uint64_t fnv1a_order(const std::vector<ReadRecord>& v) {
    uint64_t h = 1469598103934665603ull;             // FNV offset basis
    auto mix = [&](uint64_t x) {
        for (int b = 0; b < 8; ++b) {                // fold 8 bytes of x
            h ^= (x & 0xFF);
            h *= 1099511628211ull;                   // FNV prime
            x >>= 8;
        }
    };
    for (const ReadRecord& r : v) {                  // order-sensitive on purpose
        mix(static_cast<uint64_t>(r.ref_id));
        mix(static_cast<uint64_t>(r.pos));
        mix(static_cast<uint64_t>(r.strand));
        mix(static_cast<uint64_t>(r.mate_pos));
        mix(static_cast<uint64_t>(r.id));
    }
    return h;
}

int main(int argc, char** argv) {
    // ---- 1. Load the aligned reads -----------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/reads_sample.txt";
    ReadSet rs;
    try {
        rs = load_readset(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int n = rs.n();

    // ---- 2. CPU reference (timed) ------------------------------------------
    std::vector<ReadRecord> sorted_cpu;
    std::vector<uint8_t>    isdup_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    sort_cpu(rs, sorted_cpu);
    const int dups_cpu = markdup_cpu(rs, isdup_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU result (kernels timed inside the wrappers) -----------------
    std::vector<ReadRecord> sorted_gpu;
    std::vector<uint8_t>    isdup_gpu;
    float sort_ms = 0.0f, dedup_ms = 0.0f;
    sort_gpu(rs, sorted_gpu, &sort_ms);
    const int dups_gpu = markdup_gpu(rs, isdup_gpu, &dedup_ms);

    // ---- 4. Verify (exact -- all integers, total orders) -------------------
    // (a) Sort order identical: compare the two sorted sequences field by field.
    int sort_mismatch = 0;
    for (int i = 0; i < n; ++i) {
        const ReadRecord& A = sorted_cpu[static_cast<std::size_t>(i)];
        const ReadRecord& B = sorted_gpu[static_cast<std::size_t>(i)];
        if (A.id != B.id || A.ref_id != B.ref_id || A.pos != B.pos ||
            A.strand != B.strand || A.mate_pos != B.mate_pos)
            ++sort_mismatch;
    }
    // (b) Duplicate flags identical, per original read id.
    int dup_mismatch = 0;
    for (int i = 0; i < n; ++i)
        if (isdup_cpu[static_cast<std::size_t>(i)] != isdup_gpu[static_cast<std::size_t>(i)])
            ++dup_mismatch;
    const bool pass = (sort_mismatch == 0) && (dup_mismatch == 0) && (dups_cpu == dups_gpu);

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    const int kept = n - dups_gpu;                 // reads surviving dedup
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("reads: %d aligned across %d references\n", n, rs.num_refs);
    std::printf("coordinate sort: %d reads ordered by (ref, pos, strand, id)\n", n);
    std::printf("sorted-order digest (FNV-1a): %016llx\n",
                static_cast<unsigned long long>(fnv1a_order(sorted_gpu)));
    // Show the first few reads of the sorted output so the order is legible.
    const int show = n < 8 ? n : 8;
    std::printf("first %d sorted reads (ref pos strand mate id):\n", show);
    for (int i = 0; i < show; ++i) {
        const ReadRecord& r = sorted_gpu[static_cast<std::size_t>(i)];
        std::printf("  %2d  %8d  %d  %6d  id=%d\n",
                    r.ref_id, r.pos, r.strand, r.mate_pos, r.id);
    }
    std::printf("mark duplicates: %d duplicates flagged, %d reads kept\n", dups_gpu, kept);
    std::printf("duplicate rate: %d / %d\n", dups_gpu, n);   // exact integers, no %%f
    std::printf("RESULT: %s (GPU sort+dedup match CPU exactly)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%d reads, %d refs)\n",
                 path.c_str(), n, rs.num_refs);
    std::fprintf(stderr, "[timing] CPU sort+dedup: %.3f ms   GPU sort: %.3f ms   GPU dedup: %.3f ms\n",
                 cpu_ms, sort_ms, dedup_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- on this tiny sample the GPU is "
                         "launch/copy bound; the radix-sort win grows with read count (10^8-10^9 real).\n");
    std::fprintf(stderr, "[verify] sort mismatches = %d, dup-flag mismatches = %d, dup count cpu/gpu = %d / %d\n",
                 sort_mismatch, dup_mismatch, dups_cpu, dups_gpu);

    // Exit code feeds the demo's pass/fail gate.
    return pass ? 0 : 1;
}
