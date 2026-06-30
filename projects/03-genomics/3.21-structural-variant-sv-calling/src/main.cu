// ===========================================================================
// src/main.cu  --  Entry point: load reads, call SVs on CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 3.21 : Structural Variant (SV) Calling  (REDUCED-SCOPE teaching version)
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the problem: a reference window + candidate split reads crossing a
//      deletion breakpoint (data/sample, format in data/README.md).
//   2. CPU reference (reference_cpu.cpp): refine breakpoints -> histogram -> calls.
//   3. GPU result (kernels.cu): the SAME, one thread per read + atomic voting.
//   4. VERIFY: the GPU histogram must equal the CPU histogram EXACTLY (integer
//      atomics commute -> no tolerance), and the call lists must match.
//   5. REPORT: deterministic SV calls + "did we recover the planted SV?" to
//      stdout; timing to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Run-varying numbers (timings) go to STDERR, shown
//   but not diffed (PATTERNS.md §3).
//
// Code tour: start here, then sv.h (the shared math), kernels.cuh -> kernels.cu
// (the GPU twin), reference_cpu.cpp (the baseline). See ../THEORY.md for the why.
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // sv_call_gpu (GPU path), SvDataset, SvCall
#include "reference_cpu.h"    // load_dataset, sv_call_cpu (CPU baseline)
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "3.21";
static const char* PROJECT_NAME = "Structural Variant (SV) Calling";

// A breakpoint bin must collect at least this many supporting reads to be called.
// This is the "read support" noise floor every SV caller uses; here it is fixed
// (deterministic) and small because the committed sample is tiny.
static constexpr unsigned int MIN_SUPPORT = 3u;

// ---------------------------------------------------------------------------
// genotype_str: map the integer genotype code to the VCF-style string. Pure
// presentation; the decision was made integer-only in sv_geno_from_vaf (sv.h).
// ---------------------------------------------------------------------------
static const char* genotype_str(int g) {
    switch (g) { case 2: return "1/1"; case 1: return "0/1"; default: return "0/0"; }
}

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/sv_sample.txt";
    SvDataset d;
    try {
        d = load_dataset(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<unsigned int>       hist_cpu;
    std::vector<unsigned long long> lensum_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    const std::vector<SvCall> calls_cpu = sv_call_cpu(d, MIN_SUPPORT, hist_cpu, lensum_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU result (kernel timed inside the wrapper) ------------------
    std::vector<unsigned int>       hist_gpu;
    std::vector<unsigned long long> lensum_gpu;
    float gpu_kernel_ms = 0.0f;
    const std::vector<SvCall> calls_gpu = sv_call_gpu(d, MIN_SUPPORT, hist_gpu, lensum_gpu, &gpu_kernel_ms);

    // ---- 4. Verify (histograms exact; call lists identical) ----------------
    // The headline correctness check: every breakpoint bin's vote count and
    // length-sum must match between CPU and GPU. Because both run the same
    // integer sv.h math and accumulate with commuting integer (atomic) adds,
    // agreement is EXACT -- a single mismatch is a real bug, not float noise.
    int hist_mismatch = 0;
    for (int b = 0; b < d.ref_len; ++b)
        if (hist_cpu[b] != hist_gpu[b] || lensum_cpu[b] != lensum_gpu[b]) ++hist_mismatch;

    bool calls_match = (calls_cpu.size() == calls_gpu.size());
    if (calls_match)
        for (std::size_t i = 0; i < calls_cpu.size(); ++i)
            if (calls_cpu[i].breakpoint != calls_gpu[i].breakpoint ||
                calls_cpu[i].support    != calls_gpu[i].support    ||
                calls_cpu[i].del_len    != calls_gpu[i].del_len    ||
                calls_cpu[i].genotype   != calls_gpu[i].genotype) { calls_match = false; break; }

    const bool pass = (hist_mismatch == 0) && calls_match;

    // Did the top call recover the planted synthetic SV (within the merge radius)?
    bool recovered = false;
    if (d.truth_bp >= 0 && !calls_gpu.empty()) {
        for (const SvCall& c : calls_gpu)
            if (c.breakpoint >= d.truth_bp - SV_MERGE && c.breakpoint <= d.truth_bp + SV_MERGE)
                { recovered = true; break; }
    }

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("reduced-scope teaching version: deletion calling by split-read\n");
    std::printf("realignment (banded SW) + breakpoint clustering on SYNTHETIC data\n");
    std::printf("reference length = %d bp, candidate reads = %d, min support = %u\n",
                d.ref_len, d.N(), MIN_SUPPORT);
    std::printf("SV calls (sorted by breakpoint): %d\n", static_cast<int>(calls_gpu.size()));
    for (const SvCall& c : calls_gpu) {
        // BND-style line: type is DEL in this teaching version; breakpoint, length,
        // supporting-read count, and integer-VAF genotype.
        std::printf("  DEL  bp=%d  len=%d  support=%u  GT=%s\n",
                    c.breakpoint, c.del_len, c.support, genotype_str(c.genotype));
    }
    if (d.truth_bp >= 0) {
        std::printf("planted truth: bp=%d len=%d  -> recovered: %s\n",
                    d.truth_bp, d.truth_len, recovered ? "YES" : "NO");
    }
    std::printf("RESULT: %s (GPU histogram+calls match CPU exactly)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d reads over %d bp reference)\n",
                 path.c_str(), d.N(), d.ref_len);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernel: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- tiny read count is launch-bound; the GPU's "
                         "edge grows toward millions of reads at population scale.\n");
    std::fprintf(stderr, "[verify] histogram-bin mismatches = %d, call lists %s\n",
                 hist_mismatch, calls_match ? "identical" : "DIFFER");

    return pass ? 0 : 1;
}
