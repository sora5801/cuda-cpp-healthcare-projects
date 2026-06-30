// ===========================================================================
// src/main.cu  --  Entry point: build SA/BWT/FM on CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 3.27 : Suffix Array / BWT / FM-Index Construction
//
// THE 5-STEP SHAPE every project in this repo follows:
//   1. LOAD   the DNA text (data/sample), append the '$' sentinel.
//   2. CPU    reference: suffix_array_cpu() -> trusted SA + BWT + FM count.
//   3. GPU    result:    suffix_array_gpu() -> the thing being taught (prefix
//             doubling with a hand-rolled radix sort).
//   4. VERIFY GPU SA == CPU SA exactly (it is an integer permutation, so the
//             tolerance is ZERO), plus BWT and FM-count agreement.
//   5. REPORT a deterministic summary to STDOUT (diffed by the demo); timings
//             and run-varying detail go to STDERR (shown, not diffed).
//
// Code tour: start here, then kernels.cuh -> kernels.cu (the GPU radix-sort
//   doubling), then reference_cpu.cpp (the serial baseline), with sa_core.h as
//   the shared key math both sides call. See ../THEORY.md for the "why".
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // suffix_array_gpu, SaResult
#include "reference_cpu.h"    // load_text, suffix_array_cpu, SaResult
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "3.27";
static const char* PROJECT_NAME = "Suffix Array / BWT / FM-Index Construction";

// The query whose occurrences we count via FM-index backward search. The
// synthetic sample plants the motif "ACGT" repeatedly, so "ACG" recurs a known
// number of times -- a non-trivial, eyeball-checkable answer (see data/README.md).
static const char* QUERY_PATTERN = "ACG";

// Count positions where the GPU suffix array differs from the CPU one. Zero
// means an EXACT match; since both are integer permutations there is no
// floating point involved, so the only acceptable result is 0.
static int sa_mismatches(const std::vector<int>& a, const std::vector<int>& b) {
    if (a.size() != b.size()) return -1;            // shape bug: report distinctly
    int diff = 0;
    for (std::size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) ++diff;
    return diff;
}

int main(int argc, char** argv) {
    // ---- 1. Load ----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/dna_sample.txt";
    std::string text;
    try {
        text = load_text(path);                     // appends '$'
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int n = static_cast<int>(text.size());
    const std::string pattern = QUERY_PATTERN;

    // ---- 2. CPU reference (timed) -----------------------------------------
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    const SaResult cpu = suffix_array_cpu(text, pattern);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU result (kernel-timed inside the wrapper) ------------------
    float gpu_kernel_ms = 0.0f;
    const SaResult gpu = suffix_array_gpu(text, pattern, &gpu_kernel_ms);

    // ---- 4. Verify --------------------------------------------------------
    const int    mism   = sa_mismatches(cpu.sa, gpu.sa);     // must be 0
    const bool   bwt_ok = (cpu.bwt == gpu.bwt);
    const bool   fm_ok  = (cpu.pattern_count == gpu.pattern_count);
    const bool   pass   = (mism == 0) && bwt_ok && fm_ok;

    // ---- 5a. Deterministic report -> STDOUT (diffed) ----------------------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("text length (with $ sentinel): %d\n", n);
    // Show the first few suffix-array entries so the learner sees a concrete SA.
    const int show = (n < 12) ? n : 12;
    std::printf("suffix array SA[0:%d] =", show);
    for (int i = 0; i < show; ++i) std::printf(" %d", gpu.sa[i]);
    std::printf("\n");
    // The first 32 characters of the BWT (the transform "block-sorts" the text).
    const int bshow = (n < 32) ? n : 32;
    std::printf("BWT[0:%d] = %s\n", bshow, gpu.bwt.substr(0, bshow).c_str());
    std::printf("FM-index backward search: pattern \"%s\" occurs %d time(s)\n",
                pattern.c_str(), gpu.pattern_count);
    std::printf("verify: SA mismatches=%d  BWT match=%s  FM match=%s\n",
                mism, bwt_ok ? "yes" : "no", fm_ok ? "yes" : "no");
    std::printf("RESULT: %s (GPU suffix array matches CPU exactly, tol=0)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) -----------------
    std::fprintf(stderr, "[data]   source: %s  (%d bases + 1 sentinel)\n", path.c_str(), n - 1);
    std::fprintf(stderr, "[algo]   prefix-doubling rounds: CPU=%d  GPU=%d\n",
                 cpu.doubling_rounds, gpu.doubling_rounds);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU doubling kernels: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- at n=%d the GPU is launch-bound; "
                         "the radix-sort win appears at genome scale (millions of bases).\n", n);
    std::fprintf(stderr, "[verify] SA permutation length %d, exact integer match required.\n", n);

    return pass ? 0 : 1;
}
