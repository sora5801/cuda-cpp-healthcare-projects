// ===========================================================================
// src/main.cu  --  Entry point: load sequences, align, verify, report
// ---------------------------------------------------------------------------
// Project 3.01 : Smith-Waterman / Needleman-Wunsch Alignment
//
// The repo's 5-step shape:
//   1. Load the two sequences (data/sample).
//   2. CPU reference fills the DP matrix (reference_cpu.cpp).
//   3. GPU fills the SAME matrix via the anti-diagonal wavefront (kernels.cu).
//   4. VERIFY: every cell of the GPU matrix equals the CPU matrix.
//   5. REPORT: deterministic score + alignment to stdout; timing to stderr.
//
// We traceback ONCE on the host (from the GPU matrix) to display the alignment;
// the GPU teaching point is the parallel matrix FILL, not the serial traceback.
//
// Code tour: start here, then kernels.cuh -> kernels.cu, then reference_cpu.*.
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // sw_gpu, SeqPair, scoring constants
#include "reference_cpu.h"    // load_sequences, sw_cpu, traceback, Alignment
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "3.1";
static const char* PROJECT_NAME = "Smith-Waterman / Needleman-Wunsch Alignment";

static constexpr int PREVIEW_COLS = 60;   // how many alignment columns to print

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/sequences_sample.txt";
    SeqPair sp;
    try {
        sp = load_sequences(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<int> H_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    sw_cpu(sp, H_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU wavefront (timed) -----------------------------------------
    std::vector<int> H_gpu;
    float gpu_kernel_ms = 0.0f;
    sw_gpu(sp, H_gpu, &gpu_kernel_ms);

    // ---- 4. Verify (the matrices must be identical, exact integers) -------
    int max_abs_diff = 0, mismatches = 0;
    for (std::size_t k = 0; k < H_cpu.size(); ++k) {
        const int d = H_cpu[k] - H_gpu[k];
        const int ad = d < 0 ? -d : d;
        if (ad) { ++mismatches; if (ad > max_abs_diff) max_abs_diff = ad; }
    }
    const bool pass = (mismatches == 0);

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    const Alignment a = traceback(sp, H_gpu);
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("Smith-Waterman local alignment: query (M=%d) vs target (N=%d), "
                "DNA, match=+%d mismatch=%d gap=%d\n", sp.m, sp.n, MATCH, MISMATCH, GAP);
    std::printf("best local score = %d  at cell (i,j)=(%d,%d)\n", a.score, a.end_i, a.end_j);
    if (a.length > 0) {
        const double pct = 100.0 * a.identities / a.length;
        std::printf("aligned length = %d, identities = %d/%d (%.1f%%)\n",
                    a.length, a.identities, a.length, pct);
        const int show = a.length < PREVIEW_COLS ? a.length : PREVIEW_COLS;
        std::printf("alignment (first %d columns):\n", show);
        std::printf("  Q: %s\n", a.q_line.substr(0, show).c_str());
        std::printf("     %s\n", a.m_line.substr(0, show).c_str());
        std::printf("  T: %s\n", a.t_line.substr(0, show).c_str());
    } else {
        std::printf("aligned length = 0 (no positive-scoring local alignment)\n");
    }
    std::printf("RESULT: %s (GPU matrix matches CPU exactly)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (M=%d, N=%d, %zu DP cells)\n",
                 path.c_str(), sp.m, sp.n, H_cpu.size());
    std::fprintf(stderr, "[timing] CPU fill: %.3f ms   GPU wavefront: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- a single small alignment issues many tiny "
                         "diagonal launches; the GPU wins on large matrices / batched pairs.\n");
    std::fprintf(stderr, "[verify] matrix mismatches = %d, max_abs_diff = %d\n",
                 mismatches, max_abs_diff);

    return pass ? 0 : 1;
}
