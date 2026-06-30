// ===========================================================================
// src/main.cu  --  Entry point: load RNA, fold on CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 3.10 : RNA Secondary-Structure Prediction  (Nussinov base-pair DP)
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load one RNA sequence (data/sample, or a built-in synthetic fallback).
//   2. CPU reference fills the Nussinov matrix (reference_cpu.cpp).
//   3. GPU fills the SAME matrix via the anti-diagonal wavefront (kernels.cu).
//   4. VERIFY: every upper-triangle cell of the GPU matrix equals the CPU's
//      (exact integer equality -- the math is identical, see PATTERNS §4).
//   5. REPORT: deterministic max-pair count + dot-bracket structure to stdout;
//      timing to stderr.
//
//   We traceback ONCE on the host (from the GPU matrix) to display the folding;
//   the GPU teaching point is the parallel matrix FILL, not the serial traceback.
//   STDOUT is kept byte-for-byte deterministic so demo/run_demo can diff it.
//
// Code tour: start here, then kernels.cuh -> kernels.cu, then reference_cpu.*.
// See ../THEORY.md for the science and the "why".
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // nussinov_gpu, RnaSeq (GPU path)
#include "reference_cpu.h"    // load_rna, nussinov_cpu, traceback (CPU baseline)
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "3.10";
static const char* PROJECT_NAME = "RNA Secondary-Structure Prediction (Nussinov)";

// ---------------------------------------------------------------------------
// make_synthetic: the built-in problem used when no data file is supplied.
//   A designed 18-nt RNA with a clear hairpin: a GC-rich stem closing a small
//   AAAA loop, with two unpaired bases at the 3' end. It folds to a known,
//   easily-checked structure -- 6 base pairs, "((((((....))))))..".  That is
//   what stdout reports. (The committed data/sample file holds this same
//   sequence; see data/README.)
// ---------------------------------------------------------------------------
static RnaSeq make_synthetic() {
    RnaSeq r;
    const std::string seq = "GGGCGCAAAAGCGCCCAU";   // synthetic teaching hairpin
    for (char c : seq) {
        uint8_t code = 0;
        switch (c) { case 'A': code=0; break; case 'C': code=1; break;
                     case 'G': code=2; break; default:  code=3; break; }  // U
        r.s.push_back(code);
        r.raw.push_back(c);
    }
    r.n = static_cast<int>(r.s.size());
    return r;
}

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    RnaSeq r;
    const char* source = "synthetic (built-in)";
    if (argc > 1) {
        try {
            r = load_rna(argv[1]);     // first usable line of the sample file
            source = argv[1];
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[error] %s\n", e.what());
            return 2;
        }
    } else {
        r = make_synthetic();
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<int> M_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    nussinov_cpu(r, M_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU wavefront (timed) -----------------------------------------
    std::vector<int> M_gpu;
    float gpu_kernel_ms = 0.0f;
    nussinov_gpu(r, M_gpu, &gpu_kernel_ms);

    // ---- 4. Verify: the two matrices must be identical integers ------------
    // Only the upper triangle (i < j) carries meaning; we still scan the whole
    // matrix because the lower triangle and diagonal are 0 on both sides too.
    int mismatches = 0, max_abs_diff = 0;
    for (std::size_t k = 0; k < M_cpu.size(); ++k) {
        const int d = M_cpu[k] - M_gpu[k];
        const int ad = d < 0 ? -d : d;
        if (ad) { ++mismatches; if (ad > max_abs_diff) max_abs_diff = ad; }
    }
    const bool pass = (mismatches == 0);

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    // Traceback on the GPU matrix to show the predicted folding (dot-bracket).
    const Structure st = traceback(r, M_gpu);
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("RNA length n = %d  (alphabet ACGU, min hairpin loop = %d)\n", r.n, MIN_LOOP);
    std::printf("sequence : %s\n", r.raw.c_str());
    std::printf("structure: %s\n", st.dot_bracket.c_str());
    std::printf("max base pairs = %d\n", st.pairs);
    std::printf("RESULT: %s (GPU matrix matches CPU exactly)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR --------------------------------------
    const std::size_t cells = static_cast<std::size_t>(r.n) * r.n;
    std::fprintf(stderr, "[data]   source: %s  (n=%d, %zu DP cells)\n",
                 source, r.n, cells);
    std::fprintf(stderr, "[timing] CPU fill: %.3f ms   GPU wavefront: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- a short RNA issues many tiny per-span "
                         "launches; the GPU wins on long RNAs and on batched sequences.\n");
    std::fprintf(stderr, "[verify] matrix mismatches = %d, max_abs_diff = %d\n",
                 mismatches, max_abs_diff);

    return pass ? 0 : 1;
}
