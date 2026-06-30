// ===========================================================================
// src/main.cu  --  Entry point: load sequences, align all-vs-all, assemble, verify
// ---------------------------------------------------------------------------
// Project 3.8 : Multiple Sequence Alignment (MSA)
//
// The repo's 5-step shape, specialised to progressive MSA:
//   1. Load N DNA sequences (data/sample, a multi-FASTA file).
//   2. CPU reference: STAGE 1 pairwise NW score matrix (reference_cpu.cpp).
//   3. GPU: the SAME score matrix, one thread block per pair (kernels.cu).
//   4. VERIFY: the GPU score matrix equals the CPU score matrix, every cell,
//      exactly (integer scores -> bit-identical; tolerance is literally 0).
//   5. STAGES 2-3 + REPORT: from the (identical) matrix, build the center-star
//      progressive alignment and print it deterministically to stdout; send all
//      timing / run-varying numbers to stderr.
//
// The GPU teaching point is STAGE 1 (the embarrassingly-parallel all-vs-all NW
// scoring). STAGES 2-3 are deterministic host bookkeeping done once on the shared
// matrix -- so the printed multiple alignment is the same whether we feed it the
// CPU or the GPU matrix (they are equal).
//
// Code tour: start here, then nw_core.h -> kernels.cuh -> kernels.cu, then
// reference_cpu.* for the loader/assembly.
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // distance_matrix_gpu (STAGE 1 on the GPU)
#include "reference_cpu.h"    // load_fasta, distance_matrix_cpu, build_msa, MSA
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "3.8";
static const char* PROJECT_NAME = "Multiple Sequence Alignment (MSA)";

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/sequences_sample.fasta";
    SeqSet s;
    try {
        s = load_fasta(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference: STAGE 1 pairwise score matrix (timed) -----------
    std::vector<int>    score_cpu;
    std::vector<double> D_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    distance_matrix_cpu(s, score_cpu, D_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU: STAGE 1, one block per pair (timed via CUDA events) -------
    std::vector<int>    score_gpu;
    std::vector<double> D_gpu;
    float gpu_kernel_ms = 0.0f;
    distance_matrix_gpu(s, score_gpu, D_gpu, &gpu_kernel_ms);

    // ---- 4. Verify: the two score matrices must be IDENTICAL (exact ints) --
    int mismatches = 0, max_abs_diff = 0;
    for (std::size_t k = 0; k < score_cpu.size(); ++k) {
        const int d = score_cpu[k] - score_gpu[k];
        const int ad = d < 0 ? -d : d;
        if (ad) { ++mismatches; if (ad > max_abs_diff) max_abs_diff = ad; }
    }
    const bool pass = (mismatches == 0);

    // ---- 5. STAGES 2-3: build the multiple alignment from the GPU matrix ----
    //     (The CPU and GPU matrices are equal, so the alignment is the same; we
    //      use the GPU's to demonstrate the full GPU-driven pipeline.)
    const MSA msa = build_msa(s, D_gpu);

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("input: %d DNA sequences, max length %d; scoring match=+%d mismatch=%d gap=%d\n",
                s.n, s.max_len, NW_MATCH, NW_MISMATCH, NW_GAP);
    std::printf("pairwise NW alignments (STAGE 1) = %d  (one GPU block each)\n",
                s.n * (s.n - 1) / 2);
    std::printf("center-star sequence (STAGE 2) = index %d (\"%s\")\n",
                msa.center, s.names[msa.center].c_str());
    std::printf("multiple alignment (STAGE 3): %d rows x %d columns, Sum-of-Pairs score = %lld\n",
                msa.n, msa.width, msa.sp_score);

    // Print the alignment block. Row label = sequence name (truncated), then the
    // aligned row. Deterministic by construction (no RNG, fixed tie-breaks).
    std::printf("\n");
    for (int i = 0; i < msa.n; ++i) {
        // 10-char left-justified name field for a tidy, fixed-width block.
        char label[11];
        std::snprintf(label, sizeof(label), "%-10.10s", s.names[i].c_str());
        std::printf("%s %s%s\n", label, msa.rows[i].c_str(),
                    (i == msa.center) ? "  <- center" : "");
    }

    // A per-column conservation marker line ('*' = all rows identical & not gap).
    std::string stars(static_cast<std::size_t>(msa.width), ' ');
    for (int col = 0; col < msa.width; ++col) {
        char c0 = msa.rows[0][col];
        bool all_same = (c0 != '-');
        for (int i = 1; i < msa.n && all_same; ++i)
            if (msa.rows[i][col] != c0) all_same = false;
        if (all_same) stars[col] = '*';
    }
    std::printf("%-10.10s %s\n", "conserv.", stars.c_str());

    std::printf("\nRESULT: %s (GPU pairwise-score matrix matches CPU exactly)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (n=%d sequences, %zu residues total)\n",
                 path.c_str(), s.n, s.data.size());
    std::fprintf(stderr, "[timing] CPU STAGE-1 matrix: %.3f ms   GPU STAGE-1 kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- tiny N here means many small blocks; the\n"
                         "         GPU's edge grows with N (the O(N^2) pair count) and sequence length.\n");
    std::fprintf(stderr, "[verify] score-matrix mismatches = %d, max_abs_diff = %d (tolerance = 0)\n",
                 mismatches, max_abs_diff);

    return pass ? 0 : 1;
}
