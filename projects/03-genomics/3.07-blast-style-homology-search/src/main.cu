// ===========================================================================
// src/main.cu  --  Entry point: load FASTA, search, verify, report
// ---------------------------------------------------------------------------
// Project 3.7 : BLAST-Style Homology Search
//
// The 5-step shape every project in this repo follows:
//   1. Load the problem (a query + N DB protein sequences from data/sample).
//   2. CPU reference  (reference_cpu.cpp)  -> trusted best-HSP scores.
//   3. GPU search     (kernels.cu)         -> the thing being taught.
//   4. VERIFY: GPU agrees with CPU EXACTLY (all-integer scoring -> tolerance 0).
//   5. REPORT: deterministic top-K homology hits to stdout; timing to stderr.
//
// STDOUT is byte-for-byte deterministic (diffed by demo/run_demo against
// demo/expected_output.txt); run-to-run timings go to STDERR.
//
// Code tour: start here, then blast_core.h (the shared scoring), then
// reference_cpu.* (the baseline), then kernels.cuh -> kernels.cu (the GPU).
// ===========================================================================
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "kernels.cuh"        // blast_gpu, SeedPair
#include "reference_cpu.h"    // load_fasta, build_query_index, blast_cpu, SEED_K
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "3.7";
static const char* PROJECT_NAME = "BLAST-Style Homology Search";

// Tolerance: every score is an INTEGER computed by the SAME blast_core.h code on
// both sides, so CPU and GPU agree bit-for-bit. We verify with an EXACT integer
// comparison (no floating point anywhere). See PATTERNS.md sec 4 and THEORY.
static constexpr int TOP_K = 5;   // how many best homology hits to report

// Return indices of the TOP_K largest scores, ties broken by lower index, so the
// ranking is fully deterministic. partial_sort on an index vector.
static std::vector<int> top_k(const std::vector<int>& score, int k) {
    std::vector<int> idx(score.size());
    std::iota(idx.begin(), idx.end(), 0);                 // 0,1,2,...,n-1
    const int kk = std::min<int>(k, static_cast<int>(idx.size()));
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
        [&](int a, int b) {
            if (score[a] != score[b]) return score[a] > score[b];  // higher first
            return a < b;                                          // tie -> lower idx
        });
    idx.resize(kk);
    return idx;
}

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/proteins_sample.fasta";
    SequenceDB db;
    try {
        db = load_fasta(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // Build the query k-mer index ONCE on the host; shared by CPU and GPU so
    // they enumerate identical seeds (CPU uses the hash map; GPU uses the flat
    // sorted form produced inside blast_gpu).
    const QueryIndex query_idx = build_query_index(db.query, SEED_K);

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<int> score_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    blast_cpu(db, query_idx, score_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU search (kernel timed inside the wrapper) ------------------
    std::vector<int> score_gpu;
    float gpu_kernel_ms = 0.0f;
    blast_gpu(db, query_idx, score_gpu, &gpu_kernel_ms);

    // ---- 4. Verify (EXACT: integers, identical math on both sides) --------
    int worst_diff = 0;            // max |cpu - gpu| over all DB sequences
    bool same_shape = (score_cpu.size() == score_gpu.size());
    if (same_shape) {
        for (std::size_t i = 0; i < score_cpu.size(); ++i)
            worst_diff = std::max(worst_diff, std::abs(score_cpu[i] - score_gpu[i]));
    }
    const bool pass = same_shape && (worst_diff == 0);

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    const std::vector<int> best = top_k(score_gpu, TOP_K);
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("query: %s  (len=%d)\n", db.query_name.c_str(),
                static_cast<int>(db.query.size()));
    std::printf("seed-extend search: 1 query vs %d DB sequences "
                "(k=%d, X-drop=%d, BLOSUM62)\n", db.n, SEED_K, X_DROP);
    std::printf("top-%d homology hits (by best ungapped HSP score):\n",
                static_cast<int>(best.size()));
    for (std::size_t r = 0; r < best.size(); ++r)
        std::printf("  #%zu  db[%d]  %-10s  HSP_score = %d\n",
                    r + 1, best[r], db.names[best[r]].c_str(), score_gpu[best[r]]);
    std::printf("RESULT: %s (GPU matches CPU exactly, integer scores)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (query len=%d, %d DB sequences)\n",
                 path.c_str(), static_cast<int>(db.query.size()), db.n);
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny DB is dominated by "
                         "launch/copy overhead; the GPU wins at DB scale (millions of seqs).\n");
    std::fprintf(stderr, "[verify] max integer score diff = %d  (must be 0)\n", worst_diff);

    return pass ? 0 : 1;
}
