// ===========================================================================
// src/main.cu  --  Entry point: load problem, run CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 2.13 : MSA Generation Acceleration
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the problem (one query profile HMM + a database of N sequences).
//   2. Compute the CPU reference (reference_cpu.cpp)        -> trusted scores.
//   3. Compute the GPU result    (kernels.cu)               -> the thing taught.
//   4. VERIFY: assert GPU agrees with CPU EXACTLY            -> correctness.
//   5. REPORT: deterministic top-K hits to stdout; timing to stderr.
//
//   STDOUT is kept byte-for-byte deterministic so demo/run_demo can diff it
//   against demo/expected_output.txt. Anything that varies run-to-run (timings)
//   goes to STDERR, which the demo shows but does not diff.
//
//   WHY EXACT (tolerance == 0): every score is an integer (scaled log-odds, see
//   hmm_core.h SCORE_SCALE), and CPU and GPU run the SAME integer recurrence
//   (viterbi_step). Integer max/add is associative and order-independent, so the
//   two paths agree bit-for-bit -- no floating-point drift (PATTERNS.md §4).
//
// READ THIS FIRST in the code tour, then hmm_core.h, kernels.cuh -> kernels.cu,
// and reference_cpu.cpp for the baseline. See ../THEORY.md for the "why".
// ===========================================================================
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "kernels.cuh"        // viterbi_search_gpu (GPU path), MAX_PROFILE_L
#include "reference_cpu.h"    // load_problem, viterbi_search_cpu (CPU baseline)
#include "hmm_core.h"         // SCORE_SCALE, NEG_INF
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "2.13";
static const char* PROJECT_NAME = "MSA Generation Acceleration";

// How many top hits to report. The MSA would keep the high-scoring database
// sequences; here we print the best few so the result is small + deterministic.
static constexpr int TOP_K = 5;

// ---------------------------------------------------------------------------
// top_k_indices : indices of the K largest scores, ties broken by LOWER index.
//   Deterministic ranking (so stdout is reproducible) via partial_sort on an
//   index vector -- identical idiom to flagship 1.12.
// ---------------------------------------------------------------------------
static std::vector<int> top_k_indices(const std::vector<int>& score, int k) {
    std::vector<int> idx(score.size());
    std::iota(idx.begin(), idx.end(), 0);                  // 0,1,2,...,N-1
    const int kk = std::min<int>(k, static_cast<int>(idx.size()));
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
        [&](int a, int b) {
            if (score[a] != score[b]) return score[a] > score[b];   // higher first
            return a < b;                                           // tie -> lower idx
        });
    idx.resize(kk);
    return idx;
}

int main(int argc, char** argv) {
    // ---- 1. Load the problem -----------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/profile_db_sample.txt";
    SearchProblem prob;
    try {
        prob = load_problem(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    if (prob.hmm.L > MAX_PROFILE_L) {
        std::fprintf(stderr, "[error] profile length L=%d exceeds MAX_PROFILE_L=%d "
                             "(raise the cap in kernels.cuh and rebuild)\n",
                     prob.hmm.L, MAX_PROFILE_L);
        return 2;
    }

    // ---- 2. CPU reference (timed) ------------------------------------------
    std::vector<int> score_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    viterbi_search_cpu(prob, score_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU search (kernel timed inside the wrapper) -------------------
    std::vector<int> score_gpu;
    float gpu_kernel_ms = 0.0f;
    viterbi_search_gpu(prob, score_gpu, &gpu_kernel_ms);

    // ---- 4. Verify (EXACT integer agreement) -------------------------------
    //   Largest absolute difference between the two integer score vectors. For a
    //   correct implementation this is exactly 0 (same integer recurrence).
    long long worst = 0;
    bool same_size = (score_cpu.size() == score_gpu.size());
    if (same_size) {
        for (std::size_t i = 0; i < score_cpu.size(); ++i) {
            long long d = std::llabs(static_cast<long long>(score_cpu[i]) -
                                     static_cast<long long>(score_gpu[i]));
            if (d > worst) worst = d;
        }
    }
    const bool pass = same_size && (worst == 0);

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    //   Report the top-K database hits by GPU score, converting the scaled
    //   integer back to a human-readable log-odds (bits-ish) for display. The
    //   DISPLAYED float is derived deterministically from the integer score, so
    //   stdout stays byte-identical across runs.
    const std::vector<int> best = top_k_indices(score_gpu, TOP_K);
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("Profile-HMM Viterbi search: 1 query profile (L=%d) vs %d database sequences\n",
                prob.hmm.L, prob.db.N);
    std::printf("top-%d hits (by Viterbi log-odds score):\n", static_cast<int>(best.size()));
    for (std::size_t r = 0; r < best.size(); ++r) {
        const int i = best[r];
        const double logodds = static_cast<double>(score_gpu[i]) / SCORE_SCALE;
        std::printf("  #%zu  seq[%d]  score = %d  (log-odds = %.3f)\n",
                    r + 1, i, score_gpu[i], logodds);
    }
    std::printf("RESULT: %s (GPU matches CPU exactly; max |diff| = %lld)\n",
                pass ? "PASS" : "FAIL", worst);

    // ---- 5b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (L=%d profile, N=%d sequences)\n",
                 path.c_str(), prob.hmm.L, prob.db.N);
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny database is dominated by "
                         "launch/copy overhead; the GPU's edge grows with database size "
                         "(real searches scan hundreds of millions of sequences).\n");
    std::fprintf(stderr, "[verify] max |CPU-GPU| score difference = %lld  (must be 0)\n", worst);

    return pass ? 0 : 1;
}
