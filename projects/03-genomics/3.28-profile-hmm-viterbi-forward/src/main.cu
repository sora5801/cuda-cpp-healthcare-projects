// ===========================================================================
// src/main.cu  --  Entry point: load DB, build profile, score, verify, report
// ---------------------------------------------------------------------------
// Project 3.28 : Profile HMM (Viterbi / Forward)
//
// THE 5-STEP SHAPE every project in this repo follows
//   1. Load the problem: a tiny FASTA-like file. The FIRST record is the family
//      CONSENSUS (used to build the profile HMM); the rest are the search
//      DATABASE -- one planted homolog plus several decoys (data/README.md).
//   2. CPU reference (reference_cpu.cpp): Viterbi + Forward scores -> trusted.
//   3. GPU search (kernels.cu): the same two scores, one thread per sequence.
//   4. VERIFY: GPU agrees with CPU within tolerance (both algorithms).
//   5. REPORT: deterministic ranked hits to stdout; timing to stderr.
//
//   STDOUT is byte-for-byte deterministic (demo/run_demo diffs it against
//   demo/expected_output.txt); run-to-run timings go to STDERR.
//
//   The science check (PATTERNS.md §4): the planted homolog -- a lightly mutated
//   copy of the consensus -- should rank #1 by both scores, well above the random
//   decoys. That validates the model, not just CPU==GPU agreement.
//
// Code tour: start here, then phmm.h (the shared recurrence), kernels.cuh ->
//   kernels.cu, then reference_cpu.*.  See ../THEORY.md for the "why".
// ===========================================================================
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "kernels.cuh"        // phmm_search_gpu
#include "reference_cpu.h"    // load_database, build_profile_from_consensus, *_cpu
#include "util/io.hpp"        // util::CpuTimer, util::max_abs_err

static const char* PROJECT_ID   = "3.28";
static const char* PROJECT_NAME = "Profile HMM (Viterbi / Forward)";

// Verification tolerance. Both scorers run the IDENTICAL log-space operations on
// CPU and GPU via the shared phmm.h core, on short sequences (L <= ~30 here), in
// DOUBLE precision. So agreement is at the float round-trip level: we store the
// double score into a float on both sides, and 1e-4 (nats) is a generous margin
// over the ~1e-6 relative float epsilon at these magnitudes. (THEORY §6.)
static constexpr double TOLERANCE = 1.0e-4;

// Rank the database by a score (descending), ties broken by lower index so the
// ordering -- and thus stdout -- is fully deterministic.
static std::vector<int> rank_desc(const std::vector<float>& score) {
    std::vector<int> idx(score.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        if (score[a] != score[b]) return score[a] > score[b];  // higher first
        return a < b;                                          // tie -> lower idx
    });
    return idx;
}

int main(int argc, char** argv) {
    // ---- 1. Load ------------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/phmm_sample.fasta";
    SeqDB all;
    try {
        all = load_database(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    if (all.n < 2) {
        std::fprintf(stderr, "[error] need >= 2 records (consensus + >=1 database seq)\n");
        return 2;
    }

    // Split: record 0 is the consensus -> the profile; records 1.. are the DB.
    // We rebuild the consensus string from its residue codes so the model builder
    // can read it; this keeps the loader and builder using one alphabet.
    std::string consensus;
    {
        static const char* AA = "ACDEFGHIKLMNPQRSTVWY";   // same order as aa_code
        const int len0 = all.len[0];
        for (int j = 0; j < len0; ++j) consensus.push_back(AA[all.res[all.off[0] + j]]);
    }
    ProfileHMM profile;
    try {
        profile = build_profile_from_consensus(consensus);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // Build the search database = every record EXCEPT the consensus. We re-pack a
    // fresh SeqDB so offsets are contiguous from 0 (what the kernel expects).
    SeqDB db;
    for (int s = 1; s < all.n; ++s) {
        db.off.push_back(static_cast<int>(db.res.size()));
        db.len.push_back(all.len[s]);
        db.name.push_back(all.name[s]);
        for (int j = 0; j < all.len[s]; ++j) db.res.push_back(all.res[all.off[s] + j]);
        ++db.n;
    }

    // ---- 2. CPU reference (timed) ------------------------------------------
    std::vector<float> vit_cpu, fwd_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    viterbi_cpu(profile, db, vit_cpu);
    forward_cpu(profile, db, fwd_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU search (each kernel timed inside the wrapper) --------------
    std::vector<float> vit_gpu, fwd_gpu;
    float vit_ms = 0.0f, fwd_ms = 0.0f;
    phmm_search_gpu(profile, db, /*is_viterbi=*/true,  vit_gpu, &vit_ms);
    phmm_search_gpu(profile, db, /*is_viterbi=*/false, fwd_gpu, &fwd_ms);

    // ---- 4. Verify (both algorithms) ---------------------------------------
    const double err_vit = util::max_abs_err(vit_cpu, vit_gpu);
    const double err_fwd = util::max_abs_err(fwd_cpu, fwd_gpu);
    const bool pass = (err_vit <= TOLERANCE) && (err_fwd <= TOLERANCE);

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    // Rank by Viterbi (the best-path score, HMMER's classic ranking signal).
    const std::vector<int> order = rank_desc(vit_gpu);
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("profile: %d match columns (consensus '%s')\n",
                profile.M, consensus.c_str());
    std::printf("database: %d sequences scored (Viterbi + Forward, log-prob in nats)\n", db.n);
    std::printf("rank by Viterbi score (best path):\n");
    std::printf("  %-4s %-10s %12s %12s\n", "rank", "name", "viterbi", "forward");
    for (std::size_t r = 0; r < order.size(); ++r) {
        const int s = order[r];
        std::printf("  %-4zu %-10s %12.4f %12.4f\n",
                    r + 1, db.name[s].c_str(), vit_gpu[s], fwd_gpu[s]);
    }
    // The top hit and its margin over the runner-up: a one-line "did it work?".
    {
        const int best = order[0];
        const float runner = (order.size() > 1) ? vit_gpu[order[1]] : vit_gpu[best];
        std::printf("top hit: %s  (Viterbi %.4f, %.4f nats above runner-up)\n",
                    db.name[best].c_str(), vit_gpu[best], vit_gpu[best] - runner);
    }
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-04)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (consensus + %d database sequences)\n",
                 path.c_str(), db.n);
    std::fprintf(stderr, "[timing] CPU (both scorers): %.3f ms   "
                         "GPU Viterbi: %.3f ms   GPU Forward: %.3f ms\n",
                 cpu_ms, vit_ms, fwd_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny database is dominated "
                         "by launch/copy overhead; the GPU wins at metagenomic scale.\n");
    std::fprintf(stderr, "[verify] max_abs_err: Viterbi %.3e  Forward %.3e  (tolerance %.1e)\n",
                 err_vit, err_fwd, TOLERANCE);

    return pass ? 0 : 1;
}
