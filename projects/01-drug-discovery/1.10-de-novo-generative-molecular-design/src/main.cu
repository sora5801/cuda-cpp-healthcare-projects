// ===========================================================================
// src/main.cu  --  Entry point: load corpus, train, generate (CPU+GPU), verify
// ---------------------------------------------------------------------------
// Project 1.10 : De Novo Generative Molecular Design (reduced-scope teaching).
//
// The 5-step shape every project in this repo follows:
//   1. Load the problem  (a tiny SMILES training corpus from data/sample).
//   2. Train the model   (count char transitions -> a first-order Markov model).
//   3a. CPU reference     (reference_cpu.cpp)  -> trusted molecules + scores.
//   3b. GPU generation    (kernels.cu)         -> the thing being taught.
//   4. VERIFY: GPU scores+lengths EXACTLY equal the CPU ones (bit-identical RNG).
//   5. REPORT: deterministic distribution stats + the best molecule to stdout;
//      timing + run-varying detail to stderr.
//
// STDOUT is byte-for-byte deterministic (diffed by demo/run_demo against
// demo/expected_output.txt); run-to-run timings go to STDERR (PATTERNS.md §3).
//
// Code tour: start here, then generator.h (the model+RNG+scorer), then
// reference_cpu.* (the baseline), then kernels.cuh -> kernels.cu (the GPU twin).
// ===========================================================================
#include <climits>     // INT_MIN
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // generate_and_score_gpu, MarkovModel, Corpus
#include "reference_cpu.h"    // load_corpus, train_model, generate_and_score_cpu
#include "generator.h"        // generate_molecule, rng_seed (to rebuild best SMILES)
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "1.10";
static const char* PROJECT_NAME = "De Novo Generative Molecular Design";

// Threshold above which we call a molecule a "valid drug-like hit". This is a
// fixed, documented cutoff on the toy integer reward (generator.h::score_molecule
// returns milli-reward); it is for teaching interpretability only, NOT a real
// drug-likeness criterion. A molecule with no malformed branches and a sensible
// size already scores >= 200 here.
static constexpr int HIT_THRESHOLD = 500;

// ---------------------------------------------------------------------------
// Reconstruct the SMILES string of molecule `index` from the model + base seed.
//   Generation is fully deterministic given (model, seed, index), so we can
//   recreate any molecule's string on the host without the GPU returning it.
//   We use this to print the single best molecule (its index comes from the
//   GPU's score array) -- keeping the GPU->host transfer tiny (just two int
//   arrays) instead of shipping back every string.
// ---------------------------------------------------------------------------
static std::string rebuild_smiles(const MarkovModel& model, uint64_t seed, int index) {
    char buf[MAX_LEN + 1];
    Rng rng = rng_seed(seed, static_cast<uint64_t>(index));
    int len = generate_molecule(model, rng, buf);
    return std::string(buf, buf + len);
}

// ---------------------------------------------------------------------------
// Find the best molecule (highest score; ties -> lowest index) in an array.
//   This is the deterministic "goal-directed selection" reduction: in a real RL
//   loop you would weight gradient updates by reward; here we simply report the
//   argmax so the demo shows a concrete optimised molecule. Done on the host so
//   the tie-break rule is identical to the CPU reference (PATTERNS.md §3).
// ---------------------------------------------------------------------------
static int argmax_score(const std::vector<int>& scores) {
    int best_idx = 0, best = INT_MIN;
    for (int i = 0; i < static_cast<int>(scores.size()); ++i)
        if (scores[i] > best) { best = scores[i]; best_idx = i; }
    return best_idx;
}

int main(int argc, char** argv) {
    // ---- 1. Load corpus ----------------------------------------------------
    const std::string path = (argc > 1) ? argv[1]
                                        : "data/sample/smiles_corpus_sample.txt";
    Corpus corpus;
    try {
        corpus = load_corpus(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. Train the first-order Markov model (once, on the host) ---------
    const MarkovModel model = train_model(corpus);
    const int      n_gen = corpus.n_gen;
    const uint64_t seed  = corpus.seed;

    // ---- 3a. CPU reference (timed) ----------------------------------------
    std::vector<int> score_cpu, len_cpu;
    std::string best_smiles_cpu;
    int best_index_cpu = 0;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    generate_and_score_cpu(model, n_gen, seed, score_cpu, len_cpu,
                           best_smiles_cpu, best_index_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3b. GPU generation (kernel timed inside the wrapper) -------------
    std::vector<int> score_gpu, len_gpu;
    float gpu_kernel_ms = 0.0f;
    generate_and_score_gpu(model, n_gen, seed, score_gpu, len_gpu, &gpu_kernel_ms);

    // ---- 4. Verify: GPU == CPU exactly ------------------------------------
    // The RNG, sampling loop, and scorer are bit-identical on host and device
    // (shared generator.h, integer arithmetic), so the score and length arrays
    // must match EXACTLY -- tolerance is literally zero (PATTERNS.md §4 "exact").
    int mismatches = 0;
    for (int i = 0; i < n_gen; ++i)
        if (score_cpu[i] != score_gpu[i] || len_cpu[i] != len_gpu[i]) ++mismatches;
    const bool pass = (mismatches == 0);

    // ---- 5. Deterministic report -> STDOUT --------------------------------
    // Distribution-learning summary: how many generated molecules clear the
    // drug-like threshold, and the mean reward (in integer milli-units so the
    // sum is order-independent and reproducible).
    long long sum_reward = 0;     // 64-bit so n_gen up to millions cannot overflow
    int n_hits = 0;
    for (int i = 0; i < n_gen; ++i) {
        sum_reward += score_cpu[i];
        if (score_cpu[i] >= HIT_THRESHOLD) ++n_hits;
    }
    const long long mean_milli = sum_reward / n_gen;   // integer mean milli-reward

    // Goal-directed pick: the single best molecule (argmax of GPU scores; the
    // tie-break matches the CPU reference, so best_index agrees with the CPU's).
    const int best_index = argmax_score(score_gpu);
    const std::string best_smiles = rebuild_smiles(model, seed, best_index);

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("trained first-order Markov model on %d SMILES; generated %d molecules\n",
                static_cast<int>(corpus.train.size()), n_gen);
    std::printf("drug-like hits (score >= %d): %d / %d\n", HIT_THRESHOLD, n_hits, n_gen);
    std::printf("mean reward: %lld milli-units\n", mean_milli);
    std::printf("best molecule: idx=%d  SMILES=%s  score=%d milli-units\n",
                best_index, best_smiles.c_str(), score_gpu[best_index]);
    std::printf("RESULT: %s (GPU matches CPU exactly: %d/%d molecules identical)\n",
                pass ? "PASS" : "FAIL", n_gen - mismatches, n_gen);

    // ---- 5b. Run-varying detail -> STDERR ---------------------------------
    std::fprintf(stderr, "[data]   source: %s  (train=%d, generate=%d, seed=%llu)\n",
                 path.c_str(), static_cast<int>(corpus.train.size()), n_gen,
                 static_cast<unsigned long long>(seed));
    std::fprintf(stderr, "[model]  alphabet=%d symbols, transition table=%dx%d "
                         "(constant memory)\n", NSYM, NSYM, NSYM);
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- tiny n is launch/copy "
                         "bound; the GPU's edge grows with the molecule count.\n");
    std::fprintf(stderr, "[verify] mismatches = %d  (exact integer comparison; "
                         "tolerance = 0)\n", mismatches);
    // Cross-check the host argmax against the CPU reference's own best pick.
    std::fprintf(stderr, "[verify] best index CPU=%d GPU=%d  (agree: %s)\n",
                 best_index_cpu, best_index,
                 (best_index_cpu == best_index) ? "yes" : "NO");

    return pass ? 0 : 1;
}
