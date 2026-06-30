// ===========================================================================
// src/main.cu  --  Entry point: load alignment + trees, score, verify, report
// ---------------------------------------------------------------------------
// Project 3.9 : Phylogenetic Likelihood / Tree Inference
//
// THE 5-STEP SHAPE every project in this repo follows:
//   1. Load the problem (an alignment, kappa, and candidate trees from
//      data/sample, or a built-in synthetic fallback so the demo always runs).
//   2. CPU reference  (reference_cpu.cpp) -> trusted per-tree log-likelihoods.
//   3. GPU scoring    (kernels.cu)        -> the thing being taught.
//   4. VERIFY: GPU per-tree totals equal the CPU's (EXACTLY -- both reduce the
//      same fixed-point integers, so we demand bit-identical agreement).
//   5. REPORT: deterministic per-tree lnL + the maximum-likelihood winner to
//      stdout; timings to stderr.
//
// STDOUT is byte-for-byte deterministic (diffed by demo/run_demo against
// demo/expected_output.txt); run-to-run timings go to STDERR.
//
// Code tour: start here, then kernels.cuh -> kernels.cu, felsenstein.h (the
// shared math), reference_cpu.* (the baseline). The "why" is in ../THEORY.md.
// ===========================================================================
#include <cmath>     // std::fabs (verification)
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // score_trees_gpu
#include "reference_cpu.h"    // load_problem, score_trees_cpu, best_tree_index
#include "felsenstein.h"      // PhyloNode, PHYLO_GAP (for the synthetic fallback)
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "3.9";
static const char* PROJECT_NAME = "Phylogenetic Likelihood / Tree Inference";

// ---------------------------------------------------------------------------
// VERIFICATION TOLERANCE.
//   The CPU and GPU both sum per-site log-likelihoods as FIXED-POINT INTEGERS
//   (felsenstein.h to_fixed) and only convert back to a double at the end. Integer
//   addition is associative/commutative, so the two totals are IDENTICAL integers
//   regardless of thread order -> we verify EXACT equality (atol 0 on the fixed
//   integer, here expressed as a tiny double tolerance to absorb the single
//   /SCALE division). See ../THEORY.md "How we verify correctness".
// ---------------------------------------------------------------------------
static constexpr double TOLERANCE = 0.5 / PHYLO_FIXED_SCALE;   // < half a fixed-point ULP

// ---------------------------------------------------------------------------
// make_synthetic_problem: a tiny, fully-deterministic 4-taxon DNA dataset with a
// KNOWN true tree, used when no data file is supplied so the demo always runs.
//
//   Four taxa A,B,C,D. We make A,B very similar and C,D very similar (two pairs),
//   so the TRUE tree groups ((A,B),(C,D)). We then offer that tree plus the two
//   alternative resolutions ((A,C),(B,D)) and ((A,D),(B,C)) -- exactly the three
//   topologies an NNI move explores around the central branch. Maximum likelihood
//   should pick the true one. (data/sample/ contains the same setup at larger n;
//   this fallback keeps the binary self-contained even with no data file.)
//
//   Sequences are hand-built so AB and CD agree within their pair and differ
//   between pairs -> a clear phylogenetic signal. Labeled SYNTHETIC everywhere.
// ---------------------------------------------------------------------------
static PhyloProblem make_synthetic_problem() {
    PhyloProblem prob;
    prob.kappa = 2.0;
    prob.align.n_taxa  = 4;
    prob.align.names = {"A", "B", "C", "D"};

    // A small alignment; rows are taxa, columns are sites. Pair (A,B) shares one
    // pattern, pair (C,D) shares another, so the data supports ((A,B),(C,D)).
    const char* rows[4] = {
        // A
        "AAAACCCCGGGGTTTTAAAACCCCGGGGTTTT",
        // B  (A with a few private mutations -> still closest to A)
        "AAAACCCCGGGGTTTTAAGACCTCGGAGTTTT",
        // C
        "GGGGTTTTAAAACCCCGGGGTTTTAAAACCCC",
        // D  (C with a few private mutations -> still closest to C)
        "GGGGTTTTAAAACCCCGGAGTTCTAAGACCCC",
    };
    const int n_sites = static_cast<int>(std::string(rows[0]).size());
    prob.align.n_sites = n_sites;
    prob.align.data.assign(static_cast<std::size_t>(n_sites) * 4, PHYLO_GAP);
    auto enc = [](char c) -> unsigned char {
        switch (c) { case 'A': return 0; case 'C': return 1;
                     case 'G': return 2; case 'T': return 3; default: return PHYLO_GAP; }
    };
    for (int r = 0; r < 4; ++r)
        for (int j = 0; j < n_sites; ++j)
            prob.align.data[static_cast<std::size_t>(j) * 4 + r] = enc(rows[r][j]);

    // Three candidate rooted binary trees on leaves A=0,B=1,C=2,D=3. Each has
    // n_internal = n_taxa-1 = 3 nodes in post-order (children before parents):
    //   node 4 = cherry of the first pair, node 5 = cherry of the second pair,
    //   node 6 = root joining nodes 4 and 5. Branch lengths are small within a
    //   cherry (close relatives) and the connecting branch is longer.
    const double tip = 0.05;   // short tip branch (within a close pair)
    const double mid = 0.40;   // longer branch separating the two pairs
    auto make_tree = [&](const char* label, int p, int q, int r, int s) {
        CandidateTree tree; tree.label = label;
        tree.nodes.push_back({p, q, tip, tip});   // node 4: cherry (p,q)
        tree.nodes.push_back({r, s, tip, tip});   // node 5: cherry (r,s)
        tree.nodes.push_back({4, 5, mid, mid});   // node 6: root joins the cherries
        return tree;
    };
    prob.trees.push_back(make_tree("((A,B),(C,D))_true", 0, 1, 2, 3));
    prob.trees.push_back(make_tree("((A,C),(B,D))_NNI1", 0, 2, 1, 3));
    prob.trees.push_back(make_tree("((A,D),(B,C))_NNI2", 0, 3, 1, 2));
    return prob;
}

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    PhyloProblem prob;
    const char* source = nullptr;
    if (argc > 1) {
        try {
            prob = load_problem(argv[1]);
            source = argv[1];
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[error] %s\n", e.what());
            return 2;
        }
    } else {
        prob = make_synthetic_problem();
        source = "synthetic (built-in 4-taxon)";
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<double> lnL_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    score_trees_cpu(prob, lnL_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU scoring (kernels timed inside the wrapper) ----------------
    std::vector<double> lnL_gpu;
    float gpu_kernel_ms = 0.0f;
    score_trees_gpu(prob, lnL_gpu, &gpu_kernel_ms);

    // ---- 4. Verify (EXACT integer agreement) ------------------------------
    double worst = 0.0;
    for (std::size_t t = 0; t < lnL_cpu.size(); ++t) {
        const double d = std::fabs(lnL_cpu[t] - lnL_gpu[t]);
        if (d > worst) worst = d;
    }
    const bool pass = (lnL_cpu.size() == lnL_gpu.size()) && (worst <= TOLERANCE);

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    const int best = best_tree_index(lnL_gpu);
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("alignment: %d taxa x %d sites   model: K2P (kappa = %.2f)\n",
                prob.align.n_taxa, prob.align.n_sites, prob.kappa);
    std::printf("candidate trees scored: %d\n", static_cast<int>(prob.trees.size()));
    // Per-tree total log-likelihood (higher = better supported by the data).
    for (std::size_t t = 0; t < prob.trees.size(); ++t)
        std::printf("  tree[%zu] %-22s lnL = %.6f\n",
                    t, prob.trees[t].label.c_str(), lnL_gpu[t]);
    std::printf("MAXIMUM-LIKELIHOOD TREE: tree[%d] %s  (lnL = %.6f)\n",
                best, prob.trees[best].label.c_str(), lnL_gpu[best]);
    std::printf("RESULT: %s (GPU per-tree lnL match CPU exactly)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d taxa, %d sites, %d trees)\n",
                 source, prob.align.n_taxa, prob.align.n_sites,
                 static_cast<int>(prob.trees.size()));
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernels: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny sample is dominated by "
                         "launch/copy overhead; the GPU wins at genome scale (millions of sites).\n");
    std::fprintf(stderr, "[verify] max |lnL_cpu - lnL_gpu| = %.3e  (tolerance %.1e; exact in fixed-point)\n",
                 worst, TOLERANCE);

    return pass ? 0 : 1;
}
