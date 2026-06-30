// ===========================================================================
// src/main.cu  --  Entry point: load reads, count + sketch on CPU & GPU, verify
// ---------------------------------------------------------------------------
// Project 3.6 : k-mer Counting & Minimiser Sketching
//
// 6-step shape:
//   1. Load two labelled read sets A and B (data/sample).
//   2. CPU reference: count canonical k-mers (set A) + build minimiser sketches
//      (A and B) + estimate their Jaccard similarity.
//   3. GPU: the same, via the device hash table and minimiser kernel.
//   4. VERIFY: histograms match key-by-key and count-by-count; sketches match
//      hash-by-hash; Jaccard estimates match. All EXACT (shared kmer.h math,
//      integer counts) -- tolerance is 0.
//   5. REPORT (stdout, deterministic): distinct-k-mer count, the top k-mers by
//      count (ties broken by key), sketch sizes, and the Jaccard estimate.
//   6. TIMING (stderr): CPU vs GPU kernel ms -- a teaching artifact, not a benchmark.
//
// Code tour: start here, then kmer.h (the shared math), reference_cpu.cpp (the
// readable baseline), kernels.cu (the GPU twins).
// ===========================================================================
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

#include "kernels.cuh"        // count_kmers_gpu, sketch_gpu
#include "reference_cpu.h"    // load_reads, count_kmers_cpu, sketch_cpu, jaccard_estimate
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "3.6";
static const char* PROJECT_NAME = "k-mer Counting & Minimiser Sketching";

// How many top-frequency k-mers to print (deterministic: by count desc, key asc).
static constexpr int TOP_N = 8;

// ---------------------------------------------------------------------------
// histograms_equal: exact entry-by-entry comparison of two sorted histograms.
//   Both come back sorted ascending by key, so a parallel walk suffices. Returns
//   true iff identical (same keys, same counts). This is our headline check:
//   the GPU hash table must reproduce the CPU std::map exactly.
// ---------------------------------------------------------------------------
static bool histograms_equal(const std::vector<KmerCount>& a,
                             const std::vector<KmerCount>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i].key != b[i].key || a[i].count != b[i].count) return false;
    return true;
}

// sketches_equal: exact comparison of two bottom-s sketches (sorted, distinct).
static bool sketches_equal(const Sketch& a, const Sketch& b) {
    return a.hashes == b.hashes;
}

int main(int argc, char** argv) {
    // ---- 1. Load ----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/kmer_sample.txt";
    ReadSet A, B;
    int s = 0;
    try {
        A = load_reads(path, B, s);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) ----------------------------------------
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    std::vector<KmerCount> hist_cpu = count_kmers_cpu(A);
    Sketch sa_cpu = sketch_cpu(A, s);
    Sketch sb_cpu = sketch_cpu(B, s);
    const double jac_cpu = jaccard_estimate(sa_cpu, sb_cpu, s);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU (kernel times captured separately) -----------------------
    float count_ms = 0.0f, sketchA_ms = 0.0f, sketchB_ms = 0.0f;
    std::vector<KmerCount> hist_gpu = count_kmers_gpu(A, &count_ms);
    Sketch sa_gpu = sketch_gpu(A, s, &sketchA_ms);
    Sketch sb_gpu = sketch_gpu(B, s, &sketchB_ms);
    const double jac_gpu = jaccard_estimate(sa_gpu, sb_gpu, s);

    // ---- 4. Verify (all exact) -------------------------------------------
    const bool hist_ok    = histograms_equal(hist_cpu, hist_gpu);
    const bool sketchA_ok = sketches_equal(sa_cpu, sa_gpu);
    const bool sketchB_ok = sketches_equal(sb_cpu, sb_gpu);
    const bool jac_ok     = (jac_cpu == jac_gpu);   // exact: same integers / counts
    const bool pass = hist_ok && sketchA_ok && sketchB_ok && jac_ok;

    // ---- 5. Deterministic report -> STDOUT -------------------------------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("params: k=%d  w=%d  s=%d\n", A.k, A.w, s);
    std::printf("set A: %d reads, %zu bases\n", A.num_reads, A.bases.size());
    std::printf("set B: %d reads, %zu bases\n", B.num_reads, B.bases.size());
    std::printf("distinct canonical k-mers in A: %zu\n", hist_gpu.size());

    // Top-N k-mers by count (desc), ties broken by ascending key -> deterministic.
    std::vector<KmerCount> top = hist_gpu;
    std::sort(top.begin(), top.end(), [](const KmerCount& a, const KmerCount& b) {
        if (a.count != b.count) return a.count > b.count;   // higher count first
        return a.key < b.key;                               // tie: smaller key first
    });
    const int show = (int)std::min<std::size_t>(TOP_N, top.size());
    std::printf("top %d k-mers by count:\n", show);
    for (int i = 0; i < show; ++i)
        std::printf("  %s  count=%u\n", kmer_to_string(top[i].key, A.k).c_str(), top[i].count);

    std::printf("sketch sizes: |A|=%zu  |B|=%zu  (bottom-%d MinHash)\n",
                sa_gpu.hashes.size(), sb_gpu.hashes.size(), s);
    std::printf("Jaccard(A,B) estimate = %.4f\n", jac_gpu);
    std::printf("RESULT: %s (GPU hist+sketch+Jaccard match CPU exactly)\n",
                pass ? "PASS" : "FAIL");

    // ---- 6. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s\n", path.c_str());
    std::fprintf(stderr, "[timing] CPU(all): %.3f ms   GPU kernels: count %.3f + sketchA %.3f + sketchB %.3f ms\n",
                 cpu_ms, count_ms, sketchA_ms, sketchB_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- tiny input is launch-bound; the GPU's edge grows "
                         "with read count (real runs are 10^8-10^9 k-mers).\n");
    std::fprintf(stderr, "[verify] hist=%s  sketchA=%s  sketchB=%s  jaccard(cpu/gpu)=%.4f/%.4f\n",
                 hist_ok ? "ok" : "MISMATCH", sketchA_ok ? "ok" : "MISMATCH",
                 sketchB_ok ? "ok" : "MISMATCH", jac_cpu, jac_gpu);

    return pass ? 0 : 1;
}
