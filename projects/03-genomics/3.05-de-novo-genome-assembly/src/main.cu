// ===========================================================================
// src/main.cu  --  Entry point: load reads, sketch, overlap (CPU+GPU), verify
// ---------------------------------------------------------------------------
// Project 3.5 : De Novo Genome Assembly  (all-vs-all read-overlap stage)
//
// WHAT THIS FILE DOES  (the 5-step shape every project in this repo follows)
//   1. Load the reads (FASTA-like sample, or a built-in synthetic fallback).
//   2. SKETCH each read into its minimizer set (sketch_reads, host).
//   3. Score all read pairs on the CPU reference AND on the GPU.
//   4. VERIFY the two per-pair score arrays agree EXACTLY (integers -> tol 0).
//   5. REPORT the overlap graph (edges + connected components) to stdout;
//      timings to stderr.
//
//   The overlap graph IS the scaffold of de-novo assembly: each connected
//   component is a set of reads that tile one region of the genome -> one
//   contig. We report the component structure so the result is interpretable as
//   "assembly" (THEORY "Where this sits in the real world").
//
//   STDOUT is byte-for-byte deterministic (diffed against expected_output.txt);
//   run-varying timings go to STDERR (shown, not diffed) -- PATTERNS.md sec.3.
//
// READ THIS FIRST in the code tour, then assembly.h -> reference_cpu.* (sketch +
// CPU reference) -> kernels.cuh -> kernels.cu (the GPU twin).
// ===========================================================================
#include <cstdio>
#include <numeric>     // std::iota
#include <string>
#include <vector>

#include "kernels.cuh"        // overlap_gpu (GPU path), assembly.h (shared math)
#include "reference_cpu.h"    // load_fasta, sketch_reads, overlap_cpu, MIN_SHARED
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "3.5";
static const char* PROJECT_NAME = "De Novo Genome Assembly";

// Verification tolerance. Both sides count shared minimizers with the SAME
// integer routine (count_shared_sorted), so the results are exactly equal --
// no floating point is involved anywhere on the scored path. We therefore demand
// ZERO mismatches (PATTERNS.md sec.4 "Exact"). The check is integer ==.
static constexpr int TOLERANCE = 0;

// ---------------------------------------------------------------------------
// make_synthetic_reads: a tiny, fully-deterministic read set with a KNOWN
// answer, used when no sample file is given (and mirrored by make_synthetic.py
// and the committed sample). Construction (see data/README.md):
//   * A 120-base "genome" of fixed sequence.
//   * 6 reads, each a 60-base substring sliding by 12 bases (reads 0..5 cover
//     [0,60),[12,72),...,[60,120)). Consecutive reads overlap by 48 bases, so
//     they SHARE minimizers; reads far apart (0 vs 5) do not. The expected graph
//     is therefore a single chain 0-1-2-3-4-5 (one contig) -- a result the demo
//     recovers and the learner can reason about by hand.
//   This is SYNTHETIC data: the sequence is arbitrary, not from any organism.
// ---------------------------------------------------------------------------
static std::vector<std::string> make_synthetic_reads() {
    // A fixed 120-base pseudo-genome (labelled synthetic; no biological meaning).
    static const char* GENOME =
        "ACGTTGCAAGCTAGGCATCGATCGGATCCAACGTAGCTAGCATGCATGCTAGCTAGGCAT"
        "CGATCGATTACGGCATCCAGTACGTAGCATCGATCGTAGCTAGCATCGGATCCAACGTAG";
    const std::string g = GENOME;            // 120 bases
    const int read_len = 60, step = 12;
    std::vector<std::string> reads;
    for (int s = 0; s + read_len <= static_cast<int>(g.size()); s += step)
        reads.push_back(g.substr(s, read_len));
    return reads;
}

// ---------------------------------------------------------------------------
// connected_components: trivial union-find over the overlap edges, to count how
// many separate read clusters (≈ contigs) the graph forms and the largest one.
//   This is a stand-in for the "layout" step of overlap-layout-consensus: once
//   we know which reads overlap, reads in the same component tile one locus. We
//   keep it on the host (cheap, serial, deterministic) -- the GPU's job was the
//   O(n^2) scoring above.
//   Returns the number of components; fills comp_size_max with the largest.
// ---------------------------------------------------------------------------
static int connected_components(int n, const std::vector<Overlap>& edges, int& comp_size_max) {
    std::vector<int> parent(n);
    std::iota(parent.begin(), parent.end(), 0);          // each read its own set
    // find with path halving (iterative -> deterministic, no recursion).
    auto find = [&](int x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    for (const Overlap& e : edges) {                     // union the endpoints
        int ra = find(e.i), rb = find(e.j);
        if (ra != rb) parent[ra] = rb;
    }
    std::vector<int> size(n, 0);
    int comps = 0;
    comp_size_max = 0;
    for (int v = 0; v < n; ++v) size[find(v)]++;
    for (int v = 0; v < n; ++v) {
        if (size[v] > 0) { ++comps; if (size[v] > comp_size_max) comp_size_max = size[v]; }
    }
    return comps;
}

int main(int argc, char** argv) {
    // ---- 1. Load the reads --------------------------------------------------
    std::vector<std::string> reads;
    std::string source;
    if (argc > 1) {
        try {
            reads = load_fasta(argv[1]);
            source = argv[1];
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[error] %s\n", e.what());
            return 2;
        }
    } else {
        reads = make_synthetic_reads();
        source = "synthetic (built-in)";
    }

    // ---- 2. Sketch every read into its minimizer set (host) ----------------
    ReadSet rs = sketch_reads(reads);

    // ---- 3a. CPU reference (timed) -----------------------------------------
    std::vector<Overlap> ov_cpu;
    std::vector<int>     score_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    overlap_cpu(rs, ov_cpu, &score_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3b. GPU result (kernel timed inside the wrapper) ------------------
    std::vector<int> score_gpu;
    float gpu_kernel_ms = 0.0f;
    overlap_gpu(rs, score_gpu, &gpu_kernel_ms);

    // ---- 4. Verify: every pair's shared count must match EXACTLY -----------
    //   Integer comparison -> tolerance 0. A single mismatch fails the run.
    int max_diff = 0, n_mismatch = 0;
    if (score_cpu.size() != score_gpu.size()) {
        max_diff = 1 << 30; n_mismatch = 1;   // shape bug -> guaranteed fail
    } else {
        for (std::size_t p = 0; p < score_cpu.size(); ++p) {
            int d = score_cpu[p] - score_gpu[p];
            if (d < 0) d = -d;
            if (d > 0) { ++n_mismatch; if (d > max_diff) max_diff = d; }
        }
    }
    const bool pass = (max_diff <= TOLERANCE);

    // The thresholded edge list and component structure (from the CPU reference,
    // which equals the GPU when pass==true) -- the deterministic "assembly".
    int comp_max = 0;
    const int comps = connected_components(rs.n, ov_cpu, comp_max);

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    long long total_min = static_cast<long long>(rs.mins.size());
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("all-vs-all read overlap via minimizers (k=%d, w=%d)\n", K, W);
    std::printf("reads = %d   pairs = %lld   total minimizers = %lld\n",
                rs.n, num_pairs(rs.n), total_min);
    std::printf("overlap edges (shared minimizers >= %d):\n", MIN_SHARED);
    for (const Overlap& e : ov_cpu)
        std::printf("  read %d -- read %d   shared = %d\n", e.i, e.j, e.shared);
    std::printf("graph: %d edge(s), %d component(s), largest component = %d read(s)\n",
                static_cast<int>(ov_cpu.size()), comps, comp_max);
    std::printf("RESULT: %s (GPU per-pair scores match CPU exactly, tol=%d)\n",
                pass ? "PASS" : "FAIL", TOLERANCE);

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (n=%d reads)\n", source.c_str(), rs.n);
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny pair count is "
                         "dominated by launch/copy overhead; the GPU's O(n^2) edge "
                         "grows with read count.\n");
    std::fprintf(stderr, "[verify] mismatching pairs = %d / %lld   max |diff| = %d  (tol %d)\n",
                 n_mismatch, num_pairs(rs.n), max_diff, TOLERANCE);

    return pass ? 0 : 1;
}
