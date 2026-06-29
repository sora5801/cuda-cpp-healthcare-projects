// ===========================================================================
// src/main.cu  --  Entry point: build IFPs, cluster binding modes, verify, report
// ---------------------------------------------------------------------------
// Project 1.33 : Interaction Fingerprinting & Binding-Mode Clustering
//
// THE 5-STEP SHAPE (every project in this repo follows it)
//   1. LOAD   the pocket residues + ligand poses (data/sample, or argv[1]).
//   2. STAGE A: build the interaction fingerprints on CPU and GPU, and VERIFY
//      they are bit-identical (geometry -> bits is exact integer logic).
//   3. STAGE B: cluster the IFPs into binding modes on CPU and GPU, and VERIFY
//      the labels + consensus centroids match exactly (integer majority vote).
//   4. REPORT  a DETERMINISTIC summary to STDOUT (diffed by the demo): cluster
//      sizes, each cluster's consensus interactions, cost, and how well the
//      clustering recovered the planted synthetic modes (purity).
//   5. TIMING + run-varying detail -> STDERR (shown by the demo, never diffed).
//
// Code tour: start here, then ifp.h (the shared math), reference_cpu.cpp (the
// baseline), kernels.cu (the GPU twin). See ../THEORY.md for the "why".
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // build_ifps_gpu, ifp_cluster_gpu, Dataset
#include "reference_cpu.h"    // load_dataset, build_ifps, ifp_cluster_cpu, helpers
#include "ifp.h"              // NUM_RESIDUES, NUM_ITYPES, IFP_BITS, FP_WORDS
#include "util/io.hpp"        // util::CpuTimer

// Identify the program. These MUST stay in sync with demo/expected_output.txt.
static const char* PROJECT_ID   = "1.33";
static const char* PROJECT_NAME = "Interaction Fingerprinting & Binding-Mode Clustering";

// Fixed Lloyd iterations -> deterministic (no convergence test whose stopping
// point could differ between CPU and GPU). 12 is plenty for the sample to settle.
static constexpr int ITERS = 12;

// Short labels for the four interaction types, in bit order (see ifp.h).
static const char* ITYPE_NAME[NUM_ITYPES] = { "hydrophobic", "hbond", "aromatic", "ionic" };

// ---------------------------------------------------------------------------
// cluster_purity_centi : a DETERMINISTIC integer recovery metric.
//   For each cluster, count its members' most-common ground-truth mode; sum
//   those majorities over clusters and divide by P. Returned as HUNDREDTHS of a
//   percent (purity 100.00% -> 10000), all integer math, so stdout has no
//   floating-point formatting ambiguity. (Ground truth is the synthetic planted
//   mode; this measures "did clustering rediscover the modes".) Returns -1 when
//   there is no ground truth (real data with no labels).
// ---------------------------------------------------------------------------
static int cluster_purity_centi(const Dataset& d, const std::vector<int>& labels) {
    if (d.true_mode.empty()) return -1;
    // modes present = max true label + 1 (synthetic data is 0-based + contiguous).
    int modes = 0;
    for (int p = 0; p < d.P; ++p) if (d.true_mode[p] + 1 > modes) modes = d.true_mode[p] + 1;

    long correct = 0;
    for (int k = 0; k < d.K; ++k) {
        std::vector<int> hist(modes, 0);
        for (int p = 0; p < d.P; ++p)
            if (labels[p] == k) hist[d.true_mode[p]]++;
        int best = 0;
        for (int m = 0; m < modes; ++m) if (hist[m] > best) best = hist[m];
        correct += best;                       // members agreeing with the majority
    }
    // 10000 = 100% * 100 (hundredths). Integer division truncates -> deterministic.
    return static_cast<int>((correct * 10000) / d.P);
}

int main(int argc, char** argv) {
    // ---- 1. Load ----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/ifp_sample.txt";
    Dataset d;
    try {
        d = load_dataset(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. STAGE A: build IFPs (CPU + GPU), verify bit-exact -------------
    std::vector<uint64_t> fps_cpu, fps_gpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    build_ifps(d, fps_cpu);
    const double buildA_cpu_ms = cpu_timer.stop_ms();

    float buildA_gpu_ms = 0.0f;
    build_ifps_gpu(d, fps_gpu, &buildA_gpu_ms);

    bool ifp_match = (fps_cpu.size() == fps_gpu.size());
    if (ifp_match)
        for (std::size_t i = 0; i < fps_cpu.size(); ++i)
            if (fps_cpu[i] != fps_gpu[i]) { ifp_match = false; break; }

    // ---- 3. STAGE B: cluster (CPU + GPU), verify labels + centroids -------
    std::vector<uint64_t> cent_cpu, cent_gpu;
    std::vector<int> lab_cpu, lab_gpu;
    std::vector<unsigned int> sz_cpu, sz_gpu;

    cpu_timer.start();
    const double cost_cpu = ifp_cluster_cpu(fps_cpu, d.P, d.K, ITERS, cent_cpu, lab_cpu, sz_cpu);
    const double cluster_cpu_ms = cpu_timer.stop_ms();

    float cluster_gpu_ms = 0.0f;
    const double cost_gpu = ifp_cluster_gpu(fps_gpu, d.P, d.K, ITERS,
                                            cent_gpu, lab_gpu, sz_gpu, &cluster_gpu_ms);

    int label_mismatch = 0;
    for (int i = 0; i < d.P; ++i) if (lab_cpu[i] != lab_gpu[i]) ++label_mismatch;
    bool cent_match = (cent_cpu.size() == cent_gpu.size());
    if (cent_match)
        for (std::size_t i = 0; i < cent_cpu.size(); ++i)
            if (cent_cpu[i] != cent_gpu[i]) { cent_match = false; break; }

    const bool pass = ifp_match && (label_mismatch == 0) && cent_match;

    // ---- 4. Deterministic report -> STDOUT --------------------------------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("pocket: %d residues x %d interaction types = %d IFP bits\n",
                NUM_RESIDUES, NUM_ITYPES, IFP_BITS);
    std::printf("stage A: built %d interaction fingerprints (CPU==GPU: %s)\n",
                d.P, ifp_match ? "yes" : "NO");
    std::printf("stage B: %d poses -> %d binding-mode clusters, %d iterations\n",
                d.P, d.K, ITERS);

    // Per-cluster: size + the residues/types in its CONSENSUS fingerprint.
    for (int k = 0; k < d.K; ++k) {
        std::printf("  cluster %d (n=%4u): consensus contacts =", k, sz_gpu[k]);
        int printed = 0;
        for (int r = 0; r < NUM_RESIDUES; ++r) {
            for (int t = 0; t < NUM_ITYPES; ++t) {
                const int b = r * NUM_ITYPES + t;
                // Read the consensus bit straight from the GPU centroid bit-vector.
                const uint64_t* c = &cent_gpu[static_cast<std::size_t>(k) * FP_WORDS];
                if ((c[b >> 6] >> (b & 63)) & 1ull) {
                    std::printf(" R%d:%s", r, ITYPE_NAME[t]);
                    ++printed;
                }
            }
        }
        if (printed == 0) std::printf(" (none)");
        std::printf("\n");
    }

    const int purity = cluster_purity_centi(d, lab_gpu);   // hundredths of a percent
    std::printf("cost = %.4f\n", cost_gpu);
    if (purity >= 0)
        std::printf("mode recovery (purity vs planted modes) = %d.%02d%%\n",
                    purity / 100, purity % 100);
    std::printf("RESULT: %s (GPU IFPs + labels + centroids match CPU)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d poses, %d residues, K=%d)\n",
                 path.c_str(), d.P, NUM_RESIDUES, d.K);
    std::fprintf(stderr, "[timing] stage A build  CPU %.3f ms | GPU %.3f ms\n",
                 buildA_cpu_ms, buildA_gpu_ms);
    std::fprintf(stderr, "[timing] stage B cluster CPU %.3f ms | GPU %.3f ms\n",
                 cluster_cpu_ms, cluster_gpu_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- tiny sample is launch/copy bound; "
                         "the GPU's edge grows with pose count (10^3-10^6 in real runs).\n");
    std::fprintf(stderr, "[verify] IFP bit match=%s | label mismatches=%d | centroid match=%s | "
                         "cost(cpu/gpu)=%.4f/%.4f\n",
                 ifp_match ? "yes" : "no", label_mismatch, cent_match ? "yes" : "no",
                 cost_cpu, cost_gpu);

    return pass ? 0 : 1;
}
