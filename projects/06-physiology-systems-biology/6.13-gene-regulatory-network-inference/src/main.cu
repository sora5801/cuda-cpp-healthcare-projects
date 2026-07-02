// ===========================================================================
// src/main.cu  --  Entry point: load data, run CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 6.13 : Gene Regulatory Network Inference (ARACNE: MI + DPI)
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the expression matrix (from data/sample, or fail loudly).
//   2. Compute the CPU reference (reference_cpu.cpp): discretize -> MI -> DPI.
//   3. Compute the GPU result    (kernels.cu): the same pipeline, in parallel.
//   4. VERIFY: GPU MI matches CPU MI within tolerance AND the pruned edge set is
//      IDENTICAL (an exact integer comparison).
//   5. REPORT: deterministic inferred network to stdout; timing to stderr.
//
//   STDOUT is byte-for-byte deterministic (diffed by demo/run_demo against
//   demo/expected_output.txt); run-varying timings go to STDERR.
//
// READ THIS FIRST in the code tour, then grn.h -> reference_cpu.h ->
// kernels.cuh -> kernels.cu. See ../THEORY.md for the "why".
// ===========================================================================
#include <algorithm>
#include <cmath>       // std::fabs
#include <cstdio>
#include <limits>      // std::numeric_limits
#include <numeric>
#include <string>
#include <vector>

#include "kernels.cuh"        // grn_infer_gpu (GPU path)
#include "reference_cpu.h"    // GrnData, load_expression, *_cpu (CPU baseline)
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "6.13";
static const char* PROJECT_NAME = "Gene Regulatory Network Inference";

// --- Tuning constants (documented so the learner can experiment) ------------
// MI significance floor: edges below this MI (nats) are never reported. On the
// tiny synthetic sample the true edges sit well above ~0.2 nats and the noise
// pairs well below, so this cleanly separates signal from chance (THEORY sec
// "How we verify correctness"). DPI slack avoids pruning near-tied triangles.
static constexpr double MI_THRESHOLD = 0.20;   // nats
static constexpr double DPI_TOLERANCE = 0.02;  // nats

// MI verification tolerance. CPU and GPU build IDENTICAL integer joint counts
// and evaluate the SAME double-precision log sum (grn.h), so they differ only in
// the last ~1 ULP of the transcendental accumulation. 1e-9 is a generous margin
// that still catches any real algorithmic divergence (PATTERNS.md sec 4).
static constexpr double MI_TOLERANCE = 1.0e-9;

// Largest |a[i]-b[i]| over two equal-length double arrays (our MI headline
// metric). Returns +inf on a length mismatch so a shape bug can't look like a
// pass. (util::max_abs_err is float-only; MI is double, hence this local twin.)
static double max_abs_err_d(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double d = std::fabs(a[i] - b[i]);
        if (d > worst) worst = d;
    }
    return worst;
}

// Collect the surviving (i<j) edges into a deterministic list, sorted by
// descending MI with ties broken by (i, then j) so the report is reproducible.
struct Edge { int i, j; double mi; };
static std::vector<Edge> collect_edges(const std::vector<double>& mi,
                                       const std::vector<uint8_t>& keep, int G) {
    std::vector<Edge> e;
    for (int i = 0; i < G; ++i)
        for (int j = i + 1; j < G; ++j)
            if (keep[static_cast<std::size_t>(i) * G + j])
                e.push_back({i, j, mi[static_cast<std::size_t>(i) * G + j]});
    std::sort(e.begin(), e.end(), [](const Edge& a, const Edge& b) {
        if (a.mi != b.mi) return a.mi > b.mi;      // stronger edge first
        if (a.i  != b.i ) return a.i  < b.i;       // deterministic tie-break
        return a.j < b.j;
    });
    return e;
}

int main(int argc, char** argv) {
    // ---- 1. Load ----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1]
                                        : "data/sample/expression_sample.txt";
    GrnData data;
    try {
        data = load_expression(path);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "[error] %s\n", ex.what());
        return 2;
    }
    const int G = data.n_genes, S = data.n_samples;

    // ---- 2. CPU reference (timed): discretize -> MI -> DPI ----------------
    std::vector<double>  mi_cpu;
    std::vector<uint8_t> keep_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    discretize_matrix(data);                       // fills data.disc
    mi_matrix_cpu(data, mi_cpu);
    dpi_prune_cpu(mi_cpu, G, MI_THRESHOLD, DPI_TOLERANCE, keep_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU result (kernels timed inside the wrapper) -----------------
    std::vector<double>  mi_gpu;
    std::vector<uint8_t> keep_gpu;
    float mi_ms = 0.0f, dpi_ms = 0.0f;
    grn_infer_gpu(data, MI_THRESHOLD, DPI_TOLERANCE, mi_gpu, keep_gpu, &mi_ms, &dpi_ms);

    // ---- 4. Verify --------------------------------------------------------
    const double mi_err = max_abs_err_d(mi_cpu, mi_gpu);   // continuous check
    bool mask_match = (keep_cpu.size() == keep_gpu.size());// exact-set check
    for (std::size_t k = 0; mask_match && k < keep_cpu.size(); ++k)
        if (keep_cpu[k] != keep_gpu[k]) mask_match = false;
    const bool pass = (mi_err <= MI_TOLERANCE) && mask_match;

    // Build the inferred network from the (verified) GPU result.
    const std::vector<Edge> edges = collect_edges(mi_gpu, keep_gpu, G);

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("ARACNE mutual-information network: %d genes x %d samples, %d bins\n",
                G, S, N_BINS);
    std::printf("inferred %zu direct edge(s) after MI>%.2f + DPI pruning:\n",
                edges.size(), MI_THRESHOLD);
    for (const Edge& e : edges) {
        const char* ni = (e.i < (int)data.gene_names.size()) ? data.gene_names[e.i].c_str() : "?";
        const char* nj = (e.j < (int)data.gene_names.size()) ? data.gene_names[e.j].c_str() : "?";
        std::printf("  %-6s -- %-6s   I = %.4f nats\n", ni, nj, e.mi);
    }
    std::printf("RESULT: %s (GPU MI matches CPU within tol=1.0e-09; edge sets identical)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (G=%d genes, S=%d samples)\n", path.c_str(), G, S);
    std::fprintf(stderr, "[timing] CPU pipeline: %.3f ms   GPU MI: %.3f ms   GPU DPI: %.3f ms\n",
                 cpu_ms, mi_ms, dpi_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- with only %lld pairs this tiny sample "
                         "is launch-bound; the O(G^2) GPU win grows with gene count.\n",
                 static_cast<long long>(G) * (G - 1) / 2);
    std::fprintf(stderr, "[verify] MI max_abs_err = %.3e (tol %.1e); edge masks %s\n",
                 mi_err, MI_TOLERANCE, mask_match ? "identical" : "DIFFER");

    return pass ? 0 : 1;
}
