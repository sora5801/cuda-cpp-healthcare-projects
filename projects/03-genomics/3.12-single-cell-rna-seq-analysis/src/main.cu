// ===========================================================================
// src/main.cu  --  Entry point: load data, run CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 3.12 : Single-Cell RNA-seq Analysis  (reduced-scope teaching version)
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the tiny scRNA-seq count matrix from data/sample (or arg path).
//   2. Compute the CPU reference (reference_cpu.cpp): normalize + KNN graph.
//   3. Compute the GPU result    (kernels.cu): the same, in parallel.
//   4. VERIFY: neighbour indices match EXACTLY; normalized values + distances
//      match within a documented tolerance.
//   5. REPORT: deterministic KNN graph + a label-purity score to stdout;
//      timing to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Anything that varies run-to-run (timings) goes to
//   STDERR, which the demo shows but does not diff (docs/PATTERNS.md sec 3).
//
// CODE TOUR: read this first, then scrna.h (the shared math), then kernels.cuh
//   -> kernels.cu (the GPU twin), and reference_cpu.cpp (the baseline). See
//   ../THEORY.md for the science and the GPU mapping.
// ===========================================================================
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // run_gpu (GPU path), Dataset, KnnGraph
#include "reference_cpu.h"    // load_dataset, run_cpu (CPU baseline)
#include "util/io.hpp"        // util::CpuTimer

// Self-identification. Kept in sync with demo/expected_output.txt.
static const char* PROJECT_ID   = "3.12";
static const char* PROJECT_NAME = "Single-Cell RNA-seq Analysis";

// ---------------------------------------------------------------------------
// VERIFICATION TOLERANCES (documented; docs/PATTERNS.md sec 4)
//   IDX_EXACT      : neighbour INDICES are integers produced by the SAME shared
//                    scan with a deterministic tie-break -> they must match with
//                    ZERO mismatches. This is the strong, exact correctness gate.
//   FLOAT_TOL      : normalized values and reported distances are floats. The
//                    per-element math is identical on both sides AND runs in the
//                    same order, so in practice they agree to the last bit; we
//                    still allow a tiny 1e-5 slack to be honest about float (a
//                    different host-compiler FMA contraction could nudge the last
//                    ULP). If this were ever exceeded it would signal a real bug.
// ---------------------------------------------------------------------------
static constexpr int    IDX_EXACT = 0;        // required: zero neighbour-index mismatches
static constexpr double FLOAT_TOL = 1.0e-5;   // slack for normalized values + distances

// max_abs_diff_f: largest |a[i]-b[i]| over two equal-length float vectors.
static double max_abs_diff_f(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1.0e308;   // shape mismatch -> "infinitely" far
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        worst = std::fmax(worst, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    return worst;
}

int main(int argc, char** argv) {
    // ---- 1. Load the count matrix ------------------------------------------
    const std::string path = (argc > 1) ? argv[1]
                                        : "data/sample/scrna_sample.txt";
    Dataset d;
    try {
        d = load_dataset(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) ------------------------------------------
    KnnGraph cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    run_cpu(d, cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU result (kernels timed inside the wrapper) ------------------
    KnnGraph gpu;
    float gpu_kernel_ms = 0.0f;
    run_gpu(d, gpu, &gpu_kernel_ms);

    // ---- 4. Verify ----------------------------------------------------------
    // (a) Neighbour indices: count any position where CPU and GPU disagree.
    int idx_mismatch = 0;
    for (std::size_t i = 0; i < cpu.nbr_idx.size(); ++i)
        if (cpu.nbr_idx[i] != gpu.nbr_idx[i]) ++idx_mismatch;
    // (b) Normalized matrix + reported distances: float agreement.
    const double norm_diff = max_abs_diff_f(cpu.normalized, gpu.normalized);
    const double dist_diff = max_abs_diff_f(cpu.nbr_dist,  gpu.nbr_dist);
    const bool pass = (idx_mismatch == IDX_EXACT)
                      && (norm_diff <= FLOAT_TOL) && (dist_diff <= FLOAT_TOL);

    // ---- 4b. Science check: KNN label purity -------------------------------
    // The synthetic sample embeds known cell TYPES (labels). A good KNN graph
    // connects cells to others of the SAME type. We measure purity = fraction of
    // graph edges whose two endpoints share a label (skipping unknown -1). This
    // is an integer ratio (deterministic) and recovers the embedded structure.
    long same_type = 0, counted = 0;
    for (int q = 0; q < d.N; ++q) {
        const int lq = d.labels[q];
        if (lq < 0) continue;
        for (int j = 0; j < d.k; ++j) {
            const int nb = gpu.nbr_idx[static_cast<std::size_t>(q) * d.k + j];
            const int ln = d.labels[nb];
            if (ln < 0) continue;
            ++counted;
            if (ln == lq) ++same_type;
        }
    }
    // Purity in basis points (integer) so the printed value is byte-deterministic.
    const long purity_bp = counted ? (same_type * 10000 + counted / 2) / counted : 0;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("scRNA-seq KNN graph: %d cells x %d genes, k=%d, target_sum=%.0f\n",
                d.N, d.G, d.k, d.target_sum);
    std::printf("pipeline: library-size normalize (counts-per-target) + log1p, then exact brute-force KNN\n");

    // Print the KNN graph: each cell's neighbours (indices, nearest first) with
    // the nearest-neighbour distance. Integer indices + fixed precision keep this
    // line-for-line reproducible.
    std::printf("cell(type)  ->  k nearest neighbours [d1]\n");
    for (int q = 0; q < d.N; ++q) {
        std::printf("  %2d(t%d) ->", q, d.labels[q]);
        for (int j = 0; j < d.k; ++j)
            std::printf(" %2d", gpu.nbr_idx[static_cast<std::size_t>(q) * d.k + j]);
        std::printf("  [d1=%.4f]\n", gpu.nbr_dist[static_cast<std::size_t>(q) * d.k]);
    }

    // A couple of normalized values, fixed precision, so the normalize step is
    // also pinned by the expected output (not just the graph).
    std::printf("normalized[cell0, gene0..2] = %.4f %.4f %.4f\n",
                gpu.normalized[0], gpu.normalized[1], gpu.normalized[2]);
    std::printf("KNN label purity = %ld.%02ld%% (%ld/%ld edges connect same-type cells)\n",
                purity_bp / 100, purity_bp % 100, same_type, counted);
    std::printf("RESULT: %s (GPU neighbour indices match CPU exactly)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%d cells, %d genes)\n", path.c_str(), d.N, d.G);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernels: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- brute-force KNN is O(N^2); the GPU edge "
                         "grows with cell count (real runs are 10^5-10^7 cells).\n");
    std::fprintf(stderr, "[verify] neighbour-index mismatches = %d, max |norm| diff = %.3e, "
                         "max |dist| diff = %.3e (tol %.1e)\n",
                 idx_mismatch, norm_diff, dist_diff, FLOAT_TOL);

    // Exit code feeds the demo's pass/fail gate.
    return pass ? 0 : 1;
}
