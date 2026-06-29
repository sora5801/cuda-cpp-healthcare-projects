// ===========================================================================
// src/main.cu  --  Entry point: load sequences, scan, verify, report
// ---------------------------------------------------------------------------
// Project 1.34 : Amyloid / Aggregation Propensity Prediction
//
// 5-step shape (every project in this repo follows it):
//   1. LOAD the protein batch (FASTA-style sample, or a built-in fallback).
//   2. CPU reference aggregation scan (reference_cpu.cpp)   -> trusted answer.
//   3. GPU aggregation scan (kernels.cu, tiled window)      -> the thing taught.
//   4. VERIFY: GPU smoothed profiles + per-protein results == CPU within tol.
//   5. REPORT: deterministic ranking & one profile to STDOUT; timing to STDERR.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Run-varying numbers (timings) go to STDERR.
//
// Code tour: start here, then propensity.h (the shared physics) ->
//   reference_cpu.{h,cpp} (loader + CPU scan) -> kernels.{cuh,cu} (the tiling).
//   See ../THEORY.md for the science, math, and GPU mapping.
// ===========================================================================
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // scan_dataset_gpu, AGG_MAX_LEN
#include "reference_cpu.h"    // load_dataset, scan_dataset_cpu, Dataset, AggResult
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "1.34";
static const char* PROJECT_NAME = "Amyloid / Aggregation Propensity Prediction";

// Scan parameters (documented in THEORY.md; tunable -- see README Exercises).
//   WINDOW    : sliding-window width W (odd). 7 residues is a common APR scale --
//               long enough to capture a beta-prone stretch, short enough to
//               localize the hot spot. (TANGO/AGGRESCAN use ~5-11.)
//   THRESHOLD : smoothed score >= this marks an aggregation-prone residue. 0.55
//               sits above the mixed/charged background and below a solid
//               hydrophobic core for this didactic scale (THEORY §2).
//   TOL       : verification tolerance. Because the CPU and GPU call the SAME
//               windowed_mean() in the SAME order over identical floats, the
//               smoothed profiles agree to a few ULPs; 1e-5 is comfortably tight
//               and honest (PATTERNS.md §4: same-ops-both-sides -> ~fp epsilon).
static constexpr int    WINDOW    = 7;
static constexpr float  THRESHOLD = 0.55f;
static constexpr double TOL       = 1.0e-5;

// One-letter code -> readable char, for printing the peak residue. Inverse of
// code_of_char's first 20 cases; index 20 ("other") prints 'X'.
static char char_of_code(int code) {
    static const char* AA = "ARNDCQEGHILKMFPSTWYVX";
    return (code >= 0 && code < 21) ? AA[code] : 'X';
}

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path =
        (argc > 1) ? argv[1] : "data/sample/amyloid_sample.fasta";
    Dataset ds;
    try {
        ds = load_dataset(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<AggResult> res_cpu;
    std::vector<float> smooth_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    scan_dataset_cpu(ds, WINDOW, THRESHOLD, res_cpu, &smooth_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU scan (kernel timed inside the wrapper) --------------------
    std::vector<AggResult> res_gpu;
    std::vector<float> smooth_gpu;
    float gpu_kernel_ms = 0.0f;
    scan_dataset_gpu(ds, WINDOW, THRESHOLD, res_gpu, smooth_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    // (a) smoothed profiles agree element-wise within TOL; (b) the integer
    // reductions (peak_pos, prone_count, longest_apr) agree EXACTLY; (c) the
    // peak_score agrees within TOL. Any mismatch fails the demo.
    double max_err = 0.0;
    bool ints_match = true;
    for (std::size_t k = 0; k < smooth_cpu.size(); ++k)
        max_err = std::max(max_err,
            std::fabs(static_cast<double>(smooth_cpu[k]) - smooth_gpu[k]));
    for (int p = 0; p < ds.num; ++p) {
        const AggResult& a = res_cpu[p];
        const AggResult& b = res_gpu[p];
        if (a.peak_pos != b.peak_pos || a.prone_count != b.prone_count ||
            a.longest_apr != b.longest_apr) ints_match = false;
        max_err = std::max(max_err,
            std::fabs(static_cast<double>(a.peak_score) - b.peak_score));
    }
    const bool pass = (max_err <= TOL) && ints_match;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) ----------
    // Rank proteins by peak smoothed score (desc), tie-break by input order, so
    // the "most aggregation-prone" sequences come first -- the headline a
    // developability/liability screen produces. Ranking is on integers/floats
    // computed identically on both sides, so the order is deterministic.
    std::vector<int> order(ds.num);
    for (int p = 0; p < ds.num; ++p) order[p] = p;
    std::stable_sort(order.begin(), order.end(), [&](int i, int j) {
        return res_gpu[i].peak_score > res_gpu[j].peak_score;
    });

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("sequence-based APR scan: %d proteins, window W=%d, threshold=%.2f\n",
                ds.num, WINDOW, THRESHOLD);
    std::printf("(synthetic sequences; intrinsic propensity scale -- not for clinical use)\n");
    std::printf("rank  protein                         len  peak  pos  resi  prone  APR\n");
    for (int rk = 0; rk < ds.num; ++rk) {
        const int p = order[rk];
        const AggResult& r = res_gpu[p];
        const int pos = r.peak_pos;
        const char resi = char_of_code(
            ds.lengths[p] > 0 ? ds.flat_codes[(std::size_t)p * ds.stride + pos] : 20);
        // Truncate/pad the name to a fixed 30 cols so the table is deterministic.
        std::string nm = ds.proteins[p].name;
        if (nm.size() > 30) nm = nm.substr(0, 30);
        std::printf("%4d  %-30s %4d %.3f %4d   %c   %4d %4d\n",
                    rk + 1, nm.c_str(), ds.lengths[p], r.peak_score, pos, resi,
                    r.prone_count, r.longest_apr);
    }

    // Show the smoothed profile of the single most aggregation-prone protein,
    // sampled at up to 12 evenly-spaced residues, so the learner can SEE the hot
    // spot rise above the threshold.
    const int top = order[0];
    const int tlen = ds.lengths[top];
    const std::size_t tbase = static_cast<std::size_t>(top) * ds.stride;
    std::printf("top hit '%s' smoothed profile (%d pts):",
                ds.proteins[top].name.c_str(), tlen < 12 ? tlen : 12);
    const int npts = tlen < 12 ? tlen : 12;
    for (int s = 0; s < npts; ++s) {
        const int i = (npts == 1) ? 0 : (s * (tlen - 1)) / (npts - 1);
        std::printf(" %.3f", smooth_gpu[tbase + i]);
    }
    std::printf("\n");
    std::printf("RESULT: %s (GPU matches CPU: max_abs_err <= %.0e, integer fields exact)\n",
                pass ? "PASS" : "FAIL", TOL);

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) -----------------
    std::fprintf(stderr, "[data]   source: %s  (%d proteins, max_len=%d)\n",
                 path.c_str(), ds.num, ds.max_len);
    std::fprintf(stderr, "[timing] CPU scan: %.3f ms   GPU tiled scan: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- a few short sequences are launch-bound; "
                         "the GPU's edge grows when batching tens of thousands of proteins.\n");
    std::fprintf(stderr, "[verify] max_abs_err(smoothed/peak) = %.3e (tol %.1e); integer fields %s\n",
                 max_err, TOL, ints_match ? "match" : "DIFFER");

    return pass ? 0 : 1;
}
