// ===========================================================================
// src/main.cu  --  Entry point: load data, run CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 3.11 : GWAS at Scale
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the genotype matrix + phenotype (data/sample).
//   2. CPU reference (reference_cpu.cpp): standardize -> GRM -> assoc scan.
//   3. GPU result  (kernels.cu): standardize+DGEMM GRM, one-thread-per-SNP scan.
//   4. VERIFY: GPU GRM and GPU association results match the CPU within a
//      documented tolerance (the correctness guarantee).
//   5. REPORT: a DETERMINISTIC summary to stdout (top associated SNPs, GRM
//      diagnostics, recovery of the injected causal SNPs); timing to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Anything that varies run-to-run (timings) goes to
//   STDERR, which the demo shows but does not diff (PATTERNS.md §3).
//
// THE SCIENCE IN ONE BREATH
//   A GWAS asks, for each genetic variant (SNP), "does carrying more of this
//   allele shift the trait?" We answer it two ways the GPU accelerates:
//     * GRM = (1/M)·Z·Zᵀ -- who is related to whom (cuBLAS DGEMM); the input to
//       a mixed model that corrects for population structure.
//     * a per-SNP regression scan -- the association test itself.
//   The synthetic sample injects a handful of CAUSAL SNPs with a known effect,
//   so a correct pipeline must rank them at the top (PATTERNS.md §6). That
//   recovery is the headline, human-meaningful result.
//
// READ THIS FIRST in the code tour, then kernels.cuh -> kernels.cu, then
// reference_cpu.cpp for the baseline, and gwas_core.h for the shared math.
// See ../THEORY.md for the full "why".
// ===========================================================================
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "kernels.cuh"        // gpu_build_grm, gpu_assoc_scan
#include "reference_cpu.h"    // load_genotypes, standardize_columns, grm/assoc refs
#include "gwas_core.h"        // gwas::AssocResult
#include "util/io.hpp"        // util::CpuTimer

// Identify the program. Must stay in sync with demo/expected_output.txt.
static const char* PROJECT_ID   = "3.11";
static const char* PROJECT_NAME = "GWAS at Scale";

// ---- Verification tolerances (documented; PATTERNS.md §4) ------------------
// Both sides run the SAME double-precision arithmetic from gwas_core.h, but the
// GPU's DGEMM sums the Z·Zᵀ dot products in a different ORDER than the CPU's
// serial loop, and uses fused multiply-add. Floating-point addition is not
// associative, so the GRM agrees only to ~1e-10, not bit-exactly -- a real and
// teachable effect. We verify to a tolerance that is far below any genetic
// signal. The association scan accumulates in the same per-SNP order on both
// sides, so it agrees far more tightly; we still use a generous double-eps tol.
static constexpr double GRM_TOL   = 1.0e-9;   // entrywise |GRM_gpu - GRM_cpu|
static constexpr double ASSOC_TOL = 1.0e-9;   // per-SNP |chi2_gpu - chi2_cpu|

// A SNP plus its computed association, used for ranking the top hits.
struct RankedSNP {
    int    index;       // column j in the genotype matrix
    double chi2;        // association strength (t^2)
    double neg_log10p;  // -log10(p), the GWAS "Manhattan plot" height
    double beta;        // effect size (per SD of genotype)
    int    causal;      // 1 if this SNP was injected as truly causal (demo truth)
};

int main(int argc, char** argv) {
    // ---- 1. Load the genotype matrix + phenotype ---------------------------
    const std::string path =
        (argc > 1) ? argv[1] : "data/sample/gwas_sample.txt";
    GenotypeData d;
    try {
        d = load_genotypes(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int N = d.N, M = d.M;

    // Center the phenotype once (so the regression intercept drops out).
    std::vector<double> y = center_phenotype(d.pheno);

    // ---- 2. CPU reference (timed) ------------------------------------------
    //   standardize -> GRM -> association scan, all serial and obvious.
    std::vector<double> Z, freq, sd;
    std::vector<double> grm_cpu;
    std::vector<gwas::AssocResult> assoc_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    standardize_columns(d, Z, freq, sd);
    grm_reference(Z, N, M, grm_cpu);
    assoc_reference(Z, y, N, M, assoc_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU result (kernels timed inside the wrappers) -----------------
    std::vector<double> grm_gpu;
    std::vector<gwas::AssocResult> assoc_gpu;
    float std_ms = 0.0f, gemm_ms = 0.0f, assoc_ms = 0.0f;
    gpu_build_grm(d.geno, N, M, grm_gpu, &std_ms, &gemm_ms);
    gpu_assoc_scan(d.geno, y, N, M, assoc_gpu, &assoc_ms);

    // ---- 4. Verify GPU vs CPU ----------------------------------------------
    // (a) GRM: worst entrywise difference over the whole NxN matrix.
    double grm_worst = 0.0;
    for (std::size_t i = 0; i < grm_cpu.size(); ++i)
        grm_worst = std::fmax(grm_worst, std::fabs(grm_cpu[i] - grm_gpu[i]));
    // (b) Association: worst chi-square difference over all SNPs.
    double assoc_worst = 0.0;
    for (int j = 0; j < M; ++j)
        assoc_worst = std::fmax(assoc_worst,
                                std::fabs(assoc_cpu[j].chi2 - assoc_gpu[j].chi2));
    const bool pass = (grm_worst <= GRM_TOL) && (assoc_worst <= ASSOC_TOL);

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    // Rank SNPs by association strength (GPU results -- verified == CPU above).
    std::vector<RankedSNP> ranked(M);
    for (int j = 0; j < M; ++j) {
        ranked[j] = RankedSNP{ j, assoc_gpu[j].chi2, assoc_gpu[j].neg_log10p,
                               assoc_gpu[j].beta, d.causal[j] };
    }
    // Deterministic sort: by chi2 descending, ties broken by SNP index ascending
    // so the printed order never depends on the (unspecified) sort stability.
    std::sort(ranked.begin(), ranked.end(), [](const RankedSNP& a, const RankedSNP& b) {
        if (a.chi2 != b.chi2) return a.chi2 > b.chi2;
        return a.index < b.index;
    });

    // GRM diagnostics: mean diagonal (~1, self-relatedness) and largest
    // off-diagonal (a cryptic-relatedness / population-structure signal).
    double diag_sum = 0.0, max_off = 0.0;
    int off_a = 0, off_b = 0;
    for (int aa = 0; aa < N; ++aa) {
        diag_sum += grm_gpu[static_cast<std::size_t>(aa) * N + aa];
        for (int bb = 0; bb < aa; ++bb) {
            double v = grm_gpu[static_cast<std::size_t>(aa) * N + bb];
            if (std::fabs(v) > std::fabs(max_off)) { max_off = v; off_a = aa; off_b = bb; }
        }
    }
    const double mean_diag = diag_sum / static_cast<double>(N);

    // How many of the injected causal SNPs land in the top-K hits? (demo truth)
    int n_causal = std::accumulate(d.causal.begin(), d.causal.end(), 0);
    const int K = (M < 10) ? M : 10;
    int causal_in_topK = 0;
    for (int r = 0; r < K; ++r) causal_in_topK += ranked[r].causal;

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("cohort: N=%d individuals, M=%d SNPs (synthetic)\n", N, M);
    std::printf("GRM: mean diagonal=%.4f, max |off-diagonal|=%.4f at (%d,%d)\n",
                mean_diag, max_off, off_a, off_b);
    std::printf("causal SNPs injected: %d ; recovered in top %d: %d\n",
                n_causal, K, causal_in_topK);
    std::printf("top %d associated SNPs (rank: id chi2 -log10p beta causal):\n", K);
    for (int r = 0; r < K; ++r) {
        const RankedSNP& s = ranked[r];
        std::printf("  %2d: %-8s chi2=%9.4f  -log10p=%7.4f  beta=%+8.5f  %s\n",
                    r + 1, d.snp_id[s.index].c_str(), s.chi2, s.neg_log10p,
                    s.beta, s.causal ? "CAUSAL" : "-");
    }
    std::printf("RESULT: %s (GPU GRM and association match CPU within tol)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (N=%d, M=%d)\n", path.c_str(), N, M);
    std::fprintf(stderr, "[timing] CPU reference (std+GRM+assoc): %.3f ms\n", cpu_ms);
    std::fprintf(stderr, "[timing] GPU standardize: %.3f ms   cuBLAS DGEMM (GRM): %.3f ms"
                         "   assoc scan: %.3f ms\n", std_ms, gemm_ms, assoc_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this sample is tiny, so the "
                         "GPU is launch/copy bound; the DGEMM's edge grows as O(N^2 M).\n");
    std::fprintf(stderr, "[verify] GRM worst entry diff   = %.3e  (tol %.1e)\n",
                 grm_worst, GRM_TOL);
    std::fprintf(stderr, "[verify] assoc worst chi2 diff  = %.3e  (tol %.1e)\n",
                 assoc_worst, ASSOC_TOL);

    return pass ? 0 : 1;
}
