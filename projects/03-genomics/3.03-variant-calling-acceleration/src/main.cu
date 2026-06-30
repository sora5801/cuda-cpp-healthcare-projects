// ===========================================================================
// src/main.cu  --  Entry point: load reads/haplotypes, run CPU + GPU, verify
// ---------------------------------------------------------------------------
// Project 3.3 : Variant Calling Acceleration
//
// 5-step shape (the shape EVERY project in this repo follows):
//   1. Load the problem: reads, candidate haplotypes, qualities (data/sample).
//   2. CPU reference PairHMM log-likelihood matrix (reference_cpu.cpp).
//   3. GPU PairHMM log-likelihood matrix (kernels.cu): 1 thread per pair.
//   4. VERIFY: GPU matrix matches CPU within a documented tolerance.
//   5. REPORT: deterministic result to stdout; timing to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Anything that varies run-to-run (timings) goes to
//   STDERR, shown but not diffed.
//
// THE RESULT WE REPORT
//   For each read, the most-likely haplotype (argmax over its row of the
//   log-likelihood matrix). The synthetic sample is engineered so every read was
//   drawn (with a few sequencing errors) from one known "truth" haplotype, so the
//   headline number is "reads assigned to the truth haplotype" -- which should be
//   all of them. This makes the demo's correctness legible at a glance.
//
// Code tour: start here, then kernels.cuh -> kernels.cu, then reference_cpu.cpp,
// with the shared math in pairhmm_core.h. See ../THEORY.md for the "why".
// ===========================================================================
#include <cmath>      // std::fabs, std::isinf
#include <cstdio>
#include <limits>     // std::numeric_limits<double>::infinity()
#include <string>
#include <vector>

#include "kernels.cuh"        // pairhmm_gpu (GPU path), VariantData
#include "reference_cpu.h"    // load_variant_data, pairhmm_cpu, best_haplotype_per_read
#include "util/io.hpp"        // util::CpuTimer

// These two tokens identify the program; they MUST stay in sync with
// demo/expected_output.txt (the demo diffs stdout against it).
static const char* PROJECT_ID   = "3.3";
static const char* PROJECT_NAME = "Variant Calling Acceleration";

// Correctness tolerance on log10-likelihoods. CPU and GPU run the SAME IEEE-754
// double operations (pairhmm_core.h), so they agree to a few ULP; over a handful
// of multiply-adds per cell that is well under 1e-9. We verify to 1e-9 and report
// the actual error (typically ~1e-13). See PATTERNS.md §4 and THEORY.md §verify.
static constexpr double TOLERANCE = 1.0e-9;

// max_abs_err over two double vectors, treating matching -inf entries (impossible
// pairs) as equal. Returns +inf on a length mismatch so a shape bug can never be
// mistaken for agreement.
static double max_abs_err_double(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        // Both -inf -> identical impossible pair, no error contribution.
        if (std::isinf(a[i]) && std::isinf(b[i]) && (a[i] < 0) == (b[i] < 0)) continue;
        const double d = std::fabs(a[i] - b[i]);
        if (d > worst) worst = d;
    }
    return worst;
}

int main(int argc, char** argv) {
    // ---- 1. Load the problem -----------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/reads_haplotypes_sample.txt";
    VariantData v;
    try {
        v = load_variant_data(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) ------------------------------------------
    std::vector<double> loglik_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    pairhmm_cpu(v, loglik_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU PairHMM (kernel timed inside the wrapper) ------------------
    std::vector<double> loglik_gpu;
    float gpu_kernel_ms = 0.0f;
    pairhmm_gpu(v, loglik_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ----------------------------------------------------------
    const double err = max_abs_err_double(loglik_cpu, loglik_gpu);
    const bool pass = err <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    // We report from the GPU matrix (it is what we are teaching); it matches the
    // CPU within TOLERANCE, so the argmax assignments are identical either way.
    std::vector<int> best;
    best_haplotype_per_read(v, loglik_gpu, best);

    int n_correct = 0;   // reads whose best haplotype is the known truth
    for (int r = 0; r < v.n_reads; ++r)
        if (v.truth >= 0 && best[r] == v.truth) ++n_correct;

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("PairHMM forward: %d reads x %d haplotypes (read_len=%d, hap_len=%d)\n",
                v.n_reads, v.n_haps, v.read_len, v.hap_len);
    std::printf("per-read best haplotype (argmax log10 P(read|hap)):\n");
    for (int r = 0; r < v.n_reads; ++r) {
        const double ll = loglik_gpu[static_cast<std::size_t>(r) * v.n_haps + best[r]];
        std::printf("  read %2d -> hap %d   log10L = %.6f\n", r, best[r], ll);
    }
    if (v.truth >= 0)
        std::printf("reads assigned to truth haplotype %d: %d of %d\n",
                    v.truth, n_correct, v.n_reads);
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-09)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%d reads, %d haplotypes)\n",
                 path.c_str(), v.n_reads, v.n_haps);
    std::fprintf(stderr, "[model]  delta(gap-open)=%.4g  epsilon(gap-extend)=%.4g\n",
                 v.params.delta, v.params.epsilon);
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- tiny inputs are dominated by launch/copy "
                         "overhead; the GPU's edge grows with the number of read-haplotype pairs.\n");
    std::fprintf(stderr, "[verify] max_abs_err = %.3e  (tolerance %.1e)\n", err, TOLERANCE);

    // Exit code feeds the demo's pass/fail gate.
    return pass ? 0 : 1;
}
