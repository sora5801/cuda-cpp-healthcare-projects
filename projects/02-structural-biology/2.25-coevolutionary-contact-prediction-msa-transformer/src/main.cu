// ===========================================================================
// src/main.cu  --  Entry point: load MSA, run CPU + GPU, verify, predict contacts
// ---------------------------------------------------------------------------
// Project 2.25 : Coevolutionary Contact Prediction & MSA Transformer
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the MSA (FASTA from data/sample, or the path given on the cmdline).
//   2. Compute the CPU reference (reference_cpu.cpp)         -> trusted MI matrix.
//   3. Compute the GPU raw MI matrix (kernels.cu)            -> the thing taught.
//   4. VERIFY: assert the GPU MI matrix agrees with the CPU one within tolerance.
//   5. PREDICT + REPORT: apply the shared APC correction to the GPU MI, rank the
//      top column pairs as predicted CONTACTS, and report them deterministically
//      to stdout; timing/detail to stderr.
//
//   STDOUT is kept byte-for-byte deterministic so demo/run_demo can diff it
//   against demo/expected_output.txt. Anything that varies run-to-run (timings)
//   goes to STDERR, which the demo shows but does not diff.
//
//   The committed sample MSA is SYNTHETIC and engineered so a KNOWN set of
//   "contact" column pairs coevolve (data/README.md). A good method must rank
//   those planted pairs at the top -- that is exactly what we report, which both
//   teaches the result and makes it verifiable.
//
// READ THIS FIRST in the code tour, then coevolution.h -> reference_cpu.cpp ->
// kernels.cuh -> kernels.cu. See ../THEORY.md for the "why".
// ===========================================================================
#include <algorithm>   // std::sort, std::stable_sort
#include <cmath>       // std::fabs
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // coevolution_mi_gpu (GPU path), Msa, load_msa
#include "reference_cpu.h"    // coevolution_cpu, apc_correct (CPU baseline + shared APC)
#include "coevolution.h"      // CV_Q (alphabet size, only for reporting)
#include "util/io.hpp"        // util::CpuTimer

// Project identity. Kept in sync with demo/expected_output.txt.
static const char* PROJECT_ID   = "2.25";
static const char* PROJECT_NAME = "Coevolutionary Contact Prediction & MSA Transformer";

// Verification tolerance for the raw MI matrix. Both sides derive MI from EXACT
// integer counts and evaluate the SAME cv_mi_from_counts in a fixed order, so
// the only possible difference is a ~1-ulp discrepancy between the host
// compiler's std::log and nvcc's device log over the O(Q^2) sum. 1e-9 nats is
// orders of magnitude above that yet far below any meaningful MI (PATTERNS.md
// section 4, "~machine precision"). See THEORY.md "How we verify correctness".
static constexpr double MI_TOLERANCE = 1.0e-9;

// How many top-ranked contact pairs to print (deterministic, fixed count).
static constexpr int TOP_K = 8;

// ---------------------------------------------------------------------------
// max_matrix_diff: largest |a[k]-b[k]| over two equal-length matrices. Our
//   headline correctness metric for the GPU-vs-CPU MI comparison. Returns
//   +infinity on a size mismatch so a shape bug can never masquerade as success.
// ---------------------------------------------------------------------------
static double max_matrix_diff(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return 1e300;   // effectively infinity for our test
    double worst = 0.0;
    for (std::size_t k = 0; k < a.size(); ++k) {
        const double d = std::fabs(a[k] - b[k]);
        if (d > worst) worst = d;
    }
    return worst;
}

// A predicted contact: column pair (i, j) with its APC-corrected coevolution
// score. We rank these to produce the contact prediction.
struct Contact {
    int i;          // first column (0-based alignment position)
    int j;          // second column (j > i)
    double score;   // APC-corrected coevolution score (higher = more likely contact)
};

int main(int argc, char** argv) {
    // ---- 1. Load the MSA ---------------------------------------------------
    const std::string path = (argc > 1)
        ? argv[1]
        : "data/sample/coevolution_msa.fasta";
    Msa msa;
    try {
        msa = load_msa(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int L = msa.L;

    // ---- 2. CPU reference (timed): raw MI matrix + APC contact score -------
    std::vector<double> mi_cpu, score_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    coevolution_cpu(msa, mi_cpu, score_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU path: raw MI matrix on the device (kernel timed inside) ----
    std::vector<double> mi_gpu;
    float gpu_kernel_ms = 0.0f;
    coevolution_mi_gpu(msa, mi_gpu, &gpu_kernel_ms);

    // Apply the SHARED APC correction to the GPU's MI (host post-step, exactly
    // like project 11.09 finishes its reduction on the host). Using the same
    // apc_correct() as the CPU guarantees the corrected scores match too.
    std::vector<double> score_gpu;
    apc_correct(mi_gpu, L, score_gpu);

    // ---- 4. Verify the GPU raw MI matrix against the CPU reference ---------
    const double mi_err = max_matrix_diff(mi_cpu, mi_gpu);
    const bool pass = (mi_err <= MI_TOLERANCE);

    // ---- 5. Predict contacts: rank column pairs by APC score (from GPU) ----
    // Collect the strict upper triangle (i < j) so each unordered pair appears
    // once, then sort by descending score. A deterministic tie-break (by i then
    // j) keeps the printed ranking byte-identical every run.
    std::vector<Contact> contacts;
    contacts.reserve(static_cast<std::size_t>(L) * (L - 1) / 2);
    for (int i = 0; i < L; ++i)
        for (int j = i + 1; j < L; ++j)
            contacts.push_back({i, j, score_gpu[static_cast<std::size_t>(i) * L + j]});
    std::stable_sort(contacts.begin(), contacts.end(),
        [](const Contact& A, const Contact& B) {
            if (A.score != B.score) return A.score > B.score;   // higher score first
            if (A.i != B.i) return A.i < B.i;                   // then smaller i
            return A.j < B.j;                                   // then smaller j
        });

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("MSA: %d sequences x %d columns (alphabet Q=%d: 20 aa + gap)\n",
                msa.N, L, CV_Q);
    std::printf("Method: pairwise Mutual Information + Average Product Correction (APC)\n");
    const int k = (TOP_K < static_cast<int>(contacts.size()))
                  ? TOP_K : static_cast<int>(contacts.size());
    std::printf("Top %d predicted contacts (1-based columns, by APC score):\n", k);
    for (int r = 0; r < k; ++r) {
        const Contact& c = contacts[r];
        // 1-based columns are friendlier for biologists reading an alignment.
        std::printf("  #%d  (%2d, %2d)  APC=%.6f  rawMI=%.6f\n",
                    r + 1, c.i + 1, c.j + 1, c.score,
                    mi_gpu[static_cast<std::size_t>(c.i) * L + c.j]);
    }
    std::printf("RESULT: %s (GPU MI matrix matches CPU within tol=1.0e-09 nats)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%d sequences, %d columns)\n",
                 path.c_str(), msa.N, L);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU MI kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- the GPU's edge grows with L "
                         "(there are L(L-1)/2 independent pairs); real MSAs have L~10^2-10^3.\n");
    std::fprintf(stderr, "[verify] max |MI_gpu - MI_cpu| = %.3e nats  (tolerance %.1e)\n",
                 mi_err, MI_TOLERANCE);

    // Exit code feeds the demo's pass/fail gate.
    return pass ? 0 : 1;
}
