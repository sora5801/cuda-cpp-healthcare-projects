// ===========================================================================
// src/main.cu  --  Entry point: load guide+genome, scan CPU+GPU, verify, report
// ---------------------------------------------------------------------------
// Project 3.17 : CRISPR Guide Design & Off-Target Scoring
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the problem (one 20-nt guide + a reference genome from data/sample).
//   2. CPU reference scan (reference_cpu.cpp)              -> trusted answer.
//   3. GPU scan           (kernels.cu)                     -> the thing taught.
//   4. VERIFY: GPU per-window outputs == CPU within tolerance -> correctness.
//   5. REPORT: deterministic on-/off-target summary to stdout; timing to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Anything that varies run-to-run (timings) goes to
//   STDERR (shown, not diffed). The aggregate reductions (top-K, summed CFD)
//   are done on the HOST in a FIXED order so the summary is reproducible -- the
//   GPU does the per-window scoring, not the order-dependent reduction
//   (PATTERNS.md §3: float atomics would be nondeterministic).
//
// Code tour: read this first, then kernels.cuh -> kernels.cu, then cfd_score.h
// (the shared scorer) and reference_cpu.cpp (the baseline). See ../THEORY.md.
// ===========================================================================
#include <algorithm>   // std::sort, std::min
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // scan_gpu (GPU path), CrisprProblem, ScanResult
#include "reference_cpu.h"    // load_problem, scan_cpu, specificity_score, decode_base
#include "util/io.hpp"        // util::CpuTimer

// Program identity. Must stay in sync with demo/expected_output.txt.
static const char* PROJECT_ID   = "3.17";
static const char* PROJECT_NAME = "CRISPR Guide Design & Off-Target Scoring";

// How many top off-target sites to list (deterministic ranking below).
static constexpr int TOP_K = 5;

// --- Verification tolerances ------------------------------------------------
// Mismatch counts are INTEGERS computed by identical logic on both sides, so we
// require EXACT equality (any difference is a bug). CFD scores are short
// double-precision PRODUCTS evaluated in an identical fixed order with no
// adjacent add (so no FMA contraction can diverge between host and device);
// they too come out bit-identical in practice. We still verify to a tiny
// physical tolerance and SAY SO rather than claiming bit-exactness blindly
// (PATTERNS.md §4). 1e-12 is ~1e4 ulps below the [0,1] score range -- far below
// any biological meaning.
static constexpr double CFD_TOLERANCE = 1.0e-12;

// A site counts as a reportable off-target if it has a valid PAM, at least one
// mismatch, and a CFD above this floor (negligible-cutting sites are noise).
static constexpr double OFFTARGET_FLOOR = 1.0e-6;

// ---------------------------------------------------------------------------
// max_abs_diff_double: largest |a[i]-b[i]| over two equal-length double arrays.
// (util::max_abs_err is float-only; CFD is double, so we have a local helper.)
// Returns +inf on a length mismatch so a shape bug can't masquerade as success.
// ---------------------------------------------------------------------------
static double max_abs_diff_double(const std::vector<double>& a,
                                  const std::vector<double>& b) {
    if (a.size() != b.size()) return 1e300;   // "infinite": lengths disagree
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        if (d < 0) d = -d;
        if (d > worst) worst = d;
    }
    return worst;
}

// count_int_mismatches: how many positions differ between two int arrays. Used
// to verify the GPU mismatch-count array exactly matches the CPU's.
static int count_int_mismatches(const std::vector<int>& a, const std::vector<int>& b) {
    if (a.size() != b.size()) return -1;       // length disagreement
    int diff = 0;
    for (std::size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) ++diff;
    return diff;
}

// ---------------------------------------------------------------------------
// A reportable site: its genome start position, mismatch count, and CFD score.
// We sort these for the deterministic top-K listing.
// ---------------------------------------------------------------------------
struct Site {
    int    pos;
    int    mismatches;
    double cfd;
};

// protospacer_string: render the 20 protospacer bases at genome position `pos`
// as an ACGT string (PAM appended in parentheses) for a human-readable report.
static std::string protospacer_string(const CrisprProblem& prob, int pos) {
    std::string s;
    s.reserve(WINDOW_LEN + 2);
    for (int k = 0; k < GUIDE_LEN; ++k) s += decode_base(prob.genome[static_cast<std::size_t>(pos) + k]);
    s += '(';
    for (int k = 0; k < PAM_LEN; ++k)   s += decode_base(prob.genome[static_cast<std::size_t>(pos) + GUIDE_LEN + k]);
    s += ')';
    return s;
}

int main(int argc, char** argv) {
    // ---- 1. Load the problem ------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/guide_genome_sample.txt";
    CrisprProblem prob;
    try {
        prob = load_problem(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference scan (timed) -------------------------------------
    ScanResult res_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    scan_cpu(prob, res_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU scan (kernel timed inside the wrapper) ---------------------
    ScanResult res_gpu;
    float gpu_kernel_ms = 0.0f;
    scan_gpu(prob, res_gpu, &gpu_kernel_ms);

    // ---- 4. Verify GPU == CPU ----------------------------------------------
    const int    mm_diffs = count_int_mismatches(res_cpu.mismatches, res_gpu.mismatches);
    const double cfd_err  = max_abs_diff_double(res_cpu.cfd, res_gpu.cfd);
    const bool   pass     = (mm_diffs == 0) && (cfd_err <= CFD_TOLERANCE);

    // ---- 5. Aggregate on the HOST in a FIXED order (deterministic) ---------
    // Walk windows left-to-right. Classify each by mismatch count + CFD:
    //   * a valid-PAM, zero-mismatch site is an ON-TARGET hit (cfd == 1.0);
    //   * a valid-PAM, >=1-mismatch site above the floor is an OFF-TARGET.
    // We use the CPU array (already verified to equal the GPU array). Summing in
    // index order on a single thread is order-deterministic -> reproducible
    // specificity. (THEORY §"verify" explains why we do not reduce on the GPU.)
    int                on_target_count = 0;
    int                first_on_target = -1;
    double             sum_offtarget_cfd = 0.0;
    std::vector<Site>  offtargets;
    for (int i = 0; i < prob.n_windows; ++i) {
        const int mm = res_cpu.mismatches[static_cast<std::size_t>(i)];
        if (mm < 0) continue;                         // no PAM here -> not a site
        const double cfd = res_cpu.cfd[static_cast<std::size_t>(i)];
        if (mm == 0) {                                // perfect match = on-target
            ++on_target_count;
            if (first_on_target < 0) first_on_target = i;
        } else if (cfd >= OFFTARGET_FLOOR) {          // imperfect but cutting
            sum_offtarget_cfd += cfd;
            offtargets.push_back({i, mm, cfd});
        }
    }

    // Rank off-targets by CFD descending, ties broken by lower position, so the
    // printed top-K is fully deterministic regardless of discovery order.
    std::sort(offtargets.begin(), offtargets.end(), [](const Site& a, const Site& b) {
        if (a.cfd != b.cfd) return a.cfd > b.cfd;     // higher score first
        return a.pos < b.pos;                          // tie -> earlier position
    });
    const double specificity = specificity_score(sum_offtarget_cfd);
    const int    k = std::min<int>(TOP_K, static_cast<int>(offtargets.size()));

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("guide '%s' vs genome of %d bases (%d candidate windows, PAM=NGG)\n",
                prob.guide_name.c_str(), prob.genome_len, prob.n_windows);
    if (first_on_target >= 0)
        std::printf("on-target: %d site(s); first at position %d  %s\n",
                    on_target_count, first_on_target,
                    protospacer_string(prob, first_on_target).c_str());
    else
        std::printf("on-target: 0 sites (no exact PAM-flanked match found)\n");
    std::printf("off-target sites (>=1 mismatch, CFD>=%.0e): %d\n",
                OFFTARGET_FLOOR, static_cast<int>(offtargets.size()));
    std::printf("top-%d off-targets by CFD:\n", k);
    for (int r = 0; r < k; ++r) {
        const Site& s = offtargets[static_cast<std::size_t>(r)];
        std::printf("  #%d  pos %3d  mm %d  CFD %.6f  %s\n",
                    r + 1, s.pos, s.mismatches, s.cfd, protospacer_string(prob, s.pos).c_str());
    }
    std::printf("aggregate off-target CFD sum = %.6f\n", sum_offtarget_cfd);
    std::printf("guide specificity score = %.4f  (100 = perfectly specific)\n", specificity);
    std::printf("RESULT: %s (GPU matches CPU: %d mismatch-count diffs, CFD err <= %.0e)\n",
                pass ? "PASS" : "FAIL", mm_diffs, CFD_TOLERANCE);

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (genome_len=%d, windows=%d)\n",
                 path.c_str(), prob.genome_len, prob.n_windows);
    std::fprintf(stderr, "[timing] CPU scan: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny genome is dominated by "
                         "launch/copy overhead; the GPU wins at chromosome scale (10^8 windows).\n");
    std::fprintf(stderr, "[verify] mismatch-count diffs = %d  CFD max_abs_err = %.3e  (tol %.1e)\n",
                 mm_diffs, cfd_err, CFD_TOLERANCE);

    return pass ? 0 : 1;
}
