// ===========================================================================
// src/main.cu  --  Entry point: load complexes, score on CPU+GPU, verify, report
// ---------------------------------------------------------------------------
// Project 1.15 : Protein-Ligand Binding Affinity Scoring (ML)
//
// THE 5-STEP SHAPE every project in this repo follows:
//   1. Load the problem (a batch of docked protein-ligand poses from data/sample).
//   2. CPU reference  (reference_cpu.cpp) -> trusted per-complex pKd scores.
//   3. GPU scoring    (kernels.cu)        -> the batched 3D-CNN we are teaching.
//   4. VERIFY: GPU agrees with CPU within a documented tolerance.
//   5. REPORT: a deterministic per-complex score table + the rank-1 binder to
//      stdout; timings + the numeric error to stderr.
//
// STDOUT is byte-for-byte deterministic (demo/run_demo diffs it against
// demo/expected_output.txt); run-to-run timings go to STDERR (shown, not diffed).
//
// Code tour: start here, then scoring_core.h (the shared math), kernels.cuh ->
// kernels.cu (GPU), then reference_cpu.* (CPU baseline). The "why" is ../THEORY.md.
// ===========================================================================
#include <algorithm>   // std::max_element
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // score_gpu, ComplexSet
#include "reference_cpu.h"    // load_complexes, score_cpu
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "1.15";
static const char* PROJECT_NAME = "Protein-Ligand Binding Affinity Scoring (ML)";

// ---------------------------------------------------------------------------
// Verification tolerance (PATTERNS.md sec.4 -- "be honest about floating point").
//   The CPU and GPU run the SAME double-precision per-element math (scoring_core.h),
//   but the global-average POOL is summed in different ORDERS: the CPU adds the
//   4096 voxel responses left-to-right, while the GPU uses a shared-memory tree
//   reduction. Floating-point addition is not associative, so the two pooled sums
//   differ by a few ULPs (~1e-12 in double). That tiny difference flows through
//   the dense layer and the logistic squash to the pKd. We therefore verify to a
//   GENEROUS 1e-6 absolute tolerance on pKd -- ~6 orders of magnitude above the
//   actual ~1e-12 disagreement, and far below the 1e-4 precision we report. This
//   is a real, teachable effect, not a bug (THEORY "How we verify correctness").
// ---------------------------------------------------------------------------
static constexpr double TOLERANCE = 1.0e-6;

// max_abs_err over two double vectors (the float version lives in util/io.hpp;
// our scores are double, so we compute it inline here).
static double max_abs_err(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return 1e300;            // shape bug -> never "agree"
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs(a[i] - b[i]);
        if (d > worst) worst = d;
    }
    return worst;
}

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/complexes_sample.txt";
    ComplexSet cs;
    try {
        cs = load_complexes(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<double> score_cpu_v;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    score_cpu(cs, score_cpu_v);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU scoring (kernel timed inside the wrapper) -----------------
    std::vector<double> score_gpu_v;
    float gpu_kernel_ms = 0.0f;
    score_gpu(cs, score_gpu_v, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    const double err  = max_abs_err(score_cpu_v, score_gpu_v);
    const bool   pass = err <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    // The rank-1 binder = the complex the model scores highest (ties -> lower
    // index, via the strict-greater comparison in max_element's default).
    int best = 0;
    for (int i = 1; i < cs.n; ++i)
        if (score_gpu_v[i] > score_gpu_v[best]) best = i;

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("3D-CNN scorer: %d complexes, %d^3 grid, %d in-ch, %d conv filters\n",
                cs.n, GRID, CIN, COUT);
    std::printf("predicted binding affinity (pKd) per complex:\n");
    for (int i = 0; i < cs.n; ++i) {
        // Print GPU score (verified == CPU within tol) and the synthetic label so
        // the learner can see the model is at least responding to the input. The
        // label is NOT used in scoring; it is the value the pose was built to have.
        std::printf("  complex %2d  atoms=%3d  pred_pKd=%7.4f  (synthetic_label=%6.3f)\n",
                    i, cs.atom_count(i), score_gpu_v[i], cs.label[i]);
    }
    std::printf("rank-1 predicted binder: complex %d  (pred_pKd=%.4f)\n",
                best, score_gpu_v[best]);
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-06)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (n=%d complexes, %zu atoms total)\n",
                 path.c_str(), cs.n, cs.atoms.size());
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny batch is dominated by "
                         "launch/copy overhead; the GPU wins when rescoring millions of poses.\n");
    std::fprintf(stderr, "[verify] max_abs_err = %.3e pKd  (tolerance %.1e; CPU vs GPU differ only "
                         "in pooling reduction order)\n", err, TOLERANCE);

    return pass ? 0 : 1;
}
