// ===========================================================================
// src/main.cu  --  Entry point: forward-model SAXS, verify, fit, report
// ---------------------------------------------------------------------------
// Project 2.24 : SAXS / SANS Data-Driven Structure Modeling
//
// 5-step shape (the shape every project in this repo follows):
//   1. Load the atomic model + experimental curve (data/sample, or a built-in
//      synthetic fallback so the program always runs).
//   2. CPU reference: forward-model I(q) via the Debye formula (reference_cpu).
//   3. GPU result: the same Debye profile, one thread per q (kernels.cu).
//   4. VERIFY: GPU intensities match the CPU within a documented tolerance.
//   5. REPORT (deterministic -> STDOUT): the normalized profile at a few q's,
//      the recovered Guinier Rg vs the synthetic true Rg, and the reduced
//      chi-square of the model-vs-experiment fit. Timing -> STDERR.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Run-varying numbers (timings) go to STDERR.
//
// Code tour: start here, then saxs_core.h -> kernels.cuh -> kernels.cu, then
//   reference_cpu.cpp. See ../THEORY.md for the science and GPU mapping.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // debye_gpu, SaxsModel
#include "reference_cpu.h"    // load_model, debye_profile_cpu, best_scale, ...
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "2.24";
static const char* PROJECT_NAME = "SAXS / SANS Data-Driven Structure Modeling";

// Verification tolerance: GPU vs CPU MAX RELATIVE error on I(q).
//   The GPU kernel and the CPU reference call the IDENTICAL per-q routine
//   (saxs_core.h), so they execute the same operations in the same order -> the
//   only possible difference is whether the host compiler and nvcc contract a
//   multiply-add into an FMA differently. That is bounded by a few ULP per term,
//   and over an O(N^2) double-precision sum stays far below 1e-9 relative. We set
//   a generous-but-honest 1e-9 (PATTERNS.md §4: ~machine precision for short
//   double-precision computations). Relative, because I(q) spans many decades.
static constexpr double TOLERANCE = 1.0e-9;

// Number of low-q points used for the Guinier Rg linear fit (q*Rg < ~1.3 region).
static constexpr int GUINIER_POINTS = 6;

// max relative error between two equal-length double arrays (|a-b|/max(|a|,eps)).
static double max_rel_err(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return 1.0e300;   // shape mismatch -> "infinitely wrong"
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double denom = std::fabs(a[i]) > 1.0e-300 ? std::fabs(a[i]) : 1.0e-300;
        const double rel = std::fabs(a[i] - b[i]) / denom;
        if (rel > worst) worst = rel;
    }
    return worst;
}

// Build a built-in synthetic model when no sample file is supplied: a tiny
// "dumbbell" of two point clusters so the program runs with zero arguments.
// NOTE: this fallback exists only so the binary never hard-fails; the committed
// data/sample (a small synthetic globular blob) is the intended demo input and
// is what demo/expected_output.txt was captured from. We keep this fallback
// deliberately DIFFERENT and tiny so nobody mistakes it for the demo data.
static SaxsModel make_synthetic_fallback() {
    SaxsModel m;
    m.n_atoms = 2;
    m.x = {0.0, 20.0}; m.y = {0.0, 0.0}; m.z = {0.0, 0.0}; m.f = {8.0, 8.0};
    m.true_rg = 10.0;                          // two unit masses ±10 Å -> Rg=10 Å
    m.n_q = 4;
    m.q = {0.01, 0.05, 0.10, 0.20};
    // Placeholder "experiment" = exact model (chi^2 ~ 0); sigmas are 1% of I(0).
    m.I_exp.assign(m.n_q, 0.0); m.sigma.assign(m.n_q, 1.0);
    return m;
}

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/saxs_sample.txt";
    SaxsModel m;
    const char* source = path.c_str();
    try {
        m = load_model(path);
    } catch (const std::exception& e) {
        // Fall back to the built-in tiny model so the program still demonstrates
        // the pipeline (the real demo always passes the sample path).
        std::fprintf(stderr, "[warn] could not load '%s' (%s); using built-in fallback.\n",
                     path.c_str(), e.what());
        m = make_synthetic_fallback();
        source = "synthetic (built-in fallback)";
    }

    // ---- 2. CPU reference profile (timed) ---------------------------------
    std::vector<double> I_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    debye_profile_cpu(m, I_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU profile (kernel timed inside the wrapper) -----------------
    std::vector<double> I_gpu;
    float gpu_kernel_ms = 0.0f;
    debye_gpu(m, I_gpu, &gpu_kernel_ms);

    // ---- 4. Verify GPU vs CPU ---------------------------------------------
    const double err = max_rel_err(I_cpu, I_gpu);
    const bool pass = err <= TOLERANCE;

    // ---- analysis (uses the GPU profile; CPU is identical within tol) ------
    const double I0 = I_gpu.empty() ? 1.0 : I_gpu[0];              // I(q[0]) ~ I(0)
    const double c   = best_scale(I_gpu, m.I_exp, m.sigma);        // model->exp scale
    const double chi2 = reduced_chi_square(I_gpu, c, m.I_exp, m.sigma);
    const double rg  = guinier_rg(m.q, I_gpu, GUINIER_POINTS);     // recovered size

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) ----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("Debye forward model: %d atoms, %d q-points (1 GPU thread per q)\n",
                m.n_atoms, m.n_q);
    std::printf("normalized profile I(q)/I(0):\n");
    // Print the curve at a handful of representative q's (first, some middle,
    // last) at fixed precision so stdout is identical every run.
    const int idx[5] = {
        0,
        m.n_q > 1 ? m.n_q / 4 : 0,
        m.n_q > 1 ? m.n_q / 2 : 0,
        m.n_q > 1 ? (3 * m.n_q) / 4 : 0,
        m.n_q - 1
    };
    for (int t = 0; t < 5; ++t) {
        const int k = idx[t];
        std::printf("  q=%.4f 1/A   I/I0=%.6f\n", m.q[k], I_gpu[k] / I0);
    }
    std::printf("Guinier Rg (from %d low-q pts) = %.3f A", GUINIER_POINTS, rg);
    if (m.true_rg > 0.0) std::printf("   (synthetic true Rg = %.3f A)", m.true_rg);
    std::printf("\n");
    std::printf("model-vs-experiment fit: scale=%.6f  reduced chi^2=%.6f\n", c, chi2);
    std::printf("RESULT: %s (GPU matches CPU within rel tol=1.0e-09)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) -----------------
    std::fprintf(stderr, "[data]   source: %s  (%d atoms, %d q-points)\n",
                 source, m.n_atoms, m.n_q);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernel: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- the GPU's edge over the CPU grows "
                         "with n_atoms^2 * n_q; this sample is tiny.\n");
    std::fprintf(stderr, "[verify] max_rel_err = %.3e  (tolerance %.1e)\n", err, TOLERANCE);

    // Exit code feeds the demo's pass/fail gate.
    return pass ? 0 : 1;
}
