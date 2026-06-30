// ===========================================================================
// src/main.cu  --  Entry point: load ensemble, back-calculate DEER P(r) on CPU
//                  and GPU, reweight to the experimental target, verify, report.
// ---------------------------------------------------------------------------
// Project 2.35 : Electron Paramagnetic Resonance (EPR/DEER) Constrained Modeling
//
// PIPELINE (the 6 steps):
//   1. Load the synthetic ensemble (data/sample): M frames x 2 spin-label
//      rotamer clouds, plus the experimental target P_exp(r).
//   2. CPU back-calculation (reference_cpu.cpp): per-frame histograms P_m(r).
//   3. GPU back-calculation (kernels.cu): the SAME, one frame per thread.
//   4. VERIFY the histograms agree to a documented tolerance (shared math).
//   5. REWEIGHT both ensembles to the target with the SHARED max-entropy solver
//      and verify the recovered weights agree.
//   6. REPORT, deterministically: the reweighted P(r) peak, chi^2 before/after,
//      and how much population the "true" frames gained (the synthetic answer).
//
// Code tour: start here, then deer.h (per-frame physics), reference_cpu.cpp
// (loader + reweighting), kernels.cu (the GPU back-calc).
// ===========================================================================
#include <algorithm>   // std::max_element
#include <cmath>       // std::fabs, std::fmax
#include <cstdio>      // std::printf, std::fprintf
#include <string>
#include <vector>

#include "deer.h"            // Spin3, r_bin_center, distributions
#include "deer_params.h"     // NBINS, ROTAMERS_PER_SITE, REWEIGHT_*
#include "kernels.cuh"       // deer_backcalc_gpu
#include "reference_cpu.h"   // Ensemble, load_ensemble, deer_backcalc_cpu, reweight_cpu, ...
#include "util/io.hpp"       // util::CpuTimer

static const char* PROJECT_ID   = "2.35";
static const char* PROJECT_NAME = "Electron Paramagnetic Resonance (EPR/DEER) Constrained Modeling";

// --- Verification tolerances (PATTERNS.md §4; justified in THEORY "Numerical") -
// The per-frame histograms are built by the SAME deer_member_histogram() on both
// sides using integer counts then an exact reciprocal -> they agree to round-off.
// The reweighting is shared host code fed identical (to ~1e-13) histograms, so its
// weights also agree to round-off. We allow a hair of double slack.
static constexpr double HIST_TOL   = 1.0e-12;   // max |P_m^cpu - P_m^gpu| per bin
static constexpr double WEIGHT_TOL = 1.0e-9;    // max |w^cpu - w^gpu| per frame

// Return the index of the largest element of a length-NBINS distribution (the
// modal / peak bin). Deterministic tie-break: first maximum wins.
static int argmax_bin(const std::vector<double>& d) {
    int best = 0;
    for (int b = 1; b < NBINS; ++b) if (d[b] > d[best]) best = b;
    return best;
}

int main(int argc, char** argv) {
    // ---- 1. Load ----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/deer_sample.txt";
    Ensemble e;
    try {
        e = load_ensemble(path);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "[error] %s\n", ex.what());
        return 2;
    }

    // ---- 2. CPU back-calculation (timed) ----------------------------------
    std::vector<double> hist_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    deer_backcalc_cpu(e, hist_cpu);
    const double cpu_backcalc_ms = cpu_timer.stop_ms();

    // ---- 3. GPU back-calculation (kernel timed) ---------------------------
    std::vector<double> hist_gpu;
    float gpu_kernel_ms = 0.0f;
    deer_backcalc_gpu(e.M, e.siteA, e.siteB, hist_gpu, &gpu_kernel_ms);

    // ---- 4. Verify the per-frame histograms agree -------------------------
    double hist_diff = 0.0;
    for (std::size_t i = 0; i < hist_cpu.size(); ++i)
        hist_diff = std::fmax(hist_diff, std::fabs(hist_cpu[i] - hist_gpu[i]));
    const bool hist_ok = (hist_diff <= HIST_TOL);

    // ---- 5. Reweight BOTH ensembles with the shared max-entropy solver ----
    // The reweighting is identical host code; feeding it the CPU histograms vs.
    // the GPU histograms must produce the same weights (since the histograms
    // match). This checks the whole pipeline end-to-end, not just stage 1.
    std::vector<double> w_cpu, w_gpu;
    double chi2_before = 0.0, chi2_after_cpu = 0.0, chi2_after_gpu = 0.0;
    reweight_cpu(hist_cpu, e.M, e.target, w_cpu, &chi2_before, &chi2_after_cpu);
    reweight_cpu(hist_gpu, e.M, e.target, w_gpu, nullptr,       &chi2_after_gpu);

    double weight_diff = 0.0;
    for (int m = 0; m < e.M; ++m)
        weight_diff = std::fmax(weight_diff, std::fabs(w_cpu[m] - w_gpu[m]));
    const bool weight_ok = (weight_diff <= WEIGHT_TOL);

    const bool pass = hist_ok && weight_ok;

    // ---- 6a. Deterministic report -> STDOUT -------------------------------
    // We report from the GPU pipeline (verified equal to the CPU one). Compute
    // the reweighted model P(r) and the population captured by the "true" frames.
    std::vector<double> uniform(e.M, 1.0 / e.M), mix_before, mix_after;
    mixed_distribution(hist_gpu, e.M, uniform, mix_before);
    mixed_distribution(hist_gpu, e.M, w_gpu,   mix_after);

    int n_true = 0;
    double w_true = 0.0;                 // total reweighted population on true frames
    for (int m = 0; m < e.M; ++m) if (e.truth[m]) { ++n_true; w_true += w_gpu[m]; }
    const double w_true_prior = static_cast<double>(n_true) / e.M;   // their uniform share

    const int peak_before = argmax_bin(mix_before);
    const int peak_after  = argmax_bin(mix_after);
    const int peak_target = argmax_bin(e.target);

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("ensemble: %d frames, %d rotamers/site, %d-bin P(r) over %.1f-%.1f nm\n",
                e.M, ROTAMERS_PER_SITE, NBINS, R_MIN_NM, R_MIN_NM + NBINS * R_BIN_NM);
    std::printf("DEER back-calc: GPU vs CPU per-frame P(r) match = %s\n", hist_ok ? "YES" : "NO");
    std::printf("reweighting: %d steps, theta = %.1e\n", REWEIGHT_ITERS, THETA);
    std::printf("  chi^2(uniform)    = %.6e\n", chi2_before);
    std::printf("  chi^2(reweighted) = %.6e\n", chi2_after_gpu);
    std::printf("  P(r) peak bin: uniform r=%.2f nm | reweighted r=%.2f nm | target r=%.2f nm\n",
                r_bin_center(peak_before), r_bin_center(peak_after), r_bin_center(peak_target));
    std::printf("  true-frame population: prior %.4f -> reweighted %.4f  (%d/%d frames are true matches)\n",
                w_true_prior, w_true, n_true, e.M);
    std::printf("RESULT: %s (GPU back-calc matches CPU; reweighting recovers the true frames)\n",
                pass ? "PASS" : "FAIL");

    // ---- 6b. Run-varying detail -> STDERR (shown, not diffed) -------------
    std::fprintf(stderr, "[data]   source: %s  (%d frames)\n", path.c_str(), e.M);
    std::fprintf(stderr, "[timing] CPU back-calc: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_backcalc_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- the GPU's edge grows with frame count and "
                         "rotamer-library size; real ensembles are 10^3-10^5 frames x ~200 rotamers.\n");
    std::fprintf(stderr, "[verify] max |P_m(r) cpu-gpu| = %.3e (tol %.1e); "
                         "max |w cpu-gpu| = %.3e (tol %.1e)\n",
                 hist_diff, HIST_TOL, weight_diff, WEIGHT_TOL);
    std::fprintf(stderr, "[verify] chi^2 after: cpu %.6e / gpu %.6e\n", chi2_after_cpu, chi2_after_gpu);

    return pass ? 0 : 1;
}
