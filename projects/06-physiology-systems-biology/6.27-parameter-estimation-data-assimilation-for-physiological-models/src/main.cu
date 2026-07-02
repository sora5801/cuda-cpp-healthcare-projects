// ===========================================================================
// src/main.cu  --  Entry point: assimilate a pressure waveform, verify, report
// ---------------------------------------------------------------------------
// Project 6.27 : Parameter Estimation & Data Assimilation for Physiological Models
//
// 5-step shape (every project in this repo follows it):
//   1. Load the EnKF config (data/sample) and synthesize the noisy observations
//      from a KNOWN true patient (a "twin experiment").
//   2. CPU reference: run the whole Ensemble Kalman Filter serially.
//   3. GPU: run the SAME filter with the ensemble forecast on the device.
//   4. VERIFY: the two final ensembles match member-for-member (same integrator +
//      same shared analysis + same seeds -> round-off agreement).
//   5. REPORT: deterministic estimate + recovery accuracy -> stdout; timing -> stderr.
//
//   STDOUT is byte-deterministic so demo/run_demo can diff it; timings and other
//   run-varying numbers go to STDERR (shown, not diffed) -- docs/PATTERNS.md §3.
//
// Code tour: start here, then windkessel.h (the model + RK4), reference_cpu.cpp
// (observations, ensemble, the EnKF analysis), kernels.cu (the GPU forecast).
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // run_enkf_gpu, forecast_gpu
#include "reference_cpu.h"    // load_config, synthesize_observations, run_enkf_cpu
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "6.27";
static const char* PROJECT_NAME = "Parameter Estimation & Data Assimilation for Physiological Models";

// Verification tolerance. CPU and GPU run the SAME double-precision RK4 forecast
// (shared windkessel.h) and the SAME host analysis with identical seeds, so the
// per-member states differ only by floating-point re-association across the many
// windows -- a tiny physical tolerance covers it (docs/PATTERNS.md §4). We report
// the actual worst diff on stderr so the honesty is visible.
static constexpr double TOLERANCE = 1.0e-6;

int main(int argc, char** argv) {
    // ---- 1. Load config + synthesize the "measured" observations -----------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/enkf_config.txt";
    EnKFConfig c;
    try {
        c = load_config(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const std::vector<double> obs = synthesize_observations(c);

    // ---- 2. CPU reference EnKF (timed) -------------------------------------
    std::vector<double> ens_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    const EnKFResult cpu = run_enkf_cpu(c, obs, ens_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU EnKF (device forecast; kernel time summed inside) ----------
    std::vector<double> ens_gpu;
    float gpu_forecast_ms = 0.0f;
    const EnKFResult gpu = run_enkf_gpu(c, obs, ens_gpu, &gpu_forecast_ms);

    // ---- 4. Verify: final ensembles agree member-for-member ----------------
    double worst = 0.0;
    for (std::size_t i = 0; i < ens_cpu.size(); ++i)
        worst = std::fmax(worst, std::fabs(ens_cpu[i] - ens_gpu[i]));
    const bool pass = (ens_cpu.size() == ens_gpu.size()) && (worst <= TOLERANCE);

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    // Relative recovery error of the estimate against the KNOWN truth -- this is
    // the science check (did data assimilation actually recover the parameters?),
    // stronger than mere CPU==GPU agreement (docs/PATTERNS.md §4).
    const double R_err = 100.0 * std::fabs(gpu.R_hat - c.R_true) / c.R_true;
    const double C_err = 100.0 * std::fabs(gpu.C_hat - c.C_true) / c.C_true;
    // Prior error (before assimilation) so the reader sees how far the filter moved.
    const double R_prior_err = 100.0 * std::fabs(c.R_prior - c.R_true) / c.R_true;
    const double C_prior_err = 100.0 * std::fabs(c.C_prior - c.C_true) / c.C_true;

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("two-element Windkessel; EnKF joint state-parameter estimation\n");
    std::printf("ensemble=%d  windows=%d  window=%.3fs (%d x dt=%.3fs)  obs_noise=%.1f mmHg\n",
                c.m, c.n_obs, enkf_window_len(c), c.substeps, c.dt, c.obs_noise);
    std::printf("true    : R=%.4f mmHg*s/mL   C=%.4f mL/mmHg\n", c.R_true, c.C_true);
    std::printf("prior   : R=%.4f (%.1f%% off)   C=%.4f (%.1f%% off)\n",
                c.R_prior, R_prior_err, c.C_prior, C_prior_err);
    std::printf("estimate: R=%.4f (%.2f%% err)   C=%.4f (%.2f%% err)\n",
                gpu.R_hat, R_err, gpu.C_hat, C_err);
    std::printf("posterior spread: R_std=%.4f   C_std=%.4f\n", gpu.R_std, gpu.C_std);
    std::printf("final ensemble-mean pressure RMSE vs obs = %.4f mmHg\n", gpu.final_rmse);
    std::printf("RESULT: %s (GPU EnKF matches CPU within tol=1.0e-06)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s\n", path.c_str());
    std::fprintf(stderr, "[cpu]    estimate R=%.6f C=%.6f (matches GPU headline to printed digits)\n",
                 cpu.R_hat, cpu.C_hat);
    std::fprintf(stderr, "[timing] CPU total: %.3f ms   GPU forecast (summed over %d windows): %.3f ms\n",
                 cpu_ms, c.n_obs, gpu_forecast_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- tiny ensembles are launch/PCIe-bound; the GPU "
                         "forecast wins as ensemble size and per-member cost grow.\n");
    std::fprintf(stderr, "[verify] worst per-member state diff = %.3e  (tolerance %.1e)\n", worst, TOLERANCE);

    return pass ? 0 : 1;
}
