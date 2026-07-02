// ===========================================================================
// src/reference_cpu.h  --  EnKF config, deterministic RNG, and the CPU reference
// ---------------------------------------------------------------------------
// Project 6.27 : Parameter Estimation & Data Assimilation for Physiological Models
//
// This header declares everything the *host* side needs and that kernels.cu also
// reuses (the config struct + the shared forecast in windkessel.h). It is pure
// C++ (no CUDA), so both nvcc and the plain host compiler can include it.
//
// THE PIPELINE (one EnKF assimilation cycle per observation):
//   1. FORECAST : advance every ensemble member's state through one window by
//                 integrating the Windkessel ODE (windkessel.h). This is the
//                 GPU-parallel bottleneck (one thread per member).
//   2. ANALYSIS : when the window ends we have a pressure observation P_obs. Pull
//                 every member toward it with the ENSEMBLE KALMAN GAIN computed
//                 from the ensemble's own sample covariance. This is cheap dense
//                 linear algebra over a 3-vector state, done on the HOST (see
//                 THEORY §4 for why host-side is the right teaching choice).
//   The estimate after all windows is the ensemble-mean (R, C).
//
// READ THIS AFTER: windkessel.h ; BEFORE: reference_cpu.cpp, kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "windkessel.h"   // WK_NSTATE, wk_forecast_member, wk_inflow (shared HD core)

// ---------------------------------------------------------------------------
// EnKFConfig: the complete, self-contained specification of an assimilation run.
//   Loaded from the sample text file (see data/README.md for the field order).
//   Everything is deterministic given this struct + the fixed seed, so the demo
//   output is byte-reproducible (docs/PATTERNS.md §3).
// ---------------------------------------------------------------------------
struct EnKFConfig {
    // --- Ensemble & assimilation schedule ---
    int    m = 0;             // ensemble size (number of members, e.g. 256)
    int    n_obs = 0;         // number of pressure observations to assimilate
    double dt = 0.0;          // RK4 sub-step (s)
    int    substeps = 0;      // RK4 sub-steps per observation window (window = substeps*dt)

    // --- Inflow waveform Q(t) (known input, shared by all members) ---
    double T = 0.0;           // cardiac cycle length (s)
    double t_sys = 0.0;       // systolic ejection duration (s)
    double Q_peak = 0.0;      // peak inflow (mL/s)

    // --- TRUE patient parameters (used only to synthesize the observations) ---
    double R_true = 0.0;      // peripheral resistance (mmHg*s/mL)
    double C_true = 0.0;      // arterial compliance (mL/mmHg)
    double P0 = 0.0;          // initial aortic pressure (mmHg)

    // --- Noise level ---
    double obs_noise = 0.0;   // std dev of measurement noise on P_obs (mmHg)

    // --- Prior on the parameters (initial ensemble spread, in log space) ---
    double R_prior = 0.0;     // prior mean guess for R (mmHg*s/mL)
    double C_prior = 0.0;     // prior mean guess for C (mL/mmHg)
    double logR_std = 0.0;    // prior std dev of log R (dimensionless)
    double logC_std = 0.0;    // prior std dev of log C (dimensionless)

    uint64_t seed = 0;        // RNG seed -> fully reproducible ensemble & noise
};

// Convenience accessors (documented so the intent is explicit at call sites).
inline int enkf_state_dim() { return WK_NSTATE; }              // 3: [P, logR, logC]
inline double enkf_window_len(const EnKFConfig& c) { return c.substeps * c.dt; }

// ---------------------------------------------------------------------------
// SplitMix64: a tiny, deterministic, well-distributed 64-bit PRNG.
//   We use our OWN generator (not <random>) for one reason: DETERMINISM ACROSS
//   PLATFORMS. std::mt19937 is portable but the DISTRIBUTIONS (normal_distribution)
//   are NOT specified bit-for-bit by the standard, so results could differ between
//   compilers. SplitMix64 + a hand-rolled Box-Muller give identical bits
//   everywhere, which is what keeps expected_output.txt stable (PATTERNS.md §3).
//   It is also trivially seed-per-member, so each ensemble member draws an
//   independent, reproducible stream.
// ---------------------------------------------------------------------------
struct SplitMix64 {
    uint64_t state;
    explicit SplitMix64(uint64_t s) : state(s) {}
    // One step of SplitMix64 (Vigna 2015): mix the counter, avalanche the bits.
    uint64_t next_u64() {
        uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    // Uniform double in [0,1): take the top 53 bits (a double's mantissa width).
    double next_uniform() {
        return (next_u64() >> 11) * (1.0 / 9007199254740992.0);   // / 2^53
    }
    // Standard normal via Box-Muller (one of the pair; simple and deterministic).
    double next_normal() {
        double u1 = next_uniform();
        double u2 = next_uniform();
        // Guard the log against u1 == 0 (would give -inf).
        if (u1 < 1e-300) u1 = 1e-300;
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.28318530717958647692 * u2);
    }
};

// ---------------------------------------------------------------------------
// EnKFResult: the headline result of a run -- the recovered patient parameters.
// ---------------------------------------------------------------------------
struct EnKFResult {
    double R_hat = 0.0;       // estimated peripheral resistance (ensemble mean)
    double C_hat = 0.0;       // estimated arterial compliance (ensemble mean)
    double R_std = 0.0;       // posterior spread (ensemble std) of R
    double C_std = 0.0;       // posterior spread (ensemble std) of C
    double final_rmse = 0.0;  // RMSE of the final ensemble-mean pressure vs obs (mmHg)
};

// Load an EnKFConfig from the whitespace-separated text sample.
EnKFConfig load_config(const std::string& path);

// Synthesize the TRUE pressure waveform + noisy observations from the config.
//   Returns the n_obs observed pressures (mmHg) at the end of each window. The
//   noise is drawn from SplitMix64(seed) so the observation vector is fixed.
std::vector<double> synthesize_observations(const EnKFConfig& c);

// Build the initial ensemble: m members, each state [P0, logR_i, logC_i] with the
//   log-parameters drawn around the prior. Layout is row-major [m * WK_NSTATE], so
//   member i occupies ens[i*WK_NSTATE + 0..2]. The SAME initial ensemble feeds the
//   CPU reference and the GPU path (identical seed) -> identical results.
std::vector<double> build_initial_ensemble(const EnKFConfig& c);

// The FORECAST step, done on the CPU for one window: advance every member by
//   integrating the Windkessel ODE. `t0` is the window's start time. Operates in
//   place on the [m*WK_NSTATE] ensemble. (The GPU twin is forecast_kernel.)
void forecast_cpu(const EnKFConfig& c, std::vector<double>& ensemble, double t0);

// The ANALYSIS step (shared by CPU and GPU paths -- it is host-side dense linear
//   algebra over the 3-vector state, so there is ONE implementation). Given the
//   forecast ensemble and one scalar pressure observation, apply the stochastic
//   EnKF update in place. `obs_seed` makes the perturbed observations reproducible.
void enkf_analysis(const EnKFConfig& c, std::vector<double>& ensemble,
                   double p_obs, uint64_t obs_seed);

// Reduce an ensemble to (R_hat, C_hat, spreads) by averaging in log space and
//   exponentiating the mean (the natural average for a multiplicative parameter).
EnKFResult summarize_ensemble(const EnKFConfig& c, const std::vector<double>& ensemble);

// Run the WHOLE assimilation on the CPU (forecast + analysis for every window).
//   `observations` is the vector from synthesize_observations(). Returns the final
//   estimate and (via ensemble_out) the final ensemble. This is the trusted
//   reference the GPU forecast path is checked against.
EnKFResult run_enkf_cpu(const EnKFConfig& c, const std::vector<double>& observations,
                        std::vector<double>& ensemble_out);
