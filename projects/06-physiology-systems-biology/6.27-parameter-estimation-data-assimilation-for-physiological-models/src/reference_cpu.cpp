// ===========================================================================
// src/reference_cpu.cpp  --  Loader, observation synthesis, and the CPU EnKF
// ---------------------------------------------------------------------------
// Project 6.27 : Parameter Estimation & Data Assimilation for Physiological Models
//
// ROLE IN THE PROJECT
//   The trusted, plain-C++ baseline. It (a) loads the config, (b) synthesizes the
//   "measured" noisy pressure observations from a known TRUE (R,C), (c) builds the
//   initial ensemble, and (d) runs the full Ensemble Kalman Filter loop
//   (forecast + analysis) serially. main.cu runs the SAME pipeline but with the
//   forecast done on the GPU, and checks the two agree member-for-member.
//
//   The ODE/RK4 forecast lives in windkessel.h (shared __host__ __device__), so
//   the CPU forecast here and the GPU forecast in kernels.cu are bit-identical.
//   The ANALYSIS math lives HERE and is called by both paths -- there is exactly
//   one implementation of the Kalman update, so no chance of CPU/GPU drift in it.
//
//   Compiled by the host C++ compiler only (no CUDA). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, windkessel.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>
#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_config: parse the whitespace-separated sample file. Field order is fixed
//   and documented in data/README.md so the sample is human-editable. We read in
//   the exact declared order and sanity-check the essentials so a malformed file
//   fails loudly rather than producing garbage.
// ---------------------------------------------------------------------------
EnKFConfig load_config(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open config file: " + path);
    EnKFConfig c;
    if (!(in >> c.m >> c.n_obs >> c.dt >> c.substeps
             >> c.T >> c.t_sys >> c.Q_peak
             >> c.R_true >> c.C_true >> c.P0
             >> c.obs_noise
             >> c.R_prior >> c.C_prior >> c.logR_std >> c.logC_std
             >> c.seed)) {
        throw std::runtime_error(
            "bad config (expected 'm n_obs dt substeps T t_sys Q_peak "
            "R_true C_true P0 obs_noise R_prior C_prior logR_std logC_std seed') in " + path);
    }
    if (c.m <= 0 || c.n_obs <= 0 || c.dt <= 0 || c.substeps <= 0 ||
        c.R_true <= 0 || c.C_true <= 0 || c.R_prior <= 0 || c.C_prior <= 0)
        throw std::runtime_error("invalid (non-positive) config values in " + path);
    return c;
}

// ---------------------------------------------------------------------------
// synthesize_observations: build the "measured" data the filter will assimilate.
//   We integrate the TRUE patient (R_true, C_true) forward with the shared
//   Windkessel forecast, sampling the pressure at the end of each observation
//   window and adding Gaussian measurement noise. This is our stand-in for a real
//   catheter/pressure-cuff waveform; because the truth is known we can later score
//   how well the filter recovered it. Deterministic given the seed.
// ---------------------------------------------------------------------------
std::vector<double> synthesize_observations(const EnKFConfig& c) {
    std::vector<double> obs(static_cast<std::size_t>(c.n_obs), 0.0);

    // The truth's augmented state; only pressure moves (params are static).
    double x[WK_NSTATE] = { c.P0, std::log(c.R_true), std::log(c.C_true) };

    // A dedicated RNG stream for observation noise (offset the seed so it does not
    // collide with the ensemble-initialization stream).
    SplitMix64 rng(c.seed ^ 0xABCDEF0123456789ULL);

    double t = 0.0;
    for (int k = 0; k < c.n_obs; ++k) {
        // Advance the truth through one window with the SAME integrator the filter
        // uses -> the "twin experiment" setup common in data-assimilation teaching.
        wk_forecast_member(x, t, c.dt, c.substeps, c.T, c.t_sys, c.Q_peak);
        t += enkf_window_len(c);
        // Measured pressure = true pressure + Gaussian noise (mmHg).
        obs[static_cast<std::size_t>(k)] = x[0] + c.obs_noise * rng.next_normal();
    }
    return obs;
}

// ---------------------------------------------------------------------------
// build_initial_ensemble: draw m members around the prior.
//   Each member starts at the same P0 but gets its own (log R, log C) sampled from
//   a Gaussian prior centered on the guesses (R_prior, C_prior). Sampling in LOG
//   space keeps R,C positive and reflects their multiplicative uncertainty. The
//   spread encodes "how unsure we are" before seeing data. Layout is row-major
//   [m * WK_NSTATE]; member i lives at ens[i*3 + 0..2] = [P, logR, logC].
// ---------------------------------------------------------------------------
std::vector<double> build_initial_ensemble(const EnKFConfig& c) {
    std::vector<double> ens(static_cast<std::size_t>(c.m) * WK_NSTATE, 0.0);
    SplitMix64 rng(c.seed);   // the ensemble-initialization stream
    const double logR0 = std::log(c.R_prior);
    const double logC0 = std::log(c.C_prior);
    for (int i = 0; i < c.m; ++i) {
        double* xi = &ens[static_cast<std::size_t>(i) * WK_NSTATE];
        xi[0] = c.P0;                                   // shared initial pressure
        xi[1] = logR0 + c.logR_std * rng.next_normal(); // this member's log R
        xi[2] = logC0 + c.logC_std * rng.next_normal(); // this member's log C
    }
    return ens;
}

// ---------------------------------------------------------------------------
// forecast_cpu: advance every ensemble member by one window, serially.
//   This is the CPU twin of forecast_kernel in kernels.cu. Each member integrates
//   the SAME shared Windkessel forecast, so the two produce identical states.
// ---------------------------------------------------------------------------
void forecast_cpu(const EnKFConfig& c, std::vector<double>& ensemble, double t0) {
    for (int i = 0; i < c.m; ++i) {
        double* xi = &ensemble[static_cast<std::size_t>(i) * WK_NSTATE];
        wk_forecast_member(xi, t0, c.dt, c.substeps, c.T, c.t_sys, c.Q_peak);
    }
}

// ---------------------------------------------------------------------------
// enkf_analysis: the STOCHASTIC (perturbed-observation) Ensemble Kalman update.
//
//   THE MATH (state x_i in R^3, scalar observation y of the pressure P = H x with
//   H = [1,0,0]; observation-error variance Ro = obs_noise^2):
//     1. Ensemble mean:            xbar = (1/m) Σ x_i
//     2. Sample covariance:        Pf   = (1/(m-1)) Σ (x_i - xbar)(x_i - xbar)^T
//     3. Because H picks out P, the products we need are just:
//          PfHt[j] = Cov(x_j, P) = (1/(m-1)) Σ (x_i[j]-xbar[j])(P_i-Pbar)
//          HPfHt   = Var(P)      = PfHt[0]
//     4. Kalman gain (a 3-vector here):  K[j] = PfHt[j] / (HPfHt + Ro)
//     5. Update each member toward its OWN perturbed observation:
//          x_i += K * ( y + eps_i - P_i ),   eps_i ~ N(0, Ro)
//        Perturbing the observation per member (Burgers et al. 1998) keeps the
//        posterior spread statistically correct -- without it the ensemble would
//        collapse (under-estimate its own uncertainty).
//
//   WHY host-side: for this 3-vector state the covariance is a handful of scalar
//   sums -- doing it on the CPU is both clearer and faster than shipping a 3x3
//   solve to cuBLAS/cuSOLVER (which would be a black box for two lines of algebra;
//   see THEORY §4). The EXPENSIVE part (the ensemble forecast) is what the GPU does.
//   Real high-dimensional EnKF (state = a whole PDE field) is where cuBLAS earns
//   its keep -- discussed in THEORY §7 and left as Exercise 4.
//
//   Deterministic: eps_i is drawn from SplitMix64(obs_seed) in member order, and
//   all sums are accumulated in a fixed order -> identical every run and identical
//   between the CPU and GPU paths (both call THIS function).
// ---------------------------------------------------------------------------
void enkf_analysis(const EnKFConfig& c, std::vector<double>& ensemble,
                   double p_obs, uint64_t obs_seed) {
    const int m = c.m;
    const double inv_m = 1.0 / m;

    // (1) Ensemble means of the three state components (P, logR, logC).
    double mean[WK_NSTATE] = { 0.0, 0.0, 0.0 };
    for (int i = 0; i < m; ++i) {
        const double* xi = &ensemble[static_cast<std::size_t>(i) * WK_NSTATE];
        for (int j = 0; j < WK_NSTATE; ++j) mean[j] += xi[j];
    }
    for (int j = 0; j < WK_NSTATE; ++j) mean[j] *= inv_m;

    // (2-3) Cross-covariance of each state component with the OBSERVED pressure P
    //       (state index 0). PfHt[0] is Var(P) = HPfHt.
    double PfHt[WK_NSTATE] = { 0.0, 0.0, 0.0 };
    for (int i = 0; i < m; ++i) {
        const double* xi = &ensemble[static_cast<std::size_t>(i) * WK_NSTATE];
        const double dP = xi[0] - mean[0];               // pressure anomaly
        for (int j = 0; j < WK_NSTATE; ++j) PfHt[j] += (xi[j] - mean[j]) * dP;
    }
    const double inv_m1 = 1.0 / (m - 1);                  // unbiased sample cov
    for (int j = 0; j < WK_NSTATE; ++j) PfHt[j] *= inv_m1;

    // (4) Kalman gain vector. Ro = obs_noise^2 is the measurement-error variance;
    //     the (HPfHt + Ro) denominator balances trust between model and data.
    const double Ro = c.obs_noise * c.obs_noise;
    const double denom = PfHt[0] + Ro;                    // Var(P) + Ro  (> 0)
    double K[WK_NSTATE];
    for (int j = 0; j < WK_NSTATE; ++j) K[j] = (denom > 0.0) ? PfHt[j] / denom : 0.0;

    // (5) Update every member toward its own perturbed observation.
    SplitMix64 rng(obs_seed);
    const double sqrtRo = std::sqrt(Ro);
    for (int i = 0; i < m; ++i) {
        double* xi = &ensemble[static_cast<std::size_t>(i) * WK_NSTATE];
        const double y_pert = p_obs + sqrtRo * rng.next_normal();  // perturbed obs
        const double innov = y_pert - xi[0];              // innovation for member i
        for (int j = 0; j < WK_NSTATE; ++j) xi[j] += K[j] * innov;
    }
}

// ---------------------------------------------------------------------------
// summarize_ensemble: collapse the ensemble to a point estimate + spread.
//   The parameters live in log space, so the natural "average" of R across members
//   is exp(mean(log R)) -- the geometric mean, correct for a multiplicative
//   quantity. We also report the ensemble std (posterior uncertainty) in the
//   original units via a small-perturbation conversion (std_R ≈ R_hat * std_logR).
// ---------------------------------------------------------------------------
EnKFResult summarize_ensemble(const EnKFConfig& c, const std::vector<double>& ensemble) {
    const int m = c.m;
    double mean_logR = 0.0, mean_logC = 0.0;
    for (int i = 0; i < m; ++i) {
        const double* xi = &ensemble[static_cast<std::size_t>(i) * WK_NSTATE];
        mean_logR += xi[1];
        mean_logC += xi[2];
    }
    mean_logR /= m;
    mean_logC /= m;

    double var_logR = 0.0, var_logC = 0.0;
    for (int i = 0; i < m; ++i) {
        const double* xi = &ensemble[static_cast<std::size_t>(i) * WK_NSTATE];
        var_logR += (xi[1] - mean_logR) * (xi[1] - mean_logR);
        var_logC += (xi[2] - mean_logC) * (xi[2] - mean_logC);
    }
    var_logR /= (m - 1);
    var_logC /= (m - 1);

    EnKFResult r;
    r.R_hat = std::exp(mean_logR);           // geometric-mean R estimate
    r.C_hat = std::exp(mean_logC);           // geometric-mean C estimate
    r.R_std = r.R_hat * std::sqrt(var_logR); // delta-method spread in mmHg*s/mL
    r.C_std = r.C_hat * std::sqrt(var_logC); // delta-method spread in mL/mmHg
    r.final_rmse = 0.0;                       // filled by the driver (needs obs)
    return r;
}

// ---------------------------------------------------------------------------
// run_enkf_cpu: the complete serial reference filter.
//   For each observation window: FORECAST every member, then ANALYSIS against the
//   window's observation. The per-window analysis seed is derived deterministically
//   from (config seed, window index) so the whole run is reproducible AND the GPU
//   path (which reuses the identical seeds) matches exactly.
// ---------------------------------------------------------------------------
EnKFResult run_enkf_cpu(const EnKFConfig& c, const std::vector<double>& observations,
                        std::vector<double>& ensemble_out) {
    std::vector<double> ens = build_initial_ensemble(c);
    double t = 0.0;
    double sq_err = 0.0;   // accumulate (Pbar - obs)^2 over windows for a final RMSE

    for (int k = 0; k < c.n_obs; ++k) {
        // FORECAST: advance the ensemble one window using the shared integrator.
        forecast_cpu(c, ens, t);
        t += enkf_window_len(c);

        // Post-forecast ensemble-mean pressure, scored against the observation
        // (the filter's running fit quality).
        double Pbar = 0.0;
        for (int i = 0; i < c.m; ++i) Pbar += ens[static_cast<std::size_t>(i) * WK_NSTATE];
        Pbar /= c.m;
        const double d = Pbar - observations[static_cast<std::size_t>(k)];
        sq_err += d * d;

        // ANALYSIS: pull the ensemble toward the observation (shared with the GPU
        // path). Per-window seed = base seed mixed with the window index.
        const uint64_t obs_seed = c.seed * 0x100000001B3ULL + static_cast<uint64_t>(k) + 1ULL;
        enkf_analysis(c, ens, observations[static_cast<std::size_t>(k)], obs_seed);
    }

    EnKFResult r = summarize_ensemble(c, ens);
    r.final_rmse = std::sqrt(sq_err / c.n_obs);
    ensemble_out = ens;
    return r;
}
