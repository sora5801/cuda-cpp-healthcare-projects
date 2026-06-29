// ===========================================================================
// src/reference_cpu.h  --  Umbrella-sampling config, CPU reference, WHAM solver
// ---------------------------------------------------------------------------
// Project 1.24 : Umbrella Sampling / WHAM Free Energy Profiles
//
// This header declares everything the HOST side owns:
//   * UmbrellaConfig    -- the whole experiment (potential + grid + windows +
//                          dynamics settings), parsed from the data file.
//   * window_center()   -- the (window index -> restraint center x0) mapping,
//                          shared host+device so the kernel uses the same one.
//   * load_config()     -- read an UmbrellaConfig from the text sample.
//   * sample_windows_cpu() -- the CPU REFERENCE: simulate every window serially
//                          and fill a flat [n_windows * nbins] histogram array.
//                          This is the trusted baseline the GPU is checked against
//                          (identical physics in umbrella.h -> identical counts).
//   * wham_solve()      -- the WHAM self-consistent iteration that turns the stack
//                          of biased histograms into one unbiased PMF F(xi).
//                          (Runs on the CPU for both paths -- it is cheap O(iters *
//                          n_windows * nbins) post-processing, exactly as the
//                          catalog's "WHAM iteration on CPU" pattern describes.)
//
// The per-step physics (RNG, potential, Langevin update, binning) lives in
// umbrella.h so it is byte-identical on CPU and GPU. Pure C++ here; kernels.cu
// reuses UmbrellaConfig and window_center().
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "umbrella.h"   // US_HD, Potential, HistGrid, WindowSpec, simulate_window, ...

// ---------------------------------------------------------------------------
// UmbrellaConfig: one complete umbrella-sampling experiment.
//   The windows are laid out evenly: n_windows restraint centers spanning
//   [win_min, win_max], all sharing one spring constant k_spring. The histogram
//   grid (where the PMF is evaluated) is separate from the window centers so the
//   two resolutions can differ -- a real subtlety we expose to the learner.
// ---------------------------------------------------------------------------
struct UmbrellaConfig {
    Potential pot;          // the true double-well landscape (A, b)
    HistGrid  grid;         // histogram bins along the reaction coordinate

    int    n_windows = 0;   // number of umbrella windows
    double win_min = 0.0;   // first restraint center x0
    double win_max = 0.0;   // last  restraint center x0
    double k_spring = 0.0;  // harmonic spring constant shared by all windows

    double D = 0.0;         // diffusion constant of the Langevin dynamics
    double dt = 0.0;        // Langevin timestep
    int    n_equil = 0;     // warm-up steps per window (discarded)
    int    n_sample = 0;    // recorded steps per window (histogrammed)
    uint64_t seed = 0;      // base RNG seed (reproducibility)
};

// Map a window index to its restraint center x0 (evenly spaced in [win_min,
// win_max]). Shared host+device so the GPU kernel reproduces the exact centers.
US_HD inline double window_center(const UmbrellaConfig& c, int k) {
    if (c.n_windows <= 1) return c.win_min;
    return c.win_min + (c.win_max - c.win_min) * k / (c.n_windows - 1);
}

// Build the WindowSpec (center + spring) for window k.
US_HD inline WindowSpec window_spec(const UmbrellaConfig& c, int k) {
    WindowSpec w;
    w.x0       = window_center(c, k);
    w.k_spring = c.k_spring;
    return w;
}

// Total number of histogram counts across all windows = n_windows * nbins.
inline int total_hist_size(const UmbrellaConfig& c) {
    return c.n_windows * c.grid.nbins;
}

// ---------------------------------------------------------------------------
// load_config: parse an UmbrellaConfig from the whitespace-separated text format
// (see data/README.md). Throws std::runtime_error on a malformed/missing file so
// demos fail loudly rather than running on garbage.
// ---------------------------------------------------------------------------
UmbrellaConfig load_config(const std::string& path);

// ---------------------------------------------------------------------------
// sample_windows_cpu: the CPU reference. Simulate every window serially and write
// its histogram into hist_out[k*nbins .. k*nbins+nbins-1]. hist_out is sized to
// total_hist_size(c) and zeroed inside. This is the baseline the GPU must match.
// ---------------------------------------------------------------------------
void sample_windows_cpu(const UmbrellaConfig& c, std::vector<unsigned int>& hist_out);

// ---------------------------------------------------------------------------
// wham_solve: the Weighted Histogram Analysis Method.
//
//   Inputs : the full [n_windows * nbins] histogram stack and the config.
//   Output : pmf[i]  -- the unbiased free energy F(xi_i) at each bin center,
//            in units of kT, shifted so that min(pmf) == 0 (PMFs are defined only
//            up to an additive constant). Bins that no window ever visited are set
//            to a sentinel (large value) and reported as "unsampled".
//
//   WHAM solves, by fixed-point iteration, for the per-window free-energy
//   constants f_k that make all windows' reweighted estimates of p(xi) mutually
//   consistent (the math is derived in THEORY.md). `iters` controls how many
//   self-consistency sweeps to run; `n_used` (optional out) reports how many bins
//   were actually sampled. Runs identically for the CPU and GPU histograms, so a
//   matching PMF is a second, end-to-end confirmation that the two agree.
// ---------------------------------------------------------------------------
void wham_solve(const UmbrellaConfig& c,
                const std::vector<unsigned int>& hist,
                int iters,
                std::vector<double>& pmf,
                int* n_used = nullptr);
