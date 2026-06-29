// ===========================================================================
// src/reference_cpu.cpp  --  Loader, serial window sampling, and the WHAM solver
// ---------------------------------------------------------------------------
// Project 1.24 : Umbrella Sampling / WHAM Free Energy Profiles
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU is checked against, written to be OBVIOUSLY
//   correct: one window after another in a plain loop, no parallelism. When the
//   GPU's per-window histograms match these, we believe the GPU. This file also
//   hosts WHAM, which both paths share (it is cheap CPU post-processing -- the
//   catalog's "WHAM iteration on CPU" pattern).
//
//   Compiled by the host C++ compiler only (no CUDA syntax here). The actual
//   per-step physics is in umbrella.h (shared host+device). See reference_cpu.h.
//
// READ THIS AFTER: umbrella.h, reference_cpu.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <algorithm>   // std::max_element, std::fill
#include <cmath>       // std::exp, std::log
#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// load_config: read the one-block text format documented in data/README.md.
//   Order (all whitespace-separated):
//     A b                              (double-well: barrier height, half-width)
//     x_min x_max nbins                (histogram grid)
//     n_windows win_min win_max k_spring
//     D dt n_equil n_sample seed
//   We validate the essentials so a typo in the sample fails loudly rather than
//   producing a meaningless PMF.
// ---------------------------------------------------------------------------
UmbrellaConfig load_config(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open umbrella config file: " + path);

    UmbrellaConfig c;
    if (!(in >> c.pot.A >> c.pot.b
             >> c.grid.x_min >> c.grid.x_max >> c.grid.nbins
             >> c.n_windows >> c.win_min >> c.win_max >> c.k_spring
             >> c.D >> c.dt >> c.n_equil >> c.n_sample >> c.seed)) {
        throw std::runtime_error(
            "bad parameters (expected 'A b  x_min x_max nbins  n_windows win_min "
            "win_max k_spring  D dt n_equil n_sample seed') in " + path);
    }
    // Sanity: a degenerate grid or empty sampling makes WHAM ill-posed.
    if (c.pot.b <= 0.0 || c.grid.nbins <= 0 || c.grid.x_max <= c.grid.x_min ||
        c.n_windows <= 0 || c.k_spring < 0.0 || c.D <= 0.0 || c.dt <= 0.0 ||
        c.n_sample <= 0) {
        throw std::runtime_error("invalid umbrella parameters in " + path);
    }
    return c;
}

// ---------------------------------------------------------------------------
// sample_windows_cpu: simulate each window serially, writing its histogram into
// the window's slice of the flat output array. Identical work to the GPU kernel
// -- both call simulate_window() from umbrella.h -- but here it is a readable for
// loop over windows. Complexity: O(n_windows * (n_equil + n_sample)).
// ---------------------------------------------------------------------------
void sample_windows_cpu(const UmbrellaConfig& c, std::vector<unsigned int>& hist_out) {
    const int nbins = c.grid.nbins;
    hist_out.assign(static_cast<std::size_t>(total_hist_size(c)), 0u);  // zeroed

    for (int k = 0; k < c.n_windows; ++k) {
        // This window's private slice of the flat histogram array.
        unsigned int* h = hist_out.data() + static_cast<std::size_t>(k) * nbins;
        simulate_window(c.pot, c.grid, window_spec(c, k),
                        c.D, c.dt, c.n_equil, c.n_sample,
                        c.seed, k, h);
    }
}

// ---------------------------------------------------------------------------
// wham_solve: the Weighted Histogram Analysis Method (self-consistent iteration).
//
//   THE MATH (full derivation in THEORY.md). Window k samples the BIASED density
//   proportional to exp(-(U(x) + w_k(x))/kT), where w_k(x) = 1/2 k (x - x0_k)^2.
//   Pool all windows. WHAM's optimal estimate of the UNBIASED probability in bin
//   i is:
//
//                          sum_k  N_{k,i}
//        p_i  =  ------------------------------------------------
//                 sum_k  Ntot_k * exp( -( w_k(x_i) - f_k ) / kT )
//
//   where N_{k,i} is window k's count in bin i, Ntot_k is window k's total count,
//   w_k(x_i) is the bias energy at bin center x_i, and the per-window free-energy
//   shift f_k closes the loop via
//
//        exp(-f_k/kT) = sum_i p_i * exp(-w_k(x_i)/kT).
//
//   These two equations are solved by FIXED-POINT ITERATION: start f_k = 0, update
//   p_i, update f_k, repeat. The PMF is F_i = -kT ln p_i, shifted so min F = 0.
//
//   We work in log-space-friendly plain doubles here (the demo's numbers are
//   moderate); THEORY notes the log-sum-exp trick production codes use for
//   numerical safety. Bins no window sampled get a sentinel and are flagged.
// ---------------------------------------------------------------------------
void wham_solve(const UmbrellaConfig& c,
                const std::vector<unsigned int>& hist,
                int iters,
                std::vector<double>& pmf,
                int* n_used) {
    const int W = c.n_windows;
    const int B = c.grid.nbins;
    const double kT = US_KT;

    // --- Precompute the bias-energy matrix c_{k,i} = exp(-w_k(x_i)/kT) and the
    //     per-window/per-bin counts. w_k(x_i) = 1/2 k (x_i - x0_k)^2. ---
    std::vector<double> bias_boltz(static_cast<std::size_t>(W) * B);  // exp(-w/kT)
    std::vector<double> Ntot(W, 0.0);                                  // counts per window
    std::vector<double> Ni(B, 0.0);                                    // pooled counts per bin

    for (int k = 0; k < W; ++k) {
        const double x0 = window_center(c, k);
        double tot = 0.0;
        for (int i = 0; i < B; ++i) {
            const double xi = grid_bin_center(c.grid, i);
            const double w  = 0.5 * c.k_spring * (xi - x0) * (xi - x0);  // bias energy
            bias_boltz[static_cast<std::size_t>(k) * B + i] = std::exp(-w / kT);
            const double n = hist[static_cast<std::size_t>(k) * B + i];
            tot   += n;
            Ni[i] += n;
        }
        Ntot[k] = tot;
    }

    // Which bins did ANY window sample? Unsampled bins carry no information.
    int used = 0;
    for (int i = 0; i < B; ++i) if (Ni[i] > 0.0) ++used;
    if (n_used) *n_used = used;

    // --- Fixed-point iteration. f[k] is window k's free-energy shift; we store
    //     exp(f_k/kT) implicitly via f_k itself and rebuild expf each sweep. ---
    std::vector<double> f(W, 0.0);          // f_k, initialized to 0
    std::vector<double> p(B, 0.0);          // current unbiased density estimate

    for (int it = 0; it < iters; ++it) {
        // (1) Update p_i from the current f_k.
        for (int i = 0; i < B; ++i) {
            if (Ni[i] <= 0.0) { p[i] = 0.0; continue; }   // never sampled
            double denom = 0.0;
            for (int k = 0; k < W; ++k) {
                // Ntot_k * exp(f_k/kT) * exp(-w_k(x_i)/kT)
                denom += Ntot[k] * std::exp(f[k] / kT)
                         * bias_boltz[static_cast<std::size_t>(k) * B + i];
            }
            p[i] = (denom > 0.0) ? Ni[i] / denom : 0.0;
        }

        // (2) Update f_k from the new p_i:  exp(-f_k/kT) = sum_i p_i exp(-w_k/kT).
        for (int k = 0; k < W; ++k) {
            double s = 0.0;
            for (int i = 0; i < B; ++i)
                s += p[i] * bias_boltz[static_cast<std::size_t>(k) * B + i];
            f[k] = (s > 0.0) ? -kT * std::log(s) : 0.0;
        }
    }

    // --- Convert the density to a PMF: F_i = -kT ln p_i, then shift min to 0. ---
    pmf.assign(B, 0.0);
    const double SENTINEL = 1.0e30;        // marks "unsampled" so callers can skip
    double fmin = SENTINEL;
    for (int i = 0; i < B; ++i) {
        if (p[i] > 0.0) {
            pmf[i] = -kT * std::log(p[i]);
            if (pmf[i] < fmin) fmin = pmf[i];
        } else {
            pmf[i] = SENTINEL;
        }
    }
    // Shift so the lowest sampled point is exactly 0 (PMFs are relative).
    if (fmin < SENTINEL) {
        for (int i = 0; i < B; ++i)
            if (pmf[i] < SENTINEL) pmf[i] -= fmin;
    }
}
