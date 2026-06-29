// ===========================================================================
// src/reference_cpu.cpp  --  Loader, naive-DFT power, band integration
// ---------------------------------------------------------------------------
// Project 8.03 : EEG/MEG Spectral Processing (cuFFT)
// Compiled by the host compiler only. See reference_cpu.h.
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>
#include <fstream>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// The standard clinical EEG bands (Hz). gamma is capped at 100 (or Nyquist).
const char* const BAND_NAMES[N_BANDS] = {"delta", "theta", "alpha", "beta", "gamma"};
const double BAND_LO[N_BANDS] = {0.5, 4.0, 8.0, 13.0, 30.0};
const double BAND_HI[N_BANDS] = {4.0, 8.0, 13.0, 30.0, 100.0};

EegData load_eeg(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open EEG file: " + path);
    EegData d;
    if (!(in >> d.n_ch >> d.n >> d.fs) || d.n_ch <= 0 || d.n <= 0 || d.fs <= 0)
        throw std::runtime_error("bad header (expected 'n_ch n fs') in " + path);
    d.x.resize(static_cast<std::size_t>(d.n_ch) * d.n);
    for (std::size_t i = 0; i < d.x.size(); ++i)
        if (!(in >> d.x[i])) throw std::runtime_error("EEG data truncated in " + path);
    return d;
}

void dft_power_cpu(const EegData& d, std::vector<double>& power) {
    const int n = d.n;
    const int nf = n / 2 + 1;                 // number of non-redundant freq bins
    const double norm = 1.0 / (static_cast<double>(n) * n);
    power.assign(static_cast<std::size_t>(d.n_ch) * nf, 0.0);

    // Naive DFT: for each channel and each frequency bin k, sum the real signal
    // against cos/sin. O(N^2) per channel -- slow but transparently correct, the
    // whole point of a reference. cuFFT computes the SAME X[k] far faster.
    for (int c = 0; c < d.n_ch; ++c) {
        const float* xc = &d.x[static_cast<std::size_t>(c) * n];
        for (int k = 0; k < nf; ++k) {
            double re = 0.0, im = 0.0;
            const double w = -2.0 * M_PI * k / n;
            for (int t = 0; t < n; ++t) {
                re += xc[t] * std::cos(w * t);
                im += xc[t] * std::sin(w * t);
            }
            power[static_cast<std::size_t>(c) * nf + k] = (re * re + im * im) * norm;
        }
    }
}

void band_powers(const EegData& d, const std::vector<double>& power, std::vector<double>& bands) {
    const int n = d.n, nf = n / 2 + 1;
    bands.assign(static_cast<std::size_t>(d.n_ch) * N_BANDS, 0.0);
    for (int c = 0; c < d.n_ch; ++c) {
        for (int k = 0; k < nf; ++k) {
            const double f = static_cast<double>(k) * d.fs / n;   // frequency of bin k
            for (int b = 0; b < N_BANDS; ++b) {
                if (f >= BAND_LO[b] && f < BAND_HI[b]) {
                    bands[static_cast<std::size_t>(c) * N_BANDS + b] +=
                        power[static_cast<std::size_t>(c) * nf + k];
                }
            }
        }
    }
}
