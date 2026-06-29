// ===========================================================================
// src/reference_cpu.cpp  --  Loader, Gaussian FIR, serial 1-D convolution
// ---------------------------------------------------------------------------
// Project 7.10 : Physiological Signal & Waveform Analysis
// Compiled by the host compiler only. See reference_cpu.h.
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>
#include <fstream>
#include <stdexcept>

Signal load_signal(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open signal file: " + path);
    Signal s;
    if (!(in >> s.n) || s.n <= 0)
        throw std::runtime_error("bad header (expected sample count n) in " + path);
    s.x.resize(s.n);
    for (int i = 0; i < s.n; ++i)
        if (!(in >> s.x[i])) throw std::runtime_error("signal truncated in " + path);
    return s;
}

std::vector<float> make_gaussian_filter(int K, double sigma) {
    std::vector<float> h(K);
    const double c = 0.5 * (K - 1);          // center tap
    double sum = 0.0;
    for (int k = 0; k < K; ++k) {
        const double d = (k - c) / sigma;
        h[k] = static_cast<float>(std::exp(-0.5 * d * d));
        sum += h[k];
    }
    // Normalize so the taps sum to 1 (unity DC gain -> denoises without scaling).
    for (int k = 0; k < K; ++k) h[k] = static_cast<float>(h[k] / sum);
    return h;
}

void conv1d_cpu(const Signal& s, const std::vector<float>& h, std::vector<float>& y) {
    const int n = s.n;
    const int K = static_cast<int>(h.size());
    const int halo = (K - 1) / 2;            // taps reach +/- halo around n
    y.assign(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        float acc = 0.0f;
        // Centered convolution; samples outside [0,n) are treated as zero.
        for (int k = 0; k < K; ++k) {
            const int j = i - halo + k;
            if (j >= 0 && j < n) acc += h[k] * s.x[j];
        }
        y[i] = acc;
    }
}
