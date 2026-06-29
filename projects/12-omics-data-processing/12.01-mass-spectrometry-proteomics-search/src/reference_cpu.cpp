// ===========================================================================
// src/reference_cpu.cpp  --  Loader, norms, serial cosine-search reference
// ---------------------------------------------------------------------------
// Project 12.01 : Mass-Spectrometry Proteomics Search
// Compiled by the host compiler only. See reference_cpu.h.
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>
#include <fstream>
#include <stdexcept>

SpectralData load_spectra(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open spectra file: " + path);
    SpectralData s;
    if (!(in >> s.N >> s.bins >> s.target) || s.N <= 0 || s.bins <= 0)
        throw std::runtime_error("bad header (expected 'N bins target') in " + path);
    s.query.resize(s.bins);
    s.lib.resize(static_cast<std::size_t>(s.N) * s.bins);
    for (int b = 0; b < s.bins; ++b)
        if (!(in >> s.query[b])) throw std::runtime_error("query truncated in " + path);
    for (std::size_t k = 0; k < s.lib.size(); ++k)
        if (!(in >> s.lib[k])) throw std::runtime_error("library truncated in " + path);
    return s;
}

void compute_norms(const SpectralData& s, double& qnorm, std::vector<double>& libnorm) {
    double q = 0.0;
    for (int b = 0; b < s.bins; ++b) q += static_cast<double>(s.query[b]) * s.query[b];
    qnorm = std::sqrt(q);

    libnorm.assign(s.N, 0.0);
    for (int i = 0; i < s.N; ++i) {
        const float* row = &s.lib[static_cast<std::size_t>(i) * s.bins];
        double n = 0.0;
        for (int b = 0; b < s.bins; ++b) n += static_cast<double>(row[b]) * row[b];
        libnorm[i] = std::sqrt(n);
    }
}

void cosine_cpu(const SpectralData& s, double qnorm, const std::vector<double>& libnorm,
                std::vector<float>& scores) {
    scores.assign(s.N, 0.0f);
    for (int i = 0; i < s.N; ++i) {
        const float* row = &s.lib[static_cast<std::size_t>(i) * s.bins];
        // Dot product accumulated in double for accuracy (the GPU does the same).
        double dot = 0.0;
        for (int b = 0; b < s.bins; ++b)
            dot += static_cast<double>(s.query[b]) * row[b];
        const double denom = qnorm * libnorm[i];
        scores[i] = (denom > 0.0) ? static_cast<float>(dot / denom) : 0.0f;
    }
}
