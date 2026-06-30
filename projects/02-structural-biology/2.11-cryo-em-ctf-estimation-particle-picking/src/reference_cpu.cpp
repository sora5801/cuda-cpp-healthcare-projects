// ===========================================================================
// src/reference_cpu.cpp  --  Loader, naive 2-D power spectrum, CTF-fit reference
// ---------------------------------------------------------------------------
// Project 2.11 : Cryo-EM CTF Estimation & Particle Picking
//
// Compiled by the HOST compiler only (cl.exe / g++). It is the slow-but-obvious
// baseline the cuFFT GPU path is checked against. Every step here is the plain,
// transparent version of what kernels.cu does on the device. See reference_cpu.h
// for the contract and ctf_model.h for the shared physics.
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>       // std::cos, std::sin, std::sqrt
#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error
#include <algorithm>   // std::max, std::min

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// load_micrograph: parse the text format documented in data/README.md.
//   header: n pixel_size lambda cs amp_contrast true_dz
//   body  : n*n floats (row-major).
// ---------------------------------------------------------------------------
Micrograph load_micrograph(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open micrograph file: " + path);

    Micrograph m;
    // Read the optics header. We read into locals first so a malformed header is
    // rejected before we resize a (potentially huge) pixel buffer.
    double dx, lambda, cs, ac, tdz;
    if (!(in >> m.n >> dx >> lambda >> cs >> ac >> tdz) || m.n <= 0)
        throw std::runtime_error("bad header (expected 'n dx lambda cs ac true_dz') in " + path);

    m.optics.n            = m.n;
    m.optics.pixel_size   = dx;
    m.optics.lambda       = lambda;
    m.optics.cs           = cs;
    m.optics.amp_contrast = ac;
    m.true_dz             = tdz;

    // Read exactly n*n pixels; a short file is a hard error (no silent zero-fill).
    m.pix.resize(static_cast<std::size_t>(m.n) * m.n);
    for (std::size_t i = 0; i < m.pix.size(); ++i)
        if (!(in >> m.pix[i])) throw std::runtime_error("micrograph data truncated in " + path);
    return m;
}

// ---------------------------------------------------------------------------
// radial_power_profile_cpu (stages 1+2). See header for the contract.
//
//   Stage 1 -- power spectrum |X(u,v)|^2 by a direct 2-D DFT. We only need the
//   MAGNITUDE at each frequency, and the rotational average will fold the four
//   quadrants together, so we compute X over the full grid of integer frequencies
//   (u,v) in [-N/2, N/2). The DC term is removed by subtracting the image mean
//   first (otherwise |X(0,0)|^2 dwarfs every ring).
//
//   A direct 2-D DFT is O(N^4): for each of N^2 output frequencies we sum over
//   N^2 pixels. That is brutal -- it is exactly why production code (and our GPU
//   path) uses an FFT. For the small teaching image it is fine and lets us verify
//   cuFFT against an implementation with nothing hidden.
// ---------------------------------------------------------------------------
void radial_power_profile_cpu(const Micrograph& m, int nbins, std::vector<double>& prof) {
    const int N = m.n;

    // --- subtract the image mean (remove DC) ---
    double mean = 0.0;
    for (float v : m.pix) mean += v;
    mean /= static_cast<double>(N) * N;

    // Precompute cos/sin tables for the 1-D twiddle factors exp(-2pi i u x / N).
    // The 2-D DFT separates: X(u,v) = sum_x exp(-2pi i u x/N) [ sum_y img(x,y) e^... ]
    // but for clarity (this is a REFERENCE) we keep a direct double loop and just
    // cache the 1-D angles, which already makes the inner trig table-driven.
    std::vector<double> cosu(static_cast<std::size_t>(N) * N);
    std::vector<double> sinu(static_cast<std::size_t>(N) * N);
    for (int u = 0; u < N; ++u) {
        for (int x = 0; x < N; ++x) {
            const double ang = -2.0 * M_PI * u * x / N;
            cosu[static_cast<std::size_t>(u) * N + x] = std::cos(ang);
            sinu[static_cast<std::size_t>(u) * N + x] = std::sin(ang);
        }
    }

    // Accumulate the rotational average: ring_sum[r] / ring_cnt[r].
    // Radial bin r = round(sqrt(fu^2 + fv^2)) where (fu,fv) are signed frequencies
    // in [-N/2, N/2). nbins == N/2 covers DC (r=0) to Nyquist (r=N/2).
    std::vector<double> ring_sum(nbins, 0.0);
    std::vector<long>   ring_cnt(nbins, 0);

    // Separable 2-D DFT: first transform along x for every (u, y), then along y.
    // rowR[u][y] + i*rowI[u][y] = sum_x (img(x,y)-mean) * exp(-2pi i u x/N).
    std::vector<double> rowR(static_cast<std::size_t>(N) * N, 0.0);
    std::vector<double> rowI(static_cast<std::size_t>(N) * N, 0.0);
    for (int u = 0; u < N; ++u) {
        const double* cu = &cosu[static_cast<std::size_t>(u) * N];
        const double* su = &sinu[static_cast<std::size_t>(u) * N];
        for (int y = 0; y < N; ++y) {
            double re = 0.0, im = 0.0;
            const float* row = &m.pix[static_cast<std::size_t>(y) * N];
            for (int x = 0; x < N; ++x) {
                const double val = static_cast<double>(row[x]) - mean;
                re += val * cu[x];
                im += val * su[x];
            }
            rowR[static_cast<std::size_t>(u) * N + y] = re;
            rowI[static_cast<std::size_t>(u) * N + y] = im;
        }
    }
    // Second pass along y: X(u,v) = sum_y (rowR+ i rowI)(u,y) * exp(-2pi i v y/N).
    for (int v = 0; v < N; ++v) {
        const double* cv = &cosu[static_cast<std::size_t>(v) * N]; // reuse u-table (same N)
        const double* sv = &sinu[static_cast<std::size_t>(v) * N];
        // signed frequency of v (fold [N/2, N) to negative)
        const int fv = (v <= N / 2) ? v : v - N;
        for (int u = 0; u < N; ++u) {
            double re = 0.0, im = 0.0;
            for (int y = 0; y < N; ++y) {
                const double ar = rowR[static_cast<std::size_t>(u) * N + y];
                const double ai = rowI[static_cast<std::size_t>(u) * N + y];
                // (ar + i ai) * (cv - i sv)  -> complex multiply with e^{-i...}
                re += ar * cv[y] + ai * sv[y];
                im += ai * cv[y] - ar * sv[y];
            }
            const double power = re * re + im * im;
            const int fu = (u <= N / 2) ? u : u - N;
            const double rr = std::sqrt(static_cast<double>(fu) * fu +
                                        static_cast<double>(fv) * fv);
            const int r = static_cast<int>(rr + 0.5);   // nearest ring bin
            if (r >= 0 && r < nbins) {
                ring_sum[r] += power;
                ring_cnt[r] += 1;
            }
        }
    }

    // Rotational average: mean power in each ring (empty rings stay 0).
    std::vector<double> raw(nbins, 0.0);
    for (int r = 0; r < nbins; ++r)
        raw[r] = (ring_cnt[r] > 0) ? ring_sum[r] / ring_cnt[r] : 0.0;

    // Flatten the smooth background (shared with the GPU path -- same window).
    flatten_background(raw, /*win=*/4, prof);
}

// ---------------------------------------------------------------------------
// flatten_background: subtract a running-mean background. The raw radial power
// falls off smoothly (the "envelope"); the Thon rings are the small oscillation
// riding on top. Subtracting a windowed mean keeps only the oscillation, which is
// what carries the defocus. Using a plain symmetric window keeps it trivially
// reproducible on CPU and GPU.
// ---------------------------------------------------------------------------
void flatten_background(const std::vector<double>& raw, int win, std::vector<double>& out) {
    const int n = static_cast<int>(raw.size());
    out.assign(n, 0.0);
    for (int r = 0; r < n; ++r) {
        const int lo = std::max(0, r - win);
        const int hi = std::min(n - 1, r + win);
        double s = 0.0;
        for (int j = lo; j <= hi; ++j) s += raw[j];
        const double bg = s / (hi - lo + 1);  // local mean = background estimate
        out[r] = raw[r] - bg;                  // residual = ring signal
    }
}

// ---------------------------------------------------------------------------
// fit_ctf_cpu (stage 3): score every candidate defocus and take the argmax. The
// scoring is ncc_model_vs_profile() from ctf_model.h -- the SAME function the GPU
// kernel calls, so the two score curves agree to ~1e-12 and the argmax is exact.
// ---------------------------------------------------------------------------
CtfFitResult fit_ctf_cpu(const std::vector<double>& prof, const CtfParams& optics,
                         const CtfFitConfig& cfg) {
    CtfFitResult res;
    res.scores.assign(cfg.n_dz, -2.0);

    const int    half      = optics.n / 2;               // N/2 (Nyquist radius)
    const double nyquist_k = 0.5 / optics.pixel_size;    // 1/A at Nyquist

    double best_score = -3.0;
    int    best_idx   = -1;
    for (int i = 0; i < cfg.n_dz; ++i) {
        const double dz = dz_of_index(cfg, i);
        const double s  = ncc_model_vs_profile(prof.data(), cfg.nbins,
                                               cfg.r_lo, cfg.r_hi, dz,
                                               optics, half, nyquist_k);
        res.scores[i] = s;
        // Deterministic argmax: strictly-greater keeps the FIRST (lowest-index)
        // maximum on ties, matching the GPU reduction we use in main.cu.
        if (s > best_score) { best_score = s; best_idx = i; }
    }
    res.best_idx = best_idx;
    res.best_dz  = (best_idx >= 0) ? dz_of_index(cfg, best_idx) : 0.0;
    return res;
}
