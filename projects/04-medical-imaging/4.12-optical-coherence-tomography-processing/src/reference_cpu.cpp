// ===========================================================================
// src/reference_cpu.cpp  --  The plain-C++ SD-OCT reconstruction we trust
// ---------------------------------------------------------------------------
// Project 4.12 : Optical Coherence Tomography Processing (SD-OCT reconstruction)
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- a single readable loop, a NAIVE O(N^2) DFT, no
//   parallelism, no cleverness -- so that when the GPU (cuFFT) and CPU agree, we
//   believe the GPU. The per-sample preprocessing (DC removal, window, dispersion
//   compensation) is shared with the GPU via oct_core.h, so the ONLY difference
//   between the two paths is naive-DFT-vs-cuFFT rounding.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: oct_core.h, reference_cpu.h. Compare against kernels.cu (GPU).
// ===========================================================================
#include "reference_cpu.h"
#include "oct_core.h"        // preprocess_sample, Cplx, cadd/cmul, cabs2 (SHARED)

#include <cmath>             // std::cos, std::sin
#include <fstream>           // std::ifstream
#include <stdexcept>         // std::runtime_error

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// load_bscan: read the tiny committed text B-scan. Format (data/README.md):
//     header:  n_ascan  n_spec  a2  a3
//     body:    n_ascan rows, each n_spec raw spectrum floats.
// We validate the header aggressively -- a truncated file must not silently
// reconstruct a smaller, wrong image.
// ---------------------------------------------------------------------------
OctBscan load_bscan(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open OCT B-scan file: " + path);

    OctBscan b;
    if (!(in >> b.n_ascan >> b.n_spec >> b.a2 >> b.a3) ||
        b.n_ascan <= 0 || b.n_spec <= 1 || (b.n_spec % 2) != 0) {
        throw std::runtime_error(
            "bad header (expected 'n_ascan n_spec a2 a3', n_spec even >= 2) in " + path);
    }

    b.raw.resize(static_cast<std::size_t>(b.n_ascan) * b.n_spec);
    for (std::size_t i = 0; i < b.raw.size(); ++i) {
        if (!(in >> b.raw[i]))
            throw std::runtime_error("OCT raw spectra truncated in " + path);
    }
    return b;
}

// ---------------------------------------------------------------------------
// reconstruct_cpu: the reference SD-OCT reconstruction.
//
//   For each A-scan a (independent of the others):
//     1. DC = mean of the raw spectrum (the strong non-interferometric offset).
//     2. Build the complex FFT input for every spectral sample via the SHARED
//        preprocess_sample() -- DC removal + Hann window + dispersion phase.
//     3. Naive DFT: for each depth bin z,
//            A[z] = sum_i in_i * exp(-2*pi*i*z*i_idx / N)
//        computed by hand with cos/sin. O(N^2) -- transparently correct.
//     4. Keep the first N/2 depth bins (Hermitian symmetry: the rest is a mirror).
//     5. Normalise the linear power to the A-scan's own peak, so image values are
//        0..1 and comparable across A-scans regardless of overall brightness.
//
//   The GPU path (kernels.cu) computes the SAME A[z] with cuFFT, then the same
//   |A|^2 / peak normalisation, so the two images match within tolerance.
// ---------------------------------------------------------------------------
void reconstruct_cpu(const OctBscan& b, std::vector<double>& image) {
    const int N = b.n_spec;                  // FFT length
    const int nd = oct_depth_count(N);       // depths kept = N/2
    image.assign(static_cast<std::size_t>(b.n_ascan) * nd, 0.0);

    // Reusable per-A-scan buffer of preprocessed complex FFT inputs.
    std::vector<Cplx> in(static_cast<std::size_t>(N));

    for (int a = 0; a < b.n_ascan; ++a) {
        const float* spec = &b.raw[static_cast<std::size_t>(a) * N];

        // -- step 1: DC (mean) of this A-scan's raw spectrum ------------------
        double dc = 0.0;
        for (int i = 0; i < N; ++i) dc += spec[i];
        dc /= static_cast<double>(N);

        // -- step 2: SHARED per-sample preprocessing -> complex FFT input -----
        for (int i = 0; i < N; ++i) {
            in[i] = preprocess_sample(static_cast<double>(spec[i]), dc, i, N, b.a2, b.a3);
        }

        // -- step 3+4: naive DFT, keep first N/2 depth bins ------------------
        // A[z] = sum_i in[i] * (cos(theta) + i sin(theta)),  theta = -2*pi*z*i/N.
        double peak = 0.0;
        // First pass: compute |A[z]|^2 into image and track the A-scan peak.
        for (int z = 0; z < nd; ++z) {
            Cplx acc = cplx(0.0, 0.0);
            const double w = -2.0 * M_PI * static_cast<double>(z) / N;   // per-z base angle
            for (int i = 0; i < N; ++i) {
                const double theta = w * static_cast<double>(i);          // -2*pi*z*i/N
                acc = cadd(acc, cmul(in[i], cplx(std::cos(theta), std::sin(theta))));
            }
            const double p = cabs2(acc);          // linear power |A[z]|^2
            image[static_cast<std::size_t>(a) * nd + z] = p;
            if (p > peak) peak = p;               // track this A-scan's brightest bin
        }

        // -- step 5: normalise the A-scan to its own peak (0..1) -------------
        if (peak > 0.0) {
            for (int z = 0; z < nd; ++z)
                image[static_cast<std::size_t>(a) * nd + z] /= peak;
        }
    }
}

// ---------------------------------------------------------------------------
// peak_depths: argmax depth bin per A-scan -- the deterministic integer result.
//   Ties are broken by the LOWEST index (the first `>` keeps the earlier bin),
//   which is order-independent and therefore identical on CPU and GPU.
// ---------------------------------------------------------------------------
void peak_depths(const std::vector<double>& image, int n_ascan, int n_depth,
                 std::vector<int>& out) {
    out.assign(static_cast<std::size_t>(n_ascan), 0);
    for (int a = 0; a < n_ascan; ++a) {
        const double* prof = &image[static_cast<std::size_t>(a) * n_depth];
        int best = 0;
        double bestv = prof[0];
        for (int z = 1; z < n_depth; ++z) {
            if (prof[z] > bestv) { bestv = prof[z]; best = z; }
        }
        out[static_cast<std::size_t>(a)] = best;
    }
}
