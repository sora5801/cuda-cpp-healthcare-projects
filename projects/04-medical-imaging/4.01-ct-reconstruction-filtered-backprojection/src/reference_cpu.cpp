// ===========================================================================
// src/reference_cpu.cpp  --  Loader, ramp filter, serial backprojection
// ---------------------------------------------------------------------------
// Project 4.01 : CT Reconstruction (Filtered Backprojection)
//
// Compiled by the host C++ compiler only. See reference_cpu.h.
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>
#include <fstream>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

CTProblem load_ct(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open sinogram file: " + path);
    CTProblem ct;
    if (!(in >> ct.n_angles >> ct.n_det >> ct.ds >> ct.img >> ct.world_half))
        throw std::runtime_error("bad header (expected n_angles n_det ds img world_half) in " + path);
    if (ct.n_angles <= 0 || ct.n_det <= 0 || ct.img <= 0)
        throw std::runtime_error("non-positive geometry in " + path);
    ct.sino.resize(static_cast<std::size_t>(ct.n_angles) * ct.n_det);
    for (std::size_t k = 0; k < ct.sino.size(); ++k) {
        if (!(in >> ct.sino[k]))
            throw std::runtime_error("sinogram truncated in " + path);
    }
    return ct;
}

void compute_trig(int n_angles, std::vector<float>& cosv, std::vector<float>& sinv) {
    cosv.resize(n_angles);
    sinv.resize(n_angles);
    for (int k = 0; k < n_angles; ++k) {
        // Parallel-beam scan spans 180 degrees: theta_k = k * pi / n_angles.
        const double theta = M_PI * k / n_angles;
        cosv[k] = static_cast<float>(std::cos(theta));
        sinv[k] = static_cast<float>(std::sin(theta));
    }
}

void ramp_filter(const CTProblem& ct, std::vector<float>& filtered) {
    const int n = ct.n_det;
    const double ds = ct.ds;

    // Discrete Ram-Lak ramp kernel h[lag] (spatial domain):
    //   h[0]        = 1 / (4*ds^2)
    //   h[even != 0]= 0
    //   h[odd]      = -1 / (pi^2 * lag^2 * ds^2)
    // Convolving a projection with h (then x ds) approximates the continuous
    // ramp-filtered projection FBP needs; without it the image is 1/r-blurred.
    auto hker = [ds](int lag) -> double {
        if (lag == 0)          return 1.0 / (4.0 * ds * ds);
        if (lag % 2 == 0)      return 0.0;
        return -1.0 / (M_PI * M_PI * static_cast<double>(lag) * lag * ds * ds);
    };
    // n is small here, so an O(n^2) direct convolution per row is plenty
    // (a production implementation filters in the frequency domain via FFT).

    filtered.assign(ct.sino.size(), 0.0f);
    for (int a = 0; a < ct.n_angles; ++a) {
        const float* row = &ct.sino[static_cast<std::size_t>(a) * n];
        float* out = &filtered[static_cast<std::size_t>(a) * n];
        for (int j = 0; j < n; ++j) {
            double acc = 0.0;
            for (int jp = 0; jp < n; ++jp)
                acc += static_cast<double>(row[jp]) * hker(j - jp);
            out[j] = static_cast<float>(acc * ds);   // ds = integration measure
        }
    }
}

void backproject_cpu(const CTProblem& ct, const std::vector<float>& filtered,
                     const std::vector<float>& cosv, const std::vector<float>& sinv,
                     std::vector<float>& image) {
    const int N = ct.img, n_det = ct.n_det;
    const float ds = ct.ds, W = ct.world_half;
    const float center = 0.5f * (n_det - 1);          // detector index of s=0
    const float scale = static_cast<float>(M_PI) / ct.n_angles;  // d(theta)
    const float pix = (N > 1) ? (2.0f * W / (N - 1)) : 0.0f;      // world units/pixel

    image.assign(static_cast<std::size_t>(N) * N, 0.0f);
    for (int py = 0; py < N; ++py) {
        const float wy = -W + py * pix;               // world y of this row
        for (int px = 0; px < N; ++px) {
            const float wx = -W + px * pix;           // world x of this pixel
            float acc = 0.0f;
            // Sum the filtered projection sampled where this pixel's ray hits the
            // detector at each angle: s = wx*cos + wy*sin, then linear interp.
            for (int k = 0; k < ct.n_angles; ++k) {
                const float s = wx * cosv[k] + wy * sinv[k];
                const float fidx = s / ds + center;   // fractional detector index
                const int j0 = static_cast<int>(std::floor(fidx));
                if (j0 >= 0 && j0 + 1 < n_det) {
                    const float w = fidx - j0;          // interpolation weight
                    const float* row = &filtered[static_cast<std::size_t>(k) * n_det];
                    acc += row[j0] * (1.0f - w) + row[j0 + 1] * w;
                }
            }
            image[static_cast<std::size_t>(py) * N + px] = acc * scale;
        }
    }
}
