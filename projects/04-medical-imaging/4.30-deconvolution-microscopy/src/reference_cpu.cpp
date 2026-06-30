// ===========================================================================
// src/reference_cpu.cpp  --  CPU Richardson-Lucy reference (direct convolution)
// ---------------------------------------------------------------------------
// Project 4.30 : Deconvolution Microscopy
//
// Compiled by the HOST compiler only (cl.exe / g++). It implements the trusted,
// obviously-correct baseline that the cuFFT-based GPU result is verified against:
//   * direct (spatial) CIRCULAR convolution -- the same operator cuFFT computes,
//   * the same per-pixel RL ratio/update math from rl_core.h (shared with the
//     GPU), so the two paths differ ONLY in how they convolve.
//
// See reference_cpu.h for the data model and the function contracts.
// ===========================================================================
#include "reference_cpu.h"
#include "rl_core.h"          // rl_ratio(), rl_update()  -- shared with the GPU

#include <cmath>              // std::exp, std::sqrt, std::fabs
#include <fstream>            // std::ifstream
#include <stdexcept>          // std::runtime_error

// ---------------------------------------------------------------------------
// load_image: parse the tiny text image format (see data/README.md).
//   header: "<w> <h>"  then h rows of w doubles. We store row-major in pix.
// ---------------------------------------------------------------------------
Image load_image(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open image file: " + path);
    Image img;
    if (!(in >> img.w >> img.h) || img.w <= 0 || img.h <= 0)
        throw std::runtime_error("bad header (expected 'w h') in " + path);
    img.pix.resize(static_cast<std::size_t>(img.w) * img.h);
    for (std::size_t i = 0; i < img.pix.size(); ++i)
        if (!(in >> img.pix[i]))
            throw std::runtime_error("image data truncated in " + path);
    return img;
}

// ---------------------------------------------------------------------------
// make_gaussian_psf: a normalized 2-D Gaussian blur kernel.
//   weight(dx,dy) = exp(-(dx^2 + dy^2) / (2 sigma^2)), then divide by the sum
//   so the kernel integrates to 1 (intensity-conserving). r is the half-width
//   in pixels; beyond ~3*sigma the weights are negligible, so r ~ 3*sigma.
// Deterministic: identical bytes on CPU and GPU (the GPU reuses this function).
// ---------------------------------------------------------------------------
Psf make_gaussian_psf(int r, double sigma) {
    Psf psf;
    psf.r = r;
    const int d = psf.d();
    psf.k.resize(static_cast<std::size_t>(d) * d);
    const double two_sigma2 = 2.0 * sigma * sigma;
    double sum = 0.0;
    for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
            const double g = std::exp(-(static_cast<double>(dx) * dx +
                                        static_cast<double>(dy) * dy) / two_sigma2);
            psf.k[static_cast<std::size_t>(dy + r) * d + (dx + r)] = g;
            sum += g;
        }
    }
    // Normalize to unit sum so blurring conserves total photon count.
    for (double& v : psf.k) v /= sum;
    return psf;
}

// ---------------------------------------------------------------------------
// convolve_circular: direct 2-D circular convolution, the reference operator.
//
//   out[y,x] = sum over (dx,dy) of  src[(y+dy) mod h, (x+dx) mod w] * w(dx,dy)
//
// where w(dx,dy) is the PSF weight (or the flipped PSF weight if `flip`). The
// modulo wrap-around is what makes this CIRCULAR -- and circular convolution is
// exactly what multiplying FFTs computes (the cuFFT path), so this CPU result
// and the GPU result are the *same* mathematical map, comparable pixel-by-pixel.
//
// Complexity: O(w*h*d*d) -- transparently correct but slow; that slowness is
// the whole motivation for the O(N log N) FFT path on the GPU.
// ---------------------------------------------------------------------------
void convolve_circular(const Image& src, const Psf& psf, bool flip, Image& out) {
    const int w = src.w, h = src.h, r = psf.r, d = psf.d();
    out.w = w; out.h = h;
    out.pix.assign(static_cast<std::size_t>(w) * h, 0.0);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double acc = 0.0;     // accumulator for output pixel (x,y)
            for (int dy = -r; dy <= r; ++dy) {
                // Circular row index: (y+dy) wrapped into [0,h). Adding h before
                // the modulo handles negative offsets without a branch surprise.
                const int sy = ((y + dy) % h + h) % h;
                for (int dx = -r; dx <= r; ++dx) {
                    const int sx = ((x + dx) % w + w) % w;   // circular column
                    // For a true convolution we read the PSF at (-dx,-dy); the
                    // RL forward step uses the PSF as-is and the back-projection
                    // step uses the flipped PSF. `flip` selects which one, so a
                    // single routine serves both convolutions in the RL loop.
                    const int kx = flip ? (-dx + r) : (dx + r);
                    const int ky = flip ? (-dy + r) : (dy + r);
                    const double weight = psf.k[static_cast<std::size_t>(ky) * d + kx];
                    acc += src.pix[static_cast<std::size_t>(sy) * w + sx] * weight;
                }
            }
            out.pix[static_cast<std::size_t>(y) * w + x] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// richardson_lucy_cpu: the iterative deconvolution reference.
//
// RL maximizes the Poisson likelihood of the observed (photon-counting) image
// given a blur PSF. One iteration, with H = "convolve with PSF" and
// H^T = "convolve with the flipped PSF" (the adjoint):
//
//     blurred   = H( estimate )                 // forward model
//     ratio     = observed / blurred            // data-fidelity ratio (per pixel)
//     correction= H^T( ratio )                  // back-project the ratio
//     estimate  = estimate * correction         // multiplicative update
//
// We start from a FLAT estimate equal to the mean of the observed image -- a
// standard, deterministic initialization that avoids biasing the result toward
// any structure. The two convolutions are the expensive part (here direct;
// cuFFT on the GPU). The per-pixel ratio/update use the SHARED rl_core.h math.
// ---------------------------------------------------------------------------
Image richardson_lucy_cpu(const Image& observed, const Psf& psf, int iters) {
    const int n = observed.size();

    // Flat initial estimate = mean intensity (non-negative, structure-free).
    double mean = 0.0;
    for (double v : observed.pix) mean += v;
    mean = (n > 0) ? mean / n : 0.0;

    Image est;
    est.w = observed.w; est.h = observed.h;
    est.pix.assign(static_cast<std::size_t>(n), mean);

    Image blurred, ratio, correction;   // scratch images reused each iteration
    ratio.w = observed.w; ratio.h = observed.h;
    ratio.pix.resize(static_cast<std::size_t>(n));

    for (int it = 0; it < iters; ++it) {
        // 1) Forward model: what the current estimate looks like through the scope.
        convolve_circular(est, psf, /*flip=*/false, blurred);

        // 2) Per-pixel data-fidelity ratio (shared rl_ratio()).
        for (int i = 0; i < n; ++i)
            ratio.pix[i] = rl_ratio(observed.pix[i], blurred.pix[i]);

        // 3) Back-project the ratio through the adjoint (flipped) PSF.
        convolve_circular(ratio, psf, /*flip=*/true, correction);

        // 4) Multiplicative update (shared rl_update()), in place.
        for (int i = 0; i < n; ++i)
            est.pix[i] = rl_update(est.pix[i], correction.pix[i]);
    }
    return est;
}

// ---------------------------------------------------------------------------
// sharpness: mean squared gradient magnitude (a scalar "how sharp" proxy).
//   For each interior pixel, gx = I(x+1,y)-I(x,y), gy = I(x,y+1)-I(x,y); we
//   average gx^2 + gy^2. Higher = more high-frequency content = sharper. Used
//   only for the human-readable report (blurry vs deconvolved), so the learner
//   sees in one number that RL restored high-frequency detail. Deterministic.
// ---------------------------------------------------------------------------
double sharpness(const Image& img) {
    const int w = img.w, h = img.h;
    if (w < 2 || h < 2) return 0.0;
    double acc = 0.0;
    long count = 0;
    for (int y = 0; y < h - 1; ++y) {
        for (int x = 0; x < w - 1; ++x) {
            const double c  = img.pix[static_cast<std::size_t>(y) * w + x];
            const double gx = img.pix[static_cast<std::size_t>(y) * w + (x + 1)] - c;
            const double gy = img.pix[static_cast<std::size_t>(y + 1) * w + x] - c;
            acc += gx * gx + gy * gy;
            ++count;
        }
    }
    return (count > 0) ? acc / static_cast<double>(count) : 0.0;
}
