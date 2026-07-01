// ===========================================================================
// src/reference_cpu.cpp  --  The plain-C++ Richardson-Lucy baseline we trust
// ---------------------------------------------------------------------------
// Project 4.29 : Light-Sheet Microscopy Reconstruction
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU (cuFFT) result is checked against. It runs the
//   SAME Richardson-Lucy (RL) deconvolution as kernels.cu, but does its two
//   convolutions with a direct, readable DFT instead of the FFT library. Because
//   the discrete Fourier transform is EXACT (no approximation vs. the FFT, only
//   slower), and because the per-pixel RL update comes from the shared rl_core.h,
//   the CPU and GPU compute the same math -- so agreement to a small,
//   floating-point tolerance validates the GPU path.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// COMPLEXITY (why this is the "slow but obvious" side)
//   Each direct DFT-based circular convolution of an H*W image costs O((H*W)^2)
//   real operations (a double loop over output pixels x input pixels). With two
//   convolutions per RL iteration and `iters` iterations, the CPU reference is
//   O(iters * (H*W)^2). That is fine for the tiny teaching image but explodes for
//   real volumes -- which is exactly why the GPU uses the O(N log N) FFT instead.
//
// READ THIS AFTER: reference_cpu.h, rl_core.h. Compare with kernels.cu (GPU twin).
// ===========================================================================
#include "reference_cpu.h"
#include "rl_core.h"       // rl_ratio, rl_apply  (shared per-pixel RL update)

#include <cmath>           // std::exp, std::sqrt, std::cos, std::sin
#include <fstream>         // std::ifstream
#include <stdexcept>       // std::runtime_error
#include <string>

// ---------------------------------------------------------------------------
// load_lsfm: parse the text sample (format documented in reference_cpu.h and
//   data/README.md). Header line "H W SIGMA ITERS", then H*W pixel values.
//   Throws on a missing file or a size mismatch so a broken sample fails loudly.
// ---------------------------------------------------------------------------
LsfmData load_lsfm(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open input file: " + path);

    LsfmData d;
    if (!(in >> d.H >> d.W >> d.sigma >> d.iters))
        throw std::runtime_error("bad header: expected 'H W SIGMA ITERS' in " + path);
    if (d.H <= 0 || d.W <= 0)
        throw std::runtime_error("image dimensions must be positive in " + path);

    const std::size_t n = static_cast<std::size_t>(d.H) * d.W;  // total pixels
    d.measured.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (!(in >> d.measured[i]))
            throw std::runtime_error("not enough pixel values in " + path);
    }
    return d;
}

// ---------------------------------------------------------------------------
// gaussian_psf: normalized Gaussian centered at pixel (0,0) with wrap-around.
//   For each pixel (r,c) we take the MINIMUM-IMAGE distance to the origin in each
//   axis (dr = min(r, H-r), dc = min(c, W-c)) so the kernel is symmetric under
//   the periodic (FFT) boundary. Value = exp(-(dr^2+dc^2)/(2 sigma^2)); we then
//   divide by the total so the PSF sums to 1 (convolving with it conserves flux,
//   i.e. does not brighten or darken the image, only blurs it).
//
//   Centering at the ORIGIN (not the image center) is deliberate: a PSF centered
//   at (0,0) convolves without shifting the image. If we centered it at (H/2,W/2)
//   the convolution would translate the image by half the frame -- a classic
//   FFT-deconvolution bug. Both the CPU and GPU use this same PSF.
// ---------------------------------------------------------------------------
std::vector<double> gaussian_psf(int H, int W, double sigma) {
    std::vector<double> h(static_cast<std::size_t>(H) * W);
    const double two_s2 = 2.0 * sigma * sigma;   // 2 sigma^2 denominator
    double total = 0.0;                          // running sum for normalization
    for (int r = 0; r < H; ++r) {
        int dr = r < H - r ? r : H - r;          // wrap-around distance along rows
        for (int c = 0; c < W; ++c) {
            int dc = c < W - c ? c : W - c;      // wrap-around distance along cols
            double v = std::exp(-(static_cast<double>(dr) * dr +
                                  static_cast<double>(dc) * dc) / two_s2);
            h[static_cast<std::size_t>(r) * W + c] = v;
            total += v;                          // accumulate for normalization
        }
    }
    // Normalize to unit sum (flux-preserving PSF).
    for (double& v : h) v /= total;
    return h;
}

// ---------------------------------------------------------------------------
// dft_circular_convolve (internal helper): compute  out = a (circular-conv) b
//   for two real H*W images, via the DEFINITION of circular convolution:
//       out[p,q] = sum_{r,c} a[r,c] * b[(p-r) mod H, (q-c) mod W]
//   This is exactly the operation the FFT computes as a frequency-domain product;
//   here we do it directly so the reference has ZERO dependence on any FFT code.
//   `flip_b` selects CORRELATION instead of convolution when true. The two match
//   the FREQUENCY-DOMAIN operations the GPU does, exactly:
//     * convolution  (flip_b=false)  =  IFFT( FFT(a) . FFT(b) )
//         out[p,q] = sum_{r,c} a[r,c] * b[(p-r) mod H, (q-c) mod W]
//     * correlation  (flip_b=true )  =  IFFT( FFT(a) . conj(FFT(b)) )
//         out[p,q] = sum_{r,c} a[r,c] * b[(r-p) mod H, (c-q) mod W]
//   Correlation is convolution with the mirrored kernel -- the "h^T" (adjoint)
//   step of Richardson-Lucy. For a symmetric Gaussian PSF h == h^T, but we
//   implement the true adjoint so the code is correct for ANY PSF and teaches the
//   difference clearly. These index formulas were verified numerically against
//   cuFFT's conj-product convention (see THEORY.md "How we verify correctness").
//   Complexity: O((H*W)^2). Kept simple on purpose (the trusted baseline).
// ---------------------------------------------------------------------------
static void dft_circular_convolve(const std::vector<double>& a,
                                  const std::vector<double>& b,
                                  int H, int W, bool flip_b,
                                  std::vector<double>& out) {
    out.assign(static_cast<std::size_t>(H) * W, 0.0);
    for (int p = 0; p < H; ++p) {
        for (int q = 0; q < W; ++q) {
            double acc = 0.0;                    // sum for output pixel (p,q)
            for (int r = 0; r < H; ++r) {
                // Row index into b: convolution uses (p-r), correlation uses (r-p).
                // The +H before %H makes the modulo non-negative for either sign.
                int br = flip_b ? ((r - p) % H + H) % H : ((p - r) % H + H) % H;
                for (int c = 0; c < W; ++c) {
                    int bc = flip_b ? ((c - q) % W + W) % W : ((q - c) % W + W) % W;
                    acc += a[static_cast<std::size_t>(r) * W + c] *
                           b[static_cast<std::size_t>(br) * W + bc];
                }
            }
            out[static_cast<std::size_t>(p) * W + q] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// deconvolve_cpu: the Richardson-Lucy loop (the trusted reference).
//   Given the measured image b and PSF h, initialize the estimate x_0 to a flat
//   image equal to the mean of b (a standard, unbiased start), then repeat:
//       reblur   = h  (conv)   x_k           // predict what the camera would see
//       ratio_i  = b_i / reblur_i            // per-pixel correction  (rl_ratio)
//       correct  = h^T (conv)  ratio         // back-project the correction
//       x_{k+1,i}= x_k,i * correct_i         // multiplicative update (rl_apply)
//   The two per-pixel steps come from rl_core.h so they are IDENTICAL to the GPU.
//   Returns x_iters in `estimate` (size H*W, row-major).
// ---------------------------------------------------------------------------
void deconvolve_cpu(const LsfmData& d, std::vector<double>& estimate) {
    const int H = d.H, W = d.W;
    const std::size_t n = static_cast<std::size_t>(H) * W;

    // Build the (shared) Gaussian PSF once.
    const std::vector<double> h = gaussian_psf(H, W, d.sigma);

    // Initial estimate: a flat image at the mean of the measurement. Starting
    // flat (rather than from b itself) is the textbook RL init and avoids baking
    // the blur into the seed.
    double mean = 0.0;
    for (double v : d.measured) mean += v;      // fixed-order sum -> deterministic
    mean /= static_cast<double>(n);
    estimate.assign(n, mean);

    std::vector<double> reblur, ratio, correct;  // scratch buffers reused each iter
    ratio.resize(n);

    for (int it = 0; it < d.iters; ++it) {
        // 1) Forward model: re-blur the current estimate through the PSF.
        dft_circular_convolve(h, estimate, H, W, /*flip_b=*/false, reblur);

        // 2) Per-pixel correction ratio  b / reblur  (shared rl_core.h).
        for (std::size_t i = 0; i < n; ++i)
            ratio[i] = rl_ratio(d.measured[i], reblur[i]);

        // 3) Back-project the correction through the flipped PSF (the adjoint).
        //   This is the CORRELATION of the ratio image with the PSF:
        //       correct = IFFT( FFT(ratio) . conj(FFT(psf)) )
        //   In our dft_circular_convolve(a, b, flip=true) that means a = ratio and
        //   b = psf (the FIRST argument is the one that is NOT conjugated in the
        //   frequency domain). Getting this order right is what makes the adjoint
        //   match the GPU's complex_mul_scaled(FFT(ratio), FFT(psf), conj_b=true)
        //   and keeps RL flux-conserving (verified numerically -- see THEORY.md).
        dft_circular_convolve(ratio, h, H, W, /*flip_b=*/true, correct);

        // 4) Multiplicative update  x <- x * correct  (shared rl_core.h).
        for (std::size_t i = 0; i < n; ++i)
            estimate[i] = rl_apply(estimate[i], correct[i]);
    }
}

// ---------------------------------------------------------------------------
// image_stats: deterministic summary (sum, max, L2 norm) in fixed pixel order.
//   These three numbers are what main.cu prints and what it compares between CPU
//   and GPU -- a compact, order-independent fingerprint of the reconstruction.
// ---------------------------------------------------------------------------
void image_stats(const std::vector<double>& img, double& out_sum,
                 double& out_max, double& out_l2) {
    double s = 0.0, m = 0.0, e = 0.0;   // sum, max, sum-of-squares
    for (std::size_t i = 0; i < img.size(); ++i) {
        double v = img[i];
        s += v;                          // left-to-right sum -> reproducible
        if (v > m) m = v;                // running maximum
        e += v * v;                      // energy accumulator
    }
    out_sum = s;
    out_max = m;
    out_l2 = std::sqrt(e);
}
