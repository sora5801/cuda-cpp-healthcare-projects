// ===========================================================================
// src/reference_cpu.cpp  --  Loader, synthetic weights, degradation, CPU SR
// ---------------------------------------------------------------------------
// Project 4.24 : CT/MRI Super-Resolution   (reduced-scope teaching version)
//
// Compiled by the HOST compiler only (cl.exe / g++). It implements everything
// declared in reference_cpu.h. The per-pixel network math is NOT here -- it is
// in sr_core.h (shared with the GPU). This file is the "plumbing + reference".
//
// READ THIS AFTER: reference_cpu.h, sr_core.h.
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>       // std::log10
#include <fstream>     // std::ifstream
#include <limits>      // std::numeric_limits (PSNR of identical images = +inf)
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// load_image: parse "<w> <h>" then w*h floats. See data/README.md for format.
// ---------------------------------------------------------------------------
Image load_image(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open image file: " + path);
    Image img;
    // Header: width then height. Reject non-positive dimensions early.
    if (!(in >> img.w >> img.h) || img.w <= 0 || img.h <= 0)
        throw std::runtime_error("bad header (expected '<w> <h>') in " + path);
    // The degradation model averages RxR blocks, so both sides must divide by R.
    if (img.w % SR_SCALE != 0 || img.h % SR_SCALE != 0)
        throw std::runtime_error("image dims must be multiples of SR_SCALE in " + path);
    const size_t n = static_cast<size_t>(img.w) * img.h;
    img.pix.resize(n);
    for (size_t i = 0; i < n; ++i) {
        if (!(in >> img.pix[i]))
            throw std::runtime_error("pixel data truncated in " + path);
        // Clamp defensively so downstream [0,1] assumptions hold even if the
        // sample file has a stray out-of-range value.
        if (img.pix[i] < 0.0f) img.pix[i] = 0.0f;
        if (img.pix[i] > 1.0f) img.pix[i] = 1.0f;
    }
    return img;
}

// ---------------------------------------------------------------------------
// make_sr_weights: hand-designed (synthetic) weights for the two conv layers.
//
//   WHAT A GOOD 2x SR OPERATOR ACTUALLY DOES (and why these weights):
//     Nearest-neighbour upscaling just replicates each LR pixel into an RxR
//     block -- blocky, and it wastes the information in neighbouring LR pixels.
//     A trained sub-pixel SR net learns two things on top of that: (a) smooth
//     INTERPOLATION between LR cells, and (b) EDGE SHARPENING (unsharp masking)
//     to undo the blur introduced by the low-res acquisition. Our fixed weights
//     implement exactly those two, which is why the demo beats NN in PSNR.
//
//   FEATURE LAYER (1 -> C_FEAT), 3x3 each, tap k = (dy+1)*3 + (dx+1):
//     c0: IDENTITY (center tap = 1)  -- passes the LR intensity through unblurred.
//         This is the base signal the reconstruction interpolates.
//     c1: horizontal Sobel gradient  -- a demo feature (vertical-edge response).
//     c2: vertical   Sobel gradient  -- a demo feature (horizontal-edge response).
//     c3: Laplacian                  -- the high-frequency detail used to sharpen.
//     (c1,c2 are computed to show a realistic multi-channel feature bank; this
//      fixed reconstruction happens to weight only c0 and c3. A trained network
//      would learn nonzero weights on all four -- see THEORY.md "honesty".)
//
//   RECONSTRUCTION LAYER (C_FEAT -> R*R), a 3x3 conv over the feature map:
//     For each sub-pixel phase (phx,phy) we place BILINEAR interpolation weights
//     on the IDENTITY feature (c0): the HR sub-pixel sits 1/4 of a cell off the
//     LR-cell center for R=2, so it is the (1-f)(1-f), f(1-f), (1-f)f, f*f mix of
//     the center cell and its two/one diagonal neighbours (f = 0.25). Then we add
//     a Laplacian (c3) unsharp term at the center tap for edge sharpening. That
//     is a genuine, PSNR-improving upscaling -- not a blocky copy.
//
//   None of these are trained; they are a legible stand-in with the SAME compute
//   path a real network uses. THEORY.md "honesty" says this plainly.
// ---------------------------------------------------------------------------
SrWeights make_sr_weights() {
    SrWeights W{};   // value-init: all weights/biases start at 0.

    // Helper to index a 3x3 tap by signed offset (dx,dy) in {-1,0,1}.
    auto kidx = [](int dx, int dy) { return (dy + SR_KRAD) * SR_KDIM + (dx + SR_KRAD); };
    const int kc = kidx(0, 0);                       // center tap index = 4

    // --- Feature filters -----------------------------------------------------
    // c0: IDENTITY -- pass the LR pixel through, unblurred (the base to interp).
    W.feat_w[0][kc] = 1.0f;
    W.feat_b[0] = 0.0f;

    // c1: horizontal Sobel-like gradient (responds to vertical edges).
    {
        const float g[SR_KAREA] = {-1, 0, 1,  -2, 0, 2,  -1, 0, 1};
        for (int k = 0; k < SR_KAREA; ++k) W.feat_w[1][k] = 0.125f * g[k];
        W.feat_b[1] = 0.0f;   // ReLU keeps the positive (rising-edge) side only
    }

    // c2: vertical Sobel-like gradient (responds to horizontal edges).
    {
        const float g[SR_KAREA] = {-1, -2, -1,  0, 0, 0,  1, 2, 1};
        for (int k = 0; k < SR_KAREA; ++k) W.feat_w[2][k] = 0.125f * g[k];
        W.feat_b[2] = 0.0f;
    }

    // c3: Laplacian (fine detail); center +8, neighbours -1, scaled down.
    {
        const float lap[SR_KAREA] = {-1, -1, -1,  -1, 8, -1,  -1, -1, -1};
        for (int k = 0; k < SR_KAREA; ++k) W.feat_w[3][k] = 0.0625f * lap[k];
        W.feat_b[3] = 0.0f;
    }

    // --- Reconstruction layer: bilinear interp of c0 + Laplacian unsharp ------
    const float f = 1.0f / (2.0f * SR_SCALE);   // sub-pixel offset fraction = 0.25 (R=2)
    const float sharp = 0.75f;                  // unsharp strength (tuned; see THEORY.md)
    for (int o = 0; o < SR_C_OUT; ++o) {
        const int phx = o % SR_SCALE;                    // sub-pixel phase in x (0..R-1)
        const int phy = o / SR_SCALE;                    // sub-pixel phase in y
        // Which neighbour cell this phase leans toward: phase 0 -> the cell on
        // the -x/-y side, phase 1 -> the +x/+y side (for R=2).
        const int sgnx = (phx == 0) ? -1 : 1;
        const int sgny = (phy == 0) ? -1 : 1;
        // Bilinear weights over the 4 nearest LR cells (center + two/one diagonal).
        const float wc  = (1.0f - f) * (1.0f - f);       // center cell
        const float wx  = f * (1.0f - f);                // horizontal neighbour
        const float wy  = (1.0f - f) * f;                // vertical neighbour
        const float wxy = f * f;                         // diagonal neighbour
        W.rec_w[o][0][kidx(0, 0)]       += wc;           // identity feature, center
        W.rec_w[o][0][kidx(sgnx, 0)]    += wx;           // identity, x-neighbour
        W.rec_w[o][0][kidx(0, sgny)]    += wy;           // identity, y-neighbour
        W.rec_w[o][0][kidx(sgnx, sgny)] += wxy;          // identity, diagonal
        W.rec_w[o][3][kc]               += sharp;        // Laplacian unsharp (center)
        W.rec_b[o] = 0.0f;                               // DC carried by the base
    }
    return W;
}

// ---------------------------------------------------------------------------
// downsample_avg: LR[i,j] = mean over the RxR HR block at (i*R, j*R).
//   The forward degradation model. Averaging (not subsampling) mimics a real
//   low-resolution acquisition where each LR pixel integrates a larger area.
// ---------------------------------------------------------------------------
Image downsample_avg(const Image& hr, int scale) {
    Image lr;
    lr.w = hr.w / scale;
    lr.h = hr.h / scale;
    lr.pix.assign(static_cast<size_t>(lr.w) * lr.h, 0.0f);
    const float inv = 1.0f / static_cast<float>(scale * scale);  // 1/(R*R)
    for (int ly = 0; ly < lr.h; ++ly) {
        for (int lx = 0; lx < lr.w; ++lx) {
            float acc = 0.0f;
            // Sum the RxR HR block that this LR pixel represents.
            for (int dy = 0; dy < scale; ++dy)
                for (int dx = 0; dx < scale; ++dx)
                    acc += hr.pix[(size_t)(ly * scale + dy) * hr.w + (lx * scale + dx)];
            lr.pix[(size_t)ly * lr.w + lx] = acc * inv;   // block mean
        }
    }
    return lr;
}

// ---------------------------------------------------------------------------
// super_resolve_cpu: serial forward pass. One call to sr_hr_pixel() per HR
//   pixel -- the SAME function the GPU kernel runs, so results match bit-for-bit.
// ---------------------------------------------------------------------------
Image super_resolve_cpu(const Image& lr, const SrWeights& W, int scale) {
    Image hr;
    hr.w = lr.w * scale;
    hr.h = lr.h * scale;
    hr.pix.assign(static_cast<size_t>(hr.w) * hr.h, 0.0f);
    for (int hy = 0; hy < hr.h; ++hy)
        for (int hx = 0; hx < hr.w; ++hx)
            // sr_hr_pixel lives in sr_core.h; it does the whole per-pixel network.
            hr.pix[(size_t)hy * hr.w + hx] =
                sr_hr_pixel(lr.pix.data(), lr.w, lr.h, hx, hy, W);
    return hr;
}

// ---------------------------------------------------------------------------
// psnr: 10*log10(1 / MSE) since MAX intensity is 1 for normalized images.
// ---------------------------------------------------------------------------
double psnr(const Image& a, const Image& b) {
    if (a.w != b.w || a.h != b.h)
        throw std::runtime_error("psnr: image size mismatch");
    const size_t n = static_cast<size_t>(a.w) * a.h;
    double mse = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(a.pix[i]) - static_cast<double>(b.pix[i]);
        mse += d * d;
    }
    mse /= static_cast<double>(n);
    if (mse <= 0.0) return std::numeric_limits<double>::infinity();
    // MAX = 1.0 -> PSNR = 10 log10(1/MSE) = -10 log10(MSE).
    return -10.0 * std::log10(mse);
}
