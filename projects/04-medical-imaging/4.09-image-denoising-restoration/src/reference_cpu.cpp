// ===========================================================================
// src/reference_cpu.cpp  --  Loader, PSNR metric, serial NLM baseline
// ---------------------------------------------------------------------------
// Project 4.9 : Image Denoising & Restoration  (Non-Local Means)
//
// ROLE IN THE PROJECT
//   This is the "ground truth" implementation the GPU result is checked against.
//   It is written to be OBVIOUSLY correct -- one readable double loop, no
//   parallelism -- so that when the GPU and CPU agree we believe the GPU. The
//   actual per-pixel arithmetic is NOT duplicated here: both this loop and the
//   kernel call nlm_pixel() from nlm_core.h, so they compute bit-for-bit the
//   same thing. That shared core is the whole trick behind exact verification.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, nlm_core.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>       // std::log10
#include <fstream>     // std::ifstream
#include <limits>      // std::numeric_limits
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// read_image_rows: pull height*width floats out of an already-open stream into
//   an Image. Factored out because the file stores TWO images (noisy, clean)
//   back-to-back in the same format. Throws if the stream runs dry.
// ---------------------------------------------------------------------------
static Image read_image_rows(std::ifstream& in, int width, int height, const std::string& what) {
    Image img;
    img.width  = width;
    img.height = height;
    img.pix.resize(static_cast<std::size_t>(width) * height);
    for (std::size_t i = 0; i < img.pix.size(); ++i) {
        if (!(in >> img.pix[i]))
            throw std::runtime_error("truncated " + what + " image data");
    }
    return img;
}

// ---------------------------------------------------------------------------
// load_problem: parse the text sample format (see data/README.md).
//   Header:  width height patch_radius search_radius sigma h
//   Body:    the noisy image (height rows x width cols), then the clean image.
//   We validate every field so a malformed sample fails LOUDLY rather than
//   silently denoising garbage.
// ---------------------------------------------------------------------------
DenoiseProblem load_problem(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open sample file: " + path);

    DenoiseProblem prob;
    NlmParams& p = prob.params;
    if (!(in >> p.width >> p.height >> p.patch_radius >> p.search_radius >> p.sigma >> p.h))
        throw std::runtime_error(
            "bad header (expected: width height patch_radius search_radius sigma h) in " + path);

    // Sanity-check the geometry so downstream index math is always valid.
    if (p.width <= 0 || p.height <= 0)
        throw std::runtime_error("non-positive image size in " + path);
    if (p.patch_radius < 0 || p.search_radius < 0)
        throw std::runtime_error("negative patch/search radius in " + path);
    if (p.h <= 0.0f)
        throw std::runtime_error("filter strength h must be > 0 in " + path);

    prob.noisy = read_image_rows(in, p.width, p.height, "noisy");
    prob.clean = read_image_rows(in, p.width, p.height, "clean");
    return prob;
}

// ---------------------------------------------------------------------------
// psnr: peak signal-to-noise ratio in dB (see reference_cpu.h for the formula).
//   PEAK = 1 because intensities are normalized to [0,1]. We accumulate the MSE
//   in double precision so the metric itself does not lose accuracy on large
//   images. This scores IMAGE QUALITY; it does not gate correctness (that is
//   the max_abs_err GPU-vs-CPU check in main.cu).
// ---------------------------------------------------------------------------
double psnr(const Image& a, const Image& ref) {
    if (a.width != ref.width || a.height != ref.height)
        throw std::runtime_error("psnr: image size mismatch");
    double se = 0.0;                            // sum of squared errors
    for (std::size_t i = 0; i < a.pix.size(); ++i) {
        const double d = static_cast<double>(a.pix[i]) - static_cast<double>(ref.pix[i]);
        se += d * d;
    }
    const double mse = se / static_cast<double>(a.pix.size());
    if (mse <= 0.0) return std::numeric_limits<double>::infinity();  // identical
    const double PEAK = 1.0;                    // white level for [0,1] images
    return 10.0 * std::log10((PEAK * PEAK) / mse);
}

// ---------------------------------------------------------------------------
// denoise_cpu: the trusted serial baseline.
//   For every output pixel (row,col) we call the shared nlm_pixel(), which
//   sweeps the search window, compares patches, and returns the weighted mean.
//   Complexity: O( P * (2S+1)^2 * (2R+1)^2 ) -- the exact cost the GPU kernel
//   parallelises one-thread-per-pixel. `out` is (re)sized to match `in`.
// ---------------------------------------------------------------------------
void denoise_cpu(const Image& in, const NlmParams& params, Image& out) {
    out.width  = in.width;
    out.height = in.height;
    out.pix.assign(in.pix.size(), 0.0f);

    // The classic double loop over the output grid. Each iteration is fully
    // independent of the others (it only READS the noisy input), which is
    // precisely why kernels.cu can map this loop nest onto a 2-D thread grid.
    for (int row = 0; row < in.height; ++row) {
        for (int col = 0; col < in.width; ++col) {
            out.pix[static_cast<std::size_t>(row) * in.width + col] =
                nlm_pixel(in.pix.data(), params, row, col);
        }
    }
}
