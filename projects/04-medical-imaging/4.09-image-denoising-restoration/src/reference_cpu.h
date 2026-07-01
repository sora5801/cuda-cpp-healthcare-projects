// ===========================================================================
// src/reference_cpu.h  --  Image container, loader, PSNR, serial NLM reference
// ---------------------------------------------------------------------------
// Project 4.9 : Image Denoising & Restoration  (Non-Local Means)
//
// WHAT THIS PROJECT COMPUTES
//   Given a NOISY grayscale image (here: a synthetic phantom corrupted with
//   Gaussian noise, standing in for CT quantum noise / MRI thermal noise), we
//   RESTORE it with Non-Local Means denoising: every output pixel becomes a
//   patch-similarity-weighted average of pixels in a search window around it.
//   The per-pixel math lives in nlm_core.h and is shared verbatim by the CPU
//   reference here and the GPU kernel in kernels.cu.
//
// WHY A GPU (the bottleneck NLM parallelises)
//   NLM is embarrassingly expensive but embarrassingly parallel. For an image
//   of P pixels, a search radius S and patch radius R, the cost is
//       O( P * (2S+1)^2 * (2R+1)^2 )
//   -- for every pixel, compare its patch against every patch in the window.
//   That is billions of multiply-adds for a modest clinical slice, and EACH
//   output pixel is independent (it only reads the noisy input). So we hand one
//   output pixel to one GPU thread: no locks, no atomics, pure parallel gather.
//   The catalog names this exact pattern: "custom CUDA for NLM block matching
//   (each thread computes patch distance vs. all neighbours)."
//
//   This header is PURE C++ (no CUDA) so it is safe to include from a .cu file.
//   It owns the data model (Image), the file format, and the serial baseline.
//
// READ THIS AFTER: nlm_core.h. READ THIS BEFORE: reference_cpu.cpp, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "nlm_core.h"     // NlmParams, nlm_pixel  (pure scalar math, CUDA-free)

// ---------------------------------------------------------------------------
// Image: a single-channel (grayscale) image in row-major float storage.
//   pix[row*width + col] is the intensity of pixel (row,col), normalized to
//   [0,1] (0 = black, 1 = white). Float (not uint8) so denoising arithmetic is
//   done in continuous intensity -- the natural space for the NLM average.
// ---------------------------------------------------------------------------
struct Image {
    int width  = 0;                 // columns
    int height = 0;                 // rows
    std::vector<float> pix;         // [height*width], row-major, values in [0,1]

    // Convenience: total pixel count. size_t because a clinical volume slice can
    // exceed 2^31 pixels once you stack channels.
    std::size_t size() const { return static_cast<std::size_t>(width) * height; }
};

// ---------------------------------------------------------------------------
// A denoising problem loaded from disk: the noisy image the algorithm operates
// on, the (synthetic) clean ground truth used ONLY to score quality via PSNR,
// and the NLM parameters that came with the sample file.
//   In a real clinical setting you would NOT have the clean image -- it exists
//   here purely because the data is synthetic, so we can measure how much noise
//   NLM removed. It never influences the denoising itself.
// ---------------------------------------------------------------------------
struct DenoiseProblem {
    Image noisy;        // the input the algorithm sees
    Image clean;        // synthetic ground truth (for PSNR scoring only)
    NlmParams params;   // patch/search radii, sigma, h -- from the file header
};

// Load a DenoiseProblem from the text format documented in data/README.md:
//   header line:  "<width> <height> <patch_radius> <search_radius> <sigma> <h>"
//   then height rows of width floats -- the NOISY image
//   then height rows of width floats -- the CLEAN image (ground truth)
// Throws std::runtime_error on any malformed / truncated input so demos fail loud.
DenoiseProblem load_problem(const std::string& path);

// PSNR (Peak Signal-to-Noise Ratio) in decibels between a candidate image and
// the clean reference, the standard image-restoration quality metric:
//     MSE  = mean( (a - ref)^2 )
//     PSNR = 10 * log10( PEAK^2 / MSE )   with PEAK = 1 (our images are in [0,1])
// Higher is better; a good denoiser RAISES PSNR relative to the noisy input.
// Returns +infinity if the images are identical (MSE == 0).
double psnr(const Image& a, const Image& ref);

// Serial CPU reference denoiser: fills `out` (sized like `in`) by calling
// nlm_pixel() for every output pixel in a plain double loop. This is the trusted
// baseline the GPU kernel is checked against, and the timing anchor that makes
// the GPU speed-up legible. Same math as the kernel (both call nlm_pixel).
void denoise_cpu(const Image& in, const NlmParams& params, Image& out);
