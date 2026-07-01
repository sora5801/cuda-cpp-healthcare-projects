// ===========================================================================
// src/reference_cpu.h  --  Image model, synthetic weights, and CPU SR reference
// ---------------------------------------------------------------------------
// Project 4.24 : CT/MRI Super-Resolution   (reduced-scope teaching version)
//
// WHAT THIS PROJECT COMPUTES
//   A learned SINGLE-IMAGE SUPER-RESOLUTION forward pass. We take a small
//   low-resolution (LR) grayscale medical-style image and upsample it R x
//   (R = SR_SCALE = 2) into a high-resolution (HR) image using the two operators
//   at the heart of ESRGAN/ESPCN: a feature convolution + ReLU, then a SUB-PIXEL
//   convolution whose R*R output channels are PIXEL-SHUFFLED into HR pixels.
//   The per-pixel math lives in sr_core.h (shared verbatim by CPU and GPU).
//
//   The demo also computes PSNR against a known ground-truth HR image, so the
//   learner sees a real image-quality number -- not just "GPU == CPU".
//
// WHY A GPU
//   Every HR output pixel is INDEPENDENT: it gathers a tiny LR neighbourhood and
//   runs a fixed amount of arithmetic. That is the classic imaging GATHER pattern
//   (PATTERNS.md §1, exemplar 4.01 CT backprojection): one thread per output
//   pixel, no synchronization. A clinical 512x512x100 volume through a 3D SR net
//   is ~500 GFLOPs/pass -- exactly the workload GPUs exist for.
//
// FILE ROLE
//   Pure C++ (NO CUDA) so cl.exe / g++ compiles it. Declares:
//     * Image           -- a row-major float grayscale image.
//     * load_image      -- read the tiny committed sample (text format).
//     * make_sr_weights -- build the fixed synthetic network weights.
//     * downsample_avg  -- make an LR image from an HR one (RxR box average).
//     * super_resolve_cpu -- the trusted serial SR forward pass (loops sr_core).
//     * psnr            -- image-quality metric vs. ground truth.
//   kernels.cu reuses Image + SrWeights (from sr_core.h). main.cu ties it all.
//
// READ THIS AFTER: sr_core.h (the per-pixel math). BEFORE: kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "sr_core.h"   // SrWeights, SR_SCALE, sr_hr_pixel (pure, CUDA-free here)

// ---------------------------------------------------------------------------
// Image: a single-channel (grayscale) image, row-major, intensities in [0,1].
//   pix has exactly w*h entries; pix[y*w + x] is the pixel at column x, row y.
//   We keep it float because medical images are normalized and the network is
//   float; [0,1] is the convention used throughout (see downsample/psnr).
// ---------------------------------------------------------------------------
struct Image {
    int w = 0;                 // width  in pixels (columns)
    int h = 0;                 // height in pixels (rows)
    std::vector<float> pix;    // [w*h] row-major intensities in [0,1]
};

// ---------------------------------------------------------------------------
// load_image: read the committed sample. Text format (see data/README.md):
//     line 1:  "<w> <h>"
//     then w*h whitespace-separated floats in [0,1], row-major.
//   Throws std::runtime_error on any malformed input so the demo fails loudly.
//   Returns the ground-truth HIGH-RES image (we derive LR from it, so we always
//   have a reference to score SR against).
// ---------------------------------------------------------------------------
Image load_image(const std::string& path);

// ---------------------------------------------------------------------------
// make_sr_weights: fill the fixed, SYNTHETIC network weights.
//   These are hand-designed (not trained): the feature layer is a small bank of
//   smoothing/edge filters, and the reconstruction layer is tuned so the R*R
//   sub-pixel phases interpolate + mildly sharpen. This gives a deterministic,
//   sensible upscaling for the demo. A trained model would replace ONLY these
//   numbers; the compute path is identical. Labeled synthetic in THEORY.md.
// ---------------------------------------------------------------------------
SrWeights make_sr_weights();

// ---------------------------------------------------------------------------
// downsample_avg: build an LR image from an HR one by averaging each RxR block.
//   This is the standard SR "degradation model": LR = box-downsample(HR). We
//   feed LR to the network and compare the network's HR output against the
//   original HR (the ground truth). Returns an image of size (hr.w/R, hr.h/R).
//   Requires hr.w and hr.h to be multiples of R (the loader/sample guarantee it).
// ---------------------------------------------------------------------------
Image downsample_avg(const Image& hr, int scale);

// ---------------------------------------------------------------------------
// super_resolve_cpu: the CPU REFERENCE forward pass.
//   Loops every HR output pixel and calls sr_hr_pixel() (from sr_core.h) -- the
//   exact same function the GPU kernel calls. This is the trusted baseline the
//   GPU result is verified against. Output image has size (lr.w*R, lr.h*R).
//   Complexity: O(HR_pixels * C_FEAT * KAREA * (1 + KAREA)) multiply-adds.
// ---------------------------------------------------------------------------
Image super_resolve_cpu(const Image& lr, const SrWeights& W, int scale);

// ---------------------------------------------------------------------------
// psnr: Peak Signal-to-Noise Ratio (dB) between two equal-size images in [0,1].
//   PSNR = 10 * log10( MAX^2 / MSE ), MAX = 1 for normalized images. Higher is
//   better; it is the standard SR quality metric. Returns +inf if the images are
//   identical (MSE = 0). Throws if sizes differ.
// ---------------------------------------------------------------------------
double psnr(const Image& a, const Image& b);
