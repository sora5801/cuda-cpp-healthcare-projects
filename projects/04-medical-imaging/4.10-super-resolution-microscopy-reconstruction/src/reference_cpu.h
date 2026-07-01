// ===========================================================================
// src/reference_cpu.h  --  Dataset + shared SMLM helpers + CPU reference
// ---------------------------------------------------------------------------
// Project 4.10 : Super-Resolution Microscopy Reconstruction  (STORM / PALM SMLM)
//
// Pure C++ (no CUDA constructs). The per-emitter math lives in smlm.h. This
// header declares:
//   * FrameStack  -- the loaded raw movie (F frames of H x W pixels).
//   * load_stack  -- read the tiny text sample (data/README.md format).
//   * detect_and_localize_cpu -- the trusted serial reference: DETECT + LOCALIZE
//     over the whole stack, returning every emitter's sub-pixel position.
//   * render_image -- rasterize a list of localizations into a fixed-point
//     super-resolution image, SHARED by CPU and GPU so both render identically.
//   * summarize   -- reduce a result to a small DETERMINISTIC digest for stdout
//     + verification.
//
// kernels.cu (the GPU path) reuses FrameStack + render_image + summarize from
// here, so the two paths produce bit-identical localizations and images
// (THEORY §6). Compiled by the host compiler AND included from .cu files.
//
// READ THIS AFTER: smlm.h.  READ BEFORE: reference_cpu.cpp, kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "smlm.h"   // Localization, smlm_localize, smlm_is_local_max, fixed-point

// ---------------------------------------------------------------------------
// FrameStack: a raw SMLM movie loaded into host memory.
//   F frames, each H rows x W columns, stored row-major and back-to-back:
//   pixel (f, r, c) lives at  data[((std::size_t)f*H + r)*W + c].  `background`
//   is the per-run camera offset subtracted during localization; `threshold` is
//   the detection cutoff (a pixel must exceed it to be a candidate). Both come
//   from the file header so CPU and GPU use identical values.
// ---------------------------------------------------------------------------
struct FrameStack {
    int F = 0, H = 0, W = 0;          // frames, height, width (pixels)
    double background = 0.0;          // per-pixel background to subtract
    double threshold  = 0.0;          // detection threshold (absolute intensity)
    std::vector<float> data;          // [F*H*W] pixel intensities, row-major

    // Byte size of the pixel buffer -- handy for cudaMalloc / cudaMemcpy.
    std::size_t bytes() const { return data.size() * sizeof(float); }
    // Pixels per frame (H*W) -- the stride between consecutive frames.
    std::size_t frame_pixels() const {
        return static_cast<std::size_t>(H) * static_cast<std::size_t>(W);
    }
};

// Load the text sample (data/README.md format): a header line
//   "F H W background threshold"  followed by F*H*W whitespace-separated floats.
// Throws std::runtime_error on any malformation so demos fail loudly.
FrameStack load_stack(const std::string& path);

// CPU REFERENCE (the trusted baseline).
//   Scan every interior pixel of every frame; where smlm_is_local_max() fires and
//   the peak sits far enough from the edge to hold a full 7x7 patch, run
//   smlm_localize(). Append each fit to `out` in a FIXED scan order (frame, then
//   row, then column) so the ordering is deterministic. Returns the count.
std::size_t detect_and_localize_cpu(const FrameStack& stack,
                                    std::vector<Localization>& out);

// RENDER (shared by CPU + GPU).
//   Rasterize `locs` into a super-resolution image of size
//   (stack.H*UPSAMPLE) x (stack.W*UPSAMPLE), accumulating each emitter's photons
//   into the sub-pixel bin its (x,y) falls in, in FIXED-POINT integers (so the
//   sum is order-independent -> the GPU's atomic render matches this exactly).
//   `img_fixed` is resized to srH*srW and filled; srH/srW return the render dims.
void render_image(const FrameStack& stack, const std::vector<Localization>& locs,
                  std::vector<unsigned long long>& img_fixed, int& srH, int& srW);

// ---------------------------------------------------------------------------
// ResultSummary: the small, DETERMINISTIC digest printed to stdout and used for
//   CPU-vs-GPU verification. It deliberately avoids printing per-emitter floats
//   at full precision (which could differ in the last ULP between compilers);
//   instead it reports exact integer counts/checksums plus a few rounded, robust
//   statistics that are stable well within the documented tolerance.
// ---------------------------------------------------------------------------
struct ResultSummary {
    std::size_t n_localizations = 0;     // total emitters localized (exact)
    int    srH = 0, srW = 0;             // super-resolution image dimensions
    unsigned long long img_checksum = 0; // sum of all fixed-point pixels (exact)
    std::size_t bright_bins = 0;         // # render bins with any signal (exact)
    double mean_x = 0.0, mean_y = 0.0;   // mean localization centre (camera px)
    double mean_sigma = 0.0;             // mean estimated PSF width (px)
    double mean_photons = 0.0;           // mean integrated intensity
};

// Build a ResultSummary from a localization list + its rendered image. Shared by
// both paths so main.cu can compare the two summaries field-by-field.
ResultSummary summarize(const std::vector<Localization>& locs,
                        const std::vector<unsigned long long>& img_fixed,
                        int srH, int srW);
