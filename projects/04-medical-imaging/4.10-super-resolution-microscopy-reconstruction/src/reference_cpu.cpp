// ===========================================================================
// src/reference_cpu.cpp  --  Loader, shared render, serial SMLM reference
// ---------------------------------------------------------------------------
// Project 4.10 : Super-Resolution Microscopy Reconstruction  (STORM / PALM SMLM)
//
// Compiled by the host C++ compiler only (no CUDA here). The per-emitter fit
// math is in smlm.h; this file wires it into a whole-stack pipeline that is the
// TRUSTED BASELINE the GPU is checked against. It also defines render_image()
// and summarize(), which the GPU path reuses verbatim so the two agree exactly.
//
// READ THIS AFTER: smlm.h, reference_cpu.h. Compare with kernels.cu (GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>
#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_stack: parse the tiny text sample.
//   Header:  F H W background threshold
//   Body:    F*H*W whitespace-separated pixel intensities, row-major per frame.
//   We validate aggressively so a truncated/garbled file fails loudly rather
//   than silently localizing garbage.
// ---------------------------------------------------------------------------
FrameStack load_stack(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open frame-stack file: " + path);

    FrameStack s;
    if (!(in >> s.F >> s.H >> s.W >> s.background >> s.threshold))
        throw std::runtime_error("bad header (expected 'F H W background threshold') in " + path);
    if (s.F <= 0 || s.H < PATCH || s.W < PATCH)
        throw std::runtime_error("frame stack too small (need H,W >= " +
                                 std::to_string(PATCH) + ") in " + path);

    const std::size_t total = static_cast<std::size_t>(s.F) * s.H * s.W;
    s.data.resize(total);
    for (std::size_t i = 0; i < total; ++i)
        if (!(in >> s.data[i]))
            throw std::runtime_error("frame stack truncated in " + path);
    return s;
}

// ---------------------------------------------------------------------------
// detect_and_localize_cpu: the serial reference.
//   Deterministic scan order: frame f (outer), row r, column c (inner). For each
//   interior pixel that is a strict local maximum above threshold AND far enough
//   from the edge to hold a full 7x7 patch, run smlm_localize() and append the
//   result. The fixed scan order means the localization LIST is in a canonical
//   order -- so the GPU, which must reproduce that exact order (see kernels.cu),
//   can be compared element for element.
// ---------------------------------------------------------------------------
std::size_t detect_and_localize_cpu(const FrameStack& stack,
                                    std::vector<Localization>& out) {
    out.clear();
    const int H = stack.H, W = stack.W;
    for (int f = 0; f < stack.F; ++f) {
        // Pointer to the start of frame f (row-major H*W block).
        const float* frame = stack.data.data() + static_cast<std::size_t>(f) * H * W;
        // Only interior pixels that can hold a full patch are candidates:
        //   PATCH_R <= r < H-PATCH_R  and  PATCH_R <= c < W-PATCH_R.
        for (int r = PATCH_R; r < H - PATCH_R; ++r) {
            for (int c = PATCH_R; c < W - PATCH_R; ++c) {
                if (smlm_is_local_max(frame, H, W, r, c, stack.threshold)) {
                    out.push_back(smlm_localize(frame, H, W, r, c,
                                                stack.background, f));
                }
            }
        }
    }
    return out.size();
}

// ---------------------------------------------------------------------------
// render_image: rasterize localizations into the super-resolution grid.
//   Each emitter at camera-pixel (x,y) maps to super-resolution bin
//     sc = floor(x * UPSAMPLE),  sr = floor(y * UPSAMPLE)
//   and we ADD its integrated intensity (photons) into that bin. Accumulation is
//   in FIXED-POINT integers (smlm_to_fixed) so that when the GPU does the same
//   scatter with atomicAdd, the result is order-independent and bit-identical to
//   this loop (docs/PATTERNS.md §3). Out-of-range emitters are skipped.
// ---------------------------------------------------------------------------
void render_image(const FrameStack& stack, const std::vector<Localization>& locs,
                  std::vector<unsigned long long>& img_fixed, int& srH, int& srW) {
    srH = stack.H * UPSAMPLE;
    srW = stack.W * UPSAMPLE;
    img_fixed.assign(static_cast<std::size_t>(srH) * srW, 0ull);
    for (const Localization& L : locs) {
        // floor via truncation is safe here because x,y are >= 0 for valid fits.
        const int sc = static_cast<int>(L.x * UPSAMPLE);
        const int sr = static_cast<int>(L.y * UPSAMPLE);
        if (sr < 0 || sr >= srH || sc < 0 || sc >= srW) continue;  // off-grid
        img_fixed[static_cast<std::size_t>(sr) * srW + sc] += smlm_to_fixed(L.photons);
    }
}

// ---------------------------------------------------------------------------
// summarize: reduce a full result to a small, deterministic digest.
//   Exact integer fields (counts, checksum, bright-bin count) verify the render
//   bit-for-bit; the rounded mean statistics summarize the localizations for the
//   human-readable report and give a soft cross-check on the fit positions.
// ---------------------------------------------------------------------------
ResultSummary summarize(const std::vector<Localization>& locs,
                        const std::vector<unsigned long long>& img_fixed,
                        int srH, int srW) {
    ResultSummary S;
    S.n_localizations = locs.size();
    S.srH = srH;
    S.srW = srW;

    // Exact reductions over the fixed-point image (integer adds -> deterministic).
    for (unsigned long long px : img_fixed) {
        S.img_checksum += px;
        if (px > 0) ++S.bright_bins;
    }

    // Mean localization statistics (accumulated in the same fixed order on both
    // paths because the localization list order is canonical).
    double sx = 0.0, sy = 0.0, ssig = 0.0, sph = 0.0;
    for (const Localization& L : locs) {
        sx   += L.x;
        sy   += L.y;
        ssig += L.sigma;
        sph  += L.photons;
    }
    const double n = static_cast<double>(locs.empty() ? 1 : locs.size());
    S.mean_x       = sx / n;
    S.mean_y       = sy / n;
    S.mean_sigma   = ssig / n;
    S.mean_photons = sph / n;
    return S;
}
