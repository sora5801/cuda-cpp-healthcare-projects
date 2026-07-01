// ===========================================================================
// src/reference_cpu.h  --  Image loader + serial Demons reference + SSD metric
// ---------------------------------------------------------------------------
// Project 4.8 : Deformable Image Registration (reduced-scope teaching version)
//
// Pure C++ (no CUDA). The per-pixel physics lives in demons.h; kernels.cu
// reuses DemonsParams and the same DM_HD functions, so the GPU displacement
// field matches this CPU reference within the float-accumulation tolerance
// documented in ../THEORY.md.
//
// WHAT'S DECLARED HERE
//   * DirImages   -- the loaded fixed + moving images and their geometry.
//   * load_images -- read the tiny synthetic sample (data/README.md format).
//   * register_cpu-- the trusted serial Demons solver (the ground truth).
//   * warp_image  -- resample M by a displacement field (shared with main).
//   * ssd         -- sum of squared differences, the headline similarity metric.
//
// READ THIS AFTER: demons.h. Then reference_cpu.cpp, then kernels.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "demons.h"   // DemonsParams and the shared per-pixel formulas

// ---------------------------------------------------------------------------
// DirImages -- one registration problem instance.
//   fixed / moving are row-major [ny*nx] intensity images in [0,1] (grayscale).
//   nx,ny are the dimensions. Everything downstream indexes as iy*nx + ix.
// ---------------------------------------------------------------------------
struct DirImages {
    int nx = 0, ny = 0;               // image dimensions in pixels
    std::vector<double> fixed;        // F: the target we register TO   [ny*nx]
    std::vector<double> moving;       // M: the image we deform         [ny*nx]
};

// Load the sample text file (data/README.md documents the exact format):
//   line 1: "nx ny"
//   then nx*ny fixed intensities, then nx*ny moving intensities (whitespace).
// Throws std::runtime_error on a missing/malformed file so demos fail loudly.
DirImages load_images(const std::string& path);

// ssd: sum over all pixels of (a[i]-b[i])^2. Lower = more similar. This is the
// dissimilarity metric Demons drives down; main.cu reports SSD before vs. after
// registration so the learner can SEE the moving image snapping onto the fixed.
double ssd(const std::vector<double>& a, const std::vector<double>& b);

// warp_image: resample `im.moving` at (x+ux, y+uy) for every pixel via bilinear
// interpolation, producing the warped moving image Mw = M(x+u). Used both to
// report SSD-after and by main.cu after the GPU run. `warped` is resized to
// nx*ny. Shares dm_bilinear() with the solver so warping is consistent.
void warp_image(const DirImages& im,
                const std::vector<double>& ux, const std::vector<double>& uy,
                std::vector<double>& warped);

// register_cpu: the SERIAL Demons reference. Runs P.iters iterations of
// (warp -> force -> add -> Gaussian-smooth) and returns the final displacement
// field in ux,uy (each [ny*nx], initialized to zero inside). This is the
// baseline the GPU field is asserted against, and whose wall time (measured in
// main.cu) makes the GPU speed-up legible. Complexity: O(iters * nx * ny * r).
void register_cpu(const DirImages& im, const DemonsParams& P,
                  std::vector<double>& ux, std::vector<double>& uy);
