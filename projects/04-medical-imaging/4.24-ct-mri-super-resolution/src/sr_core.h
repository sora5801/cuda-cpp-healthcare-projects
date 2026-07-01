// ===========================================================================
// src/sr_core.h  --  The ONE TRUE per-pixel super-resolution math (HD-core)
// ---------------------------------------------------------------------------
// Project 4.24 : CT/MRI Super-Resolution   (reduced-scope teaching version)
//
// WHY THIS HEADER EXISTS  (PATTERNS.md §2, the HD-macro idiom)
//   The CPU reference (reference_cpu.cpp, compiled by cl.exe / g++) and the GPU
//   kernel (kernels.cu, compiled by nvcc) must produce BYTE-FOR-BYTE identical
//   arithmetic so that "GPU == CPU" verification is exact, not approximate. The
//   only way to guarantee that is to write the per-output-pixel formula ONCE, in
//   a header both compilers include, decorated `__host__ __device__`. The host
//   loops it over every pixel; the kernel calls it from one thread per pixel.
//
//   Keep this header free of CUDA-ONLY constructs (no __global__, no <<<>>>, no
//   cudaXxx). Only the SR_HD decorator, plain math, and POD structs live here so
//   the host compiler can include it unchanged.
//
// WHAT WE ACTUALLY COMPUTE  (see ../THEORY.md for the full derivation)
//   A single-image super-resolution (SISR) forward pass built from the exact
//   two building blocks ESRGAN/ESPCN use for the UPSAMPLER:
//
//     (1) A learned feature convolution:  for each low-res (LR) pixel we apply
//         C_FEAT separate 3x3 filters (a tiny conv layer) followed by a ReLU,
//         producing a C_FEAT-channel feature map at LR resolution.
//
//     (2) A SUB-PIXEL convolution:  a second 3x3 conv maps those C_FEAT features
//         to (R*R) output channels, where R is the upscale factor. Those R*R
//         channels are then rearranged by PIXEL SHUFFLE (a.k.a. depth-to-space)
//         into an R x R block of the high-res (HR) image. This is the efficient
//         sub-pixel upsampling of Shi et al. 2016 -- the same operator ESRGAN
//         stacks to reach 4x. It replaces a transposed convolution and avoids
//         its checkerboard artifacts.
//
//   The weights here are FIXED and SYNTHETIC (a smoothing feature bank + an
//   edge-sharpening reconstruction bank), NOT trained -- see THEORY.md "honesty".
//   That keeps the demo deterministic and the math legible; a real network would
//   load learned weights, but the *compute pattern* (per-output-pixel gather +
//   small conv + pixel shuffle) is exactly what a deployed SR inference kernel
//   runs. This is the reduced-scope teaching version (CLAUDE.md §13).
//
//   THREAD-TO-DATA MAP (used by kernels.cu): one GPU thread owns one HR output
//   pixel (hx, hy). It figures out which LR pixel and which sub-pixel phase it
//   belongs to, gathers the 3x3 LR neighbourhood, and evaluates sr_hr_pixel()
//   below to produce that one HR intensity. Fully independent -> embarrassingly
//   parallel gather (PATTERNS.md §1, exemplar 4.01 CT backprojection).
//
// READ THIS AFTER: reference_cpu.h (Image struct); BEFORE: kernels.cu, main.cu.
// ===========================================================================
#pragma once

// ---------------------------------------------------------------------------
// SR_HD: expands to `__host__ __device__` only when compiled by nvcc (which
// defines __CUDACC__). Under the plain host compiler the decorators do not
// exist, so we expand to nothing -- the function is then an ordinary inline.
// This one macro is what lets a single definition serve BOTH sides.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define SR_HD __host__ __device__
#else
#define SR_HD
#endif

// ---------------------------------------------------------------------------
// Network shape constants. Small on purpose: this is a teaching forward pass,
// not a production model. Every one is documented so the learner can map the
// numbers to the math in THEORY.md.
// ---------------------------------------------------------------------------
// Upscale factor R. The HR image is R times larger in each axis than the LR
// image (so R*R more pixels). R=2 is the classic "thick-slice -> isotropic"
// and "2x CT SR" setting from the catalog deep-dive.
#define SR_SCALE 2

// Number of feature channels produced by the first conv layer. A real ESPCN
// uses 32-64; we use a tiny bank so the whole forward pass is easy to trace by
// hand, yet the code path is identical to a wide network.
#define SR_C_FEAT 4

// Convolution kernels are 3x3 (radius 1). Same footprint ESRGAN/ESPCN use for
// their body convolutions; big enough to see edges, small enough to teach.
#define SR_KRAD 1                       // kernel radius: taps span [-1, +1]
#define SR_KDIM (2 * SR_KRAD + 1)       // kernel side length = 3
#define SR_KAREA (SR_KDIM * SR_KDIM)    // taps per filter = 9

// Number of output channels of the sub-pixel conv = R*R (one per HR sub-pixel
// phase inside each LR cell). Pixel-shuffle scatters these into an R x R block.
#define SR_C_OUT (SR_SCALE * SR_SCALE)  // = 4 for R=2

// ---------------------------------------------------------------------------
// SrWeights: the (fixed, synthetic) parameters of the two conv layers.
//   Laid out as flat arrays so the SAME struct can live in host memory and in
//   GPU __constant__ memory (kernels.cu copies it there). All POD, no pointers.
//
//   feat_w[c][k]  : weight of tap k (0..8, row-major 3x3) for feature channel c.
//   feat_b[c]     : bias added to feature channel c before the ReLU.
//   rec_w[o][c][k]: weight mapping feature channel c, tap k -> output channel o.
//   rec_b[o]      : bias for output channel o (one HR sub-pixel phase).
//
//   Index math is spelled out at every use so the learner never has to guess a
//   stride. See make_sr_weights() in reference_cpu.cpp for how these are filled.
// ---------------------------------------------------------------------------
struct SrWeights {
    float feat_w[SR_C_FEAT][SR_KAREA];             // layer 1: 1 -> C_FEAT, 3x3
    float feat_b[SR_C_FEAT];                        // layer 1 biases
    float rec_w[SR_C_OUT][SR_C_FEAT][SR_KAREA];     // layer 2: C_FEAT -> R*R, 3x3
    float rec_b[SR_C_OUT];                          // layer 2 biases
};

// ---------------------------------------------------------------------------
// sr_at: clamped ("replicate"/edge-padded) read of an LR pixel.
//   Purpose: convolutions near the image border reach outside [0,w)x[0,h). We
//   clamp the coordinate to the nearest valid pixel (replicate padding) rather
//   than zero-padding, because zero-padding would darken the borders of a
//   medical image and bias the PSNR. This exact same clamp runs on CPU and GPU.
//   Params: lr [h*w] row-major LR intensities in [0,1]; (x,y) possibly OOB.
//   Returns: the intensity of the clamped pixel.
// ---------------------------------------------------------------------------
SR_HD inline float sr_at(const float* lr, int w, int h, int x, int y) {
    // Branchless-ish clamp; ternaries compile to min/max on both targets.
    x = x < 0 ? 0 : (x >= w ? w - 1 : x);
    y = y < 0 ? 0 : (y >= h ? h - 1 : y);
    return lr[(size_t)y * w + x];
}

// ---------------------------------------------------------------------------
// sr_relu: the rectified-linear activation, max(0, v). Documented as its own
//   function so THEORY.md can point at "the one nonlinearity in the network".
// ---------------------------------------------------------------------------
SR_HD inline float sr_relu(float v) { return v > 0.0f ? v : 0.0f; }

// ---------------------------------------------------------------------------
// sr_feature: evaluate ONE feature channel c at LR location (lx, ly).
//   This is layer 1 of the network for a single output feature: a 3x3
//   convolution over the (single-channel) LR image plus a bias, then ReLU.
//
//     f_c(lx,ly) = ReLU( bias_c + sum_{dy,dx} w_c[dy,dx] * LR(lx+dx, ly+dy) )
//
//   Params:
//     lr,w,h : LR image and its dimensions.
//     lx,ly  : the LR pixel whose feature we compute.
//     W      : the weight bundle (feat_w/feat_b used here).
//     c      : which feature channel (0..C_FEAT-1).
//   Returns: the scalar activation of feature channel c at (lx,ly).
//
//   Complexity: O(KAREA) = 9 multiply-adds. Called C_FEAT times per LR pixel.
// ---------------------------------------------------------------------------
SR_HD inline float sr_feature(const float* lr, int w, int h,
                              int lx, int ly, const SrWeights& W, int c) {
    float acc = W.feat_b[c];                 // start from the learned bias
    // Walk the 3x3 window; tap index k runs row-major 0..8 to match feat_w.
    int k = 0;
    for (int dy = -SR_KRAD; dy <= SR_KRAD; ++dy) {
        for (int dx = -SR_KRAD; dx <= SR_KRAD; ++dx) {
            const float px = sr_at(lr, w, h, lx + dx, ly + dy);  // edge-clamped
            acc += W.feat_w[c][k] * px;      // weight * neighbour intensity
            ++k;
        }
    }
    return sr_relu(acc);                     // the layer-1 nonlinearity
}

// ---------------------------------------------------------------------------
// sr_hr_pixel: THE headline function -- compute ONE high-res output pixel.
//   Given an HR coordinate (hx, hy), it performs the whole forward pass for
//   that pixel:
//
//     1. Decompose (hx,hy) into the LR cell (lx,ly) = (hx/R, hy/R) it lives in
//        and its SUB-PIXEL PHASE (px,py) = (hx%R, hy%R) inside that cell. The
//        phase selects which of the R*R output channels feeds this HR pixel --
//        this IS the pixel-shuffle / depth-to-space mapping, done implicitly by
//        indexing instead of by physically reshuffling channels.
//
//     2. Compute the C_FEAT features at the LR cell (layer 1, sr_feature).
//        NOTE: we recompute the features per HR pixel. That trades a little
//        arithmetic for zero inter-thread communication -- ideal for a gather
//        kernel where each thread is independent (THEORY.md "GPU mapping"
//        discusses the shared-memory optimization we deliberately skip here).
//
//     3. Apply layer 2 (the sub-pixel conv) but ONLY for the output channel
//        o = py*R + px selected by this pixel's phase: a 3x3 conv over the
//        feature map, summed across all C_FEAT channels, plus a bias.
//
//     4. Clamp the result into the valid intensity range [0,1] (medical images
//        are normalized; SR must not invent negative or >1 intensities).
//
//   Params:
//     lr,lw,lh : LR image (row-major) and its width/height.
//     hx,hy    : the HR pixel to synthesize (0..lw*R-1, 0..lh*R-1).
//     W        : the network weights.
//   Returns: the HR intensity at (hx,hy), in [0,1].
//
//   Because EVERY term here is evaluated identically on host and device (same
//   order of the same float ops), the CPU and GPU results match to the last
//   bit for R=2 -- so verification uses an essentially-zero tolerance.
// ---------------------------------------------------------------------------
SR_HD inline float sr_hr_pixel(const float* lr, int lw, int lh,
                               int hx, int hy, const SrWeights& W) {
    // (1) HR pixel -> (LR cell, sub-pixel phase). Integer div/mod by R.
    const int lx = hx / SR_SCALE;            // LR column this HR pixel maps to
    const int ly = hy / SR_SCALE;            // LR row
    const int phx = hx - lx * SR_SCALE;      // = hx % R, phase in x (0..R-1)
    const int phy = hy - ly * SR_SCALE;      // = hy % R, phase in y
    const int o = phy * SR_SCALE + phx;      // pixel-shuffle: which out channel

    // (2)+(3) Layer-2 sub-pixel conv for THIS output channel o only.
    //     The reconstruction layer is a 3x3 conv over the layer-1 FEATURE MAP:
    //     for each feature channel c and each tap k (offset (dx,dy)) it needs the
    //     feature of channel c at the NEIGHBOURING LR cell (lx+dx, ly+dy). We
    //     compute those neighbour features on the fly with sr_feature() -- keeping
    //     each thread completely self-contained (no shared state, no communication),
    //     which is exactly what makes the gather kernel embarrassingly parallel.
    //     (A production kernel would first materialize the whole feature map once
    //     and reuse it; we trade that memory for simplicity -- see THEORY.md.)
    float acc = W.rec_b[o];                   // learned bias for this phase
    int k = 0;
    for (int dy = -SR_KRAD; dy <= SR_KRAD; ++dy) {
        for (int dx = -SR_KRAD; dx <= SR_KRAD; ++dx) {
            for (int c = 0; c < SR_C_FEAT; ++c) {
                // Feature of channel c at the neighbouring LR cell (lx+dx,ly+dy).
                const float fc = sr_feature(lr, lw, lh, lx + dx, ly + dy, W, c);
                acc += W.rec_w[o][c][k] * fc; // weight * feature
            }
            ++k;
        }
    }

    // (4) Clamp to the valid normalized-intensity range.
    if (acc < 0.0f) acc = 0.0f;
    if (acc > 1.0f) acc = 1.0f;
    return acc;
}
