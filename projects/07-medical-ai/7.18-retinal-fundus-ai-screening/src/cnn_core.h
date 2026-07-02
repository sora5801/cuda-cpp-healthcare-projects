// ===========================================================================
// src/cnn_core.h  --  The ONE shared CNN math core (CPU == GPU parity)
// ---------------------------------------------------------------------------
// Project 7.18 : Retinal Fundus AI Screening
//
// WHY THIS FILE EXISTS  (PATTERNS.md section 2: the __host__ __device__ core)
//   The single most useful idiom in this repo: put the *per-element physics*
//   (here: the per-output-pixel convolution MAC and the ReLU activation) in ONE
//   header as `__host__ __device__` inline functions. Then the CPU reference
//   (reference_cpu.cpp, compiled by cl.exe) and the GPU kernel (kernels.cu,
//   compiled by nvcc) call the SAME code -> byte-for-byte identical arithmetic
//   per pixel, which makes GPU-vs-CPU verification meaningful instead of a
//   coincidence. The only differences that remain are floating-point summation
//   ORDER (see THEORY.md section 5 "Numerics"), which is why the tolerance is a
//   small physical epsilon rather than exact zero.
//
// WHAT A "FUNDUS SCREENING CNN" IS (the teaching model)
//   A colour retinal fundus photograph is graded for diabetic-retinopathy (DR)
//   severity 0..4 (none / mild / moderate / severe / proliferative). Production
//   systems (EfficientNet-B4, Swin-Transformer, RETFound) stack dozens of
//   learned convolution layers. We ship a faithful but REDUCED-SCOPE teaching
//   CNN with the same skeleton every deep net has:
//
//       input image (C_IN channels, H x W)
//         -> Conv 3x3 (C_IN -> C1)  -> ReLU  -> 2x2 max-pool     (layer 1)
//         -> Conv 3x3 (C1  -> C2)   -> ReLU  -> 2x2 max-pool     (layer 2)
//         -> global average pool  (C2 numbers)                   (embedding)
//         -> fully-connected (C2 -> NUM_CLASSES) + bias          (classifier)
//         -> softmax -> argmax = predicted DR grade
//
//   The convolution weights are FIXED, hand-designed edge/blob/colour detectors
//   (not learned) so the demo is deterministic and needs no training data or a
//   framework. THEORY.md section 7 explains exactly what a real trained model
//   changes (the weights, not the arithmetic). This is labelled a teaching
//   model everywhere and is NOT for clinical use (CLAUDE.md section 1, section 8).
//
// GPU PATTERN
//   The dominant cost of any CNN is the convolution layers. Each output pixel
//   reads a small K x K neighbourhood across all input channels; neighbouring
//   outputs share almost all of those inputs, so the naive "one thread per
//   output pixel, re-read the window from global memory" wastes bandwidth. The
//   optimized kernel TILES a block of the input into SHARED MEMORY once (with a
//   halo) and every thread reads its window from on-chip memory -- the 2-D
//   analog of the 1-D tiling lesson in flagship 7.10. See kernels.cu.
//
// READ THIS AFTER: reference_cpu.h (the model container).  Pure enough to be
// #included by BOTH the host compiler and nvcc -- so NO __global__ here, and no
// CUDA-only types; only the HD-macro decorated inline helpers.
// ===========================================================================
#pragma once

// ---------------------------------------------------------------------------
// The HD-macro idiom (PATTERNS.md section 2).
//   When this header is pulled in by nvcc (which defines __CUDACC__) we decorate
//   the helpers with `__host__ __device__` so they compile for BOTH the CPU and
//   the GPU. When pulled in by the plain host compiler (reference_cpu.cpp), the
//   decorators do not exist, so we #define them away to nothing.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define CNN_HD __host__ __device__
#else
#define CNN_HD
#endif

// --- Fixed network geometry (small on purpose: a teaching model) ------------
// These are compile-time constants so both the CPU and GPU code, and the tile
// sizing in kernels.cu, agree on the shapes without passing them around.
static constexpr int CNN_C_IN       = 3;   // input channels: R, G, B (fundus is colour)
static constexpr int CNN_C1         = 6;   // feature maps after conv layer 1
static constexpr int CNN_C2         = 12;  // feature maps after conv layer 2
static constexpr int CNN_KSIZE      = 3;   // 3x3 convolution kernels (halo = 1)
static constexpr int CNN_HALO       = (CNN_KSIZE - 1) / 2;   // = 1 border pixel each side
static constexpr int CNN_NUM_CLASSES = 5;  // DR grades 0..4 (none..proliferative)

// ---------------------------------------------------------------------------
// relu: the rectified-linear activation, max(0, x).
//   The single most common non-linearity in vision CNNs: it is cheap, keeps
//   gradients alive for positive activations, and introduces the sparsity that
//   lets stacked convolutions build up nonlinear feature detectors. Marked HD so
//   the CPU reference and the GPU kernel apply the EXACT same clamp.
// ---------------------------------------------------------------------------
CNN_HD inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

// ---------------------------------------------------------------------------
// conv_at: compute ONE output pixel of ONE output feature map from a "same"-
//   padded K x K convolution over ALL input channels. This is the per-element
//   physics shared by CPU and GPU.
//
//   Math:  out(oc, y, x) = bias[oc]
//                        + sum_{ic=0}^{C_in-1} sum_{ky=0}^{K-1} sum_{kx=0}^{K-1}
//                              W[oc,ic,ky,kx] * in(ic, y + ky - halo, x + kx - halo)
//   with zero padding for samples that fall outside [0,H) x [0,W).
//
//   Parameters (all host or device pointers into flat, row-major buffers):
//     in     : [C_in * H * W]  input image / feature stack, channel-major then row-major
//     C_in   : number of input channels
//     H, W   : spatial height and width of `in`
//     weights: [C_out * C_in * K * K] convolution weights, oc-major
//     bias   : [C_out] per-output-channel bias (added once)
//     oc     : which output channel this call computes
//     y, x   : output pixel coordinates (same H,W as input -> "same" padding)
//   Returns the pre-activation convolution sum (caller applies relu()).
//
//   The channel/row-major indexing (`ch*H*W + row*W + col`) is spelled out here
//   ONCE and reused verbatim by both the CPU loop and the GPU tile read so the
//   two implementations cannot silently disagree on memory layout.
// ---------------------------------------------------------------------------
CNN_HD inline float conv_at(const float* in, int C_in, int H, int W,
                            const float* weights, const float* bias,
                            int oc, int y, int x) {
    float acc = bias[oc];                          // start from the channel bias
    // Base offset of this output channel's weight block: oc occupies
    // C_in*K*K contiguous floats.
    const int wbase = oc * C_in * CNN_KSIZE * CNN_KSIZE;
    for (int ic = 0; ic < C_in; ++ic) {            // sum over input channels
        const int in_ch = ic * H * W;              // start of channel ic in `in`
        const int w_ch  = wbase + ic * CNN_KSIZE * CNN_KSIZE;  // weights for (oc,ic)
        for (int ky = 0; ky < CNN_KSIZE; ++ky) {
            const int iy = y + ky - CNN_HALO;      // source row (may be off-image)
            if (iy < 0 || iy >= H) continue;       // zero-padding: skip out-of-range rows
            for (int kx = 0; kx < CNN_KSIZE; ++kx) {
                const int ix = x + kx - CNN_HALO;  // source col (may be off-image)
                if (ix < 0 || ix >= W) continue;   // zero-padding: skip out-of-range cols
                const float pixel = in[in_ch + iy * W + ix];
                const float wgt   = weights[w_ch + ky * CNN_KSIZE + kx];
                acc += wgt * pixel;                // multiply-accumulate (the CNN MAC)
            }
        }
    }
    return acc;
}

// ---------------------------------------------------------------------------
// maxpool2x2_at: compute one output pixel of a 2x2 stride-2 max-pool.
//   Down-samples H x W -> (H/2) x (W/2), keeping the strongest activation in
//   each 2x2 window. This is the standard CNN way to shrink spatial size while
//   preserving "did this feature fire anywhere near here?". Shared HD so CPU and
//   GPU pool identically.
//     in   : [C * H * W] feature stack ; c : channel ; oy,ox : output coords
//     H,W  : input spatial size (assumed even; our shapes are chosen even)
// ---------------------------------------------------------------------------
CNN_HD inline float maxpool2x2_at(const float* in, int c, int H, int W, int oy, int ox) {
    const int ch = c * H * W;         // start of channel c
    const int iy = oy * 2, ix = ox * 2;   // top-left source pixel of the 2x2 window
    float m = in[ch + iy * W + ix];
    float b = in[ch + iy * W + (ix + 1)];         if (b > m) m = b;
    float cc = in[ch + (iy + 1) * W + ix];        if (cc > m) m = cc;
    float d = in[ch + (iy + 1) * W + (ix + 1)];   if (d > m) m = d;
    return m;
}
