// ===========================================================================
// src/reference_cpu.h  --  Model definition + shared per-element math + CPU path
// ---------------------------------------------------------------------------
// Project 7.1 : Diagnostic Imaging Classifier   (REDUCED-SCOPE teaching version)
//
// WHAT THIS PROJECT COMPUTES
//   A small CONVOLUTIONAL NEURAL NETWORK doing INFERENCE (a forward pass) that
//   classifies each grayscale "medical image patch" into one of two classes:
//       class 0 = "normal"   (roughly flat / low-contrast tissue)
//       class 1 = "lesion"   (a bright blob -- a stand-in for a nodule/mass).
//
//   The network is the classic minimal image classifier, exactly the compute-
//   dominant path named in the catalog deep-dive (conv backbone -> classifier):
//
//       input image  [H x W]              (1 channel, e.g. 16x16)
//         |  CONV 2D  (F filters, KxK, "valid" padding)   <-- the GPU-heavy step
//         v
//       feature maps [F x (H-K+1) x (W-K+1)]
//         |  + per-filter BIAS, then ReLU
//         v
//       2x2 MAX-POOL  (stride 2)          <-- spatial downsample
//         v
//       FLATTEN -> DENSE (fully connected) -> 2 logits
//         v
//       SOFTMAX -> probabilities -> ARGMAX -> predicted class
//
//   We run a whole BATCH of images (independent samples), score every one, and
//   report per-image predictions plus batch accuracy against known labels.
//
// WHY A GPU  (the lesson this file sets up)
//   The CONVOLUTION dominates: for each of B*F*CONV_H*CONV_W output pixels we do
//   K*K multiply-adds. That is a huge number of INDEPENDENT dot products -> a
//   textbook data-parallel workload. Real frameworks hand this to cuDNN / tensor
//   cores; here we hand-write the naive "one GPU thread per output pixel"
//   convolution (kernels.cu) so nothing is a black box (CLAUDE.md section 6),
//   and THEORY.md explains how cuDNN / im2col+GEMM would do it faster.
//
// REDUCED SCOPE (honest labeling, CLAUDE.md section 13)
//   We do INFERENCE with FIXED, SYNTHETIC weights (a hand-designed lesion
//   detector), not TRAINING. There is no backprop, no cuDNN, no mixed precision,
//   no real DICOM images. THEORY.md "Where this sits in the real world"
//   describes the full training pipeline (MONAI, cuDNN, NCCL, TensorRT). The
//   point here is the forward-pass math and its GPU mapping.
//
// THE SHARED-CORE IDIOM (PATTERNS.md section 2)
//   The per-output-pixel convolution math lives in ONE inline function,
//   conv_pixel(), marked __host__ __device__. reference_cpu.cpp calls it from a
//   host loop; kernels.cu calls it from one GPU thread. Same code -> the GPU and
//   CPU results are byte-for-byte identical, so verification is EXACT (tol = 0).
//
//   This header is pure C++ plus the HD macro (no <<<>>>, no __global__), so the
//   host compiler (cl.exe) can include it for reference_cpu.cpp AND nvcc can
//   include it for kernels.cu.
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh, kernels.cu, main.cu.
// ===========================================================================
#pragma once

#include <cmath>      // std::exp
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// HD: the "host+device" decorator macro (PATTERNS.md section 2).
//   When compiled by nvcc (__CUDACC__ defined) it expands to `__host__
//   __device__`, so the SAME function is emitted for both the CPU and the GPU.
//   When compiled by the plain host compiler for reference_cpu.cpp, __CUDACC__
//   is NOT defined, so it expands to nothing and the decorators simply vanish.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ---------------------------------------------------------------------------
// Model geometry (compile-time constants so both paths agree exactly).
//   Kept small so the demo is instant and the numbers are hand-checkable, but
//   the code is written for arbitrary sizes -- bump these for a heavier run.
// ---------------------------------------------------------------------------
static constexpr int IMG_H   = 16;  // input image height  (pixels)
static constexpr int IMG_W   = 16;  // input image width   (pixels)
static constexpr int NUM_F   = 4;   // number of conv filters (output channels)
static constexpr int KERNEL  = 3;   // conv kernel is KERNEL x KERNEL, "valid"
static constexpr int CONV_H  = IMG_H - KERNEL + 1;   // feature-map height (14)
static constexpr int CONV_W  = IMG_W - KERNEL + 1;   // feature-map width  (14)
static constexpr int POOL    = 2;   // 2x2 max-pool, stride 2
static constexpr int POOL_H  = CONV_H / POOL;        // pooled height (7)
static constexpr int POOL_W  = CONV_W / POOL;        // pooled width  (7)
static constexpr int FLAT    = NUM_F * POOL_H * POOL_W;  // dense input size (196)
static constexpr int NUM_CLS = 2;   // output classes: 0=normal, 1=lesion

// Convenience sizes (in floats) used by both the CPU and GPU paths.
static constexpr int IMG_SIZE  = IMG_H * IMG_W;        // pixels per image (256)
static constexpr int FMAP_SIZE = NUM_F * CONV_H * CONV_W;  // conv output per image
static constexpr int CONV_WSZ  = NUM_F * KERNEL * KERNEL;  // conv weight count

// ---------------------------------------------------------------------------
// Weights: the fixed, synthetic network parameters (shared by CPU + GPU).
//   In a trained model these come from backprop; here they are a hand-designed
//   "lesion detector" produced by make_synthetic.py and loaded from the sample
//   file (or a built-in fallback). Layout is row-major and documented per field.
// ---------------------------------------------------------------------------
struct Weights {
    // Convolution filters: NUM_F filters, each KERNEL*KERNEL taps, row-major.
    //   conv_w[(f*KERNEL + ky)*KERNEL + kx]  = tap (ky,kx) of filter f.
    std::vector<float> conv_w;   // [NUM_F * KERNEL * KERNEL]
    std::vector<float> conv_b;   // [NUM_F]  one bias per filter

    // Dense (fully connected) layer: maps FLAT features -> NUM_CLS logits.
    //   dense_w[c*FLAT + j] = weight from flattened feature j to class c.
    std::vector<float> dense_w;  // [NUM_CLS * FLAT]
    std::vector<float> dense_b;  // [NUM_CLS]
};

// ---------------------------------------------------------------------------
// A batch of images to classify, plus their ground-truth labels (for accuracy).
// ---------------------------------------------------------------------------
struct Dataset {
    int n = 0;                    // number of images in the batch
    std::vector<float> images;    // [n * IMG_SIZE] row-major, pixel in [0,1]
    std::vector<int>   labels;    // [n] ground-truth class (0 or 1); -1 if unknown
};

// ===========================================================================
// THE SHARED PER-ELEMENT MATH  (HD -> identical on CPU and GPU)
// ===========================================================================

// conv_pixel: compute ONE convolution-output pixel for image `img`, filter `f`,
//   at feature-map location (oy, ox), then add the filter bias and apply ReLU.
//   This is the single hottest operation in the whole network and the unit of
//   GPU parallelism: kernels.cu assigns exactly one thread to one (img,f,oy,ox).
//
//   img    : pointer to this image's [IMG_SIZE] pixels (row-major)
//   w      : full conv weight array [CONV_WSZ]
//   b      : full conv bias array  [NUM_F]
//   f,oy,ox: which filter and which output location
//   returns: ReLU( bias_f + sum_{ky,kx} w[f,ky,kx] * img[oy+ky, ox+kx] )
//
//   "valid" convolution: the KxK window sits fully inside the image, so the top-
//   left input pixel is (oy,ox) and there is no padding -> no boundary checks.
HD inline float conv_pixel(const float* img, const float* w, const float* b,
                           int f, int oy, int ox) {
    float acc = b[f];                              // start from the filter bias
    for (int ky = 0; ky < KERNEL; ++ky) {          // walk the KxK window rows
        for (int kx = 0; kx < KERNEL; ++kx) {      // ... and columns
            float pix = img[(oy + ky) * IMG_W + (ox + kx)];   // input pixel
            float tap = w[(f * KERNEL + ky) * KERNEL + kx];   // filter tap
            acc += tap * pix;                      // multiply-accumulate
        }
    }
    // ReLU: max(0, x). The nonlinearity that lets a CNN represent non-linear
    // decision boundaries; without it, stacked linear layers collapse to one.
    return acc > 0.0f ? acc : 0.0f;
}

// pool_pixel: 2x2 max-pool of one feature map at pooled location (py, px).
//   Takes the maximum activation over the 2x2 block starting at (2*py, 2*px) in
//   the conv feature map `fmap` (which is CONV_H x CONV_W, row-major). Max-pool
//   keeps the strongest local response (translation tolerance) and shrinks the
//   spatial size 4x, cheapening the dense layer.
HD inline float pool_pixel(const float* fmap, int py, int px) {
    int y0 = py * POOL;                            // top row of the 2x2 block
    int x0 = px * POOL;                            // left col of the 2x2 block
    float m = fmap[y0 * CONV_W + x0];              // seed with the first element
    for (int dy = 0; dy < POOL; ++dy)
        for (int dx = 0; dx < POOL; ++dx) {
            float v = fmap[(y0 + dy) * CONV_W + (x0 + dx)];
            if (v > m) m = v;                      // running maximum
        }
    return m;
}

// dense_logit: the fully-connected layer for one output class `c`.
//   logit_c = dense_b[c] + sum_j dense_w[c, j] * feat[j], over the FLAT pooled
//   features. This is a plain dot product; softmax turns the two logits into a
//   probability. Shared so CPU and GPU compute the same value.
HD inline float dense_logit(const float* feat, const float* dw, const float* db, int c) {
    float acc = db[c];
    for (int j = 0; j < FLAT; ++j) acc += dw[c * FLAT + j] * feat[j];
    return acc;
}

// argmax2 + softmax_pos1: turn two logits into a prediction + a probability.
//   argmax2 returns the index (0 or 1) of the larger logit -> predicted class.
//   softmax_pos1 returns P(class 1) via the numerically-stable softmax
//   (subtract the max before exp so we never overflow). Both are HD so the demo
//   prints identical probabilities whether the logits came from CPU or GPU.
HD inline int argmax2(float l0, float l1) { return l1 > l0 ? 1 : 0; }

HD inline float softmax_pos1(float l0, float l1) {
    float m  = l0 > l1 ? l0 : l1;                  // stability shift
    float e0 = std::exp(l0 - m);
    float e1 = std::exp(l1 - m);
    return e1 / (e0 + e1);                         // P(class = 1)
}

// ===========================================================================
// CPU reference declarations (defined in reference_cpu.cpp)
// ===========================================================================

// Load the model weights + image batch from the sample text file (data/README.md
// documents the layout). Throws std::runtime_error if the file is unreadable so
// the caller (main.cu) can fall back to a built-in synthetic problem.
void load_sample(const std::string& path, Weights& w, Dataset& d);

// Build the built-in synthetic fallback (mirrors make_synthetic.py) so the
// program runs with zero arguments and the demo is fully offline.
void make_builtin(Weights& w, Dataset& d);

// CPU forward pass over the whole batch. Fills, per image:
//   logits[i*NUM_CLS + c] = class-c logit, and pred[i] = argmax class.
// This is the trusted baseline the GPU is checked against (tol = 0, exact).
void classify_cpu(const Weights& w, const Dataset& d,
                  std::vector<float>& logits, std::vector<int>& pred);
