// ===========================================================================
// src/reference_cpu.cpp  --  The plain-C++ CNN forward pass we trust
// ---------------------------------------------------------------------------
// Project 7.1 : Diagnostic Imaging Classifier   (REDUCED-SCOPE teaching version)
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. It is written to be
//   OBVIOUSLY correct -- one image at a time, one layer after another, no
//   parallelism -- so that when the GPU and CPU agree we believe the GPU.
//
//   The heavy per-element math (conv_pixel / pool_pixel / dense_logit) is NOT
//   duplicated here: it lives once in reference_cpu.h as __host__ __device__
//   inline functions and is shared with kernels.cu. That guarantees the two
//   paths run byte-for-byte identical arithmetic -> exact verification (tol=0).
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h. Compare against kernels.cu (the GPU twin).
// ===========================================================================
#include "reference_cpu.h"
#include "util/io.hpp"        // util::read_floats

#include <cstddef>
#include <stdexcept>

// ---------------------------------------------------------------------------
// forward_one_image: run the whole network on a SINGLE image, filling that
//   image's two class logits. Broken out so both the loader-fed path and the
//   built-in path share it, and so the layer order is easy to read.
//
//   Layers, in order (all math from the shared reference_cpu.h helpers):
//     1. CONV + bias + ReLU  -> fmap  [NUM_F * CONV_H * CONV_W]
//     2. 2x2 MAX-POOL         -> feat  [FLAT] (== NUM_F * POOL_H * POOL_W)
//     3. DENSE dot products   -> logits[NUM_CLS]
//   The caller applies softmax/argmax on the logits.
// ---------------------------------------------------------------------------
static void forward_one_image(const Weights& w, const float* img, float* logits) {
    // -- Layer 1: convolution -> feature maps -------------------------------
    // fmap[(f*CONV_H + oy)*CONV_W + ox] holds filter f's activation at (oy,ox).
    // Complexity: NUM_F * CONV_H * CONV_W output pixels, each K*K multiply-adds.
    float fmap[FMAP_SIZE];
    for (int f = 0; f < NUM_F; ++f)
        for (int oy = 0; oy < CONV_H; ++oy)
            for (int ox = 0; ox < CONV_W; ++ox)
                fmap[(f * CONV_H + oy) * CONV_W + ox] =
                    conv_pixel(img, w.conv_w.data(), w.conv_b.data(), f, oy, ox);

    // -- Layer 2: 2x2 max-pool + flatten ------------------------------------
    // feat[(f*POOL_H + py)*POOL_W + px] is the max over one 2x2 block. The very
    // same indexing scheme is used by the dense layer's weight layout.
    float feat[FLAT];
    for (int f = 0; f < NUM_F; ++f) {
        const float* fmap_f = &fmap[f * CONV_H * CONV_W];   // this filter's map
        for (int py = 0; py < POOL_H; ++py)
            for (int px = 0; px < POOL_W; ++px)
                feat[(f * POOL_H + py) * POOL_W + px] = pool_pixel(fmap_f, py, px);
    }

    // -- Layer 3: dense -> class logits -------------------------------------
    for (int c = 0; c < NUM_CLS; ++c)
        logits[c] = dense_logit(feat, w.dense_w.data(), w.dense_b.data(), c);
}

// ---------------------------------------------------------------------------
// classify_cpu: forward pass over the whole batch (the public reference).
//   logits: resized to n*NUM_CLS ; pred: resized to n (argmax class per image).
//   Complexity: O(n * NUM_F * CONV_H * CONV_W * K*K) for the conv (dominant),
//   plus O(n * NUM_CLS * FLAT) for the dense layer.
// ---------------------------------------------------------------------------
void classify_cpu(const Weights& w, const Dataset& d,
                  std::vector<float>& logits, std::vector<int>& pred) {
    logits.assign(static_cast<std::size_t>(d.n) * NUM_CLS, 0.0f);
    pred.assign(static_cast<std::size_t>(d.n), 0);
    for (int i = 0; i < d.n; ++i) {
        const float* img = &d.images[static_cast<std::size_t>(i) * IMG_SIZE];
        float* lg = &logits[static_cast<std::size_t>(i) * NUM_CLS];
        forward_one_image(w, img, lg);
        pred[i] = argmax2(lg[0], lg[1]);   // predicted class = larger logit
    }
}

// ---------------------------------------------------------------------------
// load_sample: parse the whitespace-separated sample file (data/README.md).
//   Layout (all floats; ints stored as floats and rounded):
//       n
//       conv_w  [NUM_F*KERNEL*KERNEL]
//       conv_b  [NUM_F]
//       dense_w [NUM_CLS*FLAT]
//       dense_b [NUM_CLS]
//       then, repeated n times:
//           label                    (0 or 1; -1 if unknown)
//           pixels [IMG_H*IMG_W]      (row-major, in [0,1])
//   We validate the total count so a truncated file fails loudly instead of
//   reading garbage. Throws std::runtime_error on any problem.
// ---------------------------------------------------------------------------
void load_sample(const std::string& path, Weights& w, Dataset& d) {
    std::vector<float> v = util::read_floats(path);   // throws if unreadable
    std::size_t p = 0;                                // read cursor

    auto need = [&](std::size_t k) {
        if (p + k > v.size())
            throw std::runtime_error("sample file too short at offset " +
                                     std::to_string(p));
    };

    need(1);
    d.n = static_cast<int>(v[p++] + 0.5f);            // round to nearest int
    if (d.n <= 0) throw std::runtime_error("sample: n must be positive");

    need(CONV_WSZ);
    w.conv_w.assign(v.begin() + p, v.begin() + p + CONV_WSZ); p += CONV_WSZ;
    need(NUM_F);
    w.conv_b.assign(v.begin() + p, v.begin() + p + NUM_F);    p += NUM_F;
    need(static_cast<std::size_t>(NUM_CLS) * FLAT);
    w.dense_w.assign(v.begin() + p, v.begin() + p + NUM_CLS * FLAT); p += NUM_CLS * FLAT;
    need(NUM_CLS);
    w.dense_b.assign(v.begin() + p, v.begin() + p + NUM_CLS); p += NUM_CLS;

    d.images.assign(static_cast<std::size_t>(d.n) * IMG_SIZE, 0.0f);
    d.labels.assign(static_cast<std::size_t>(d.n), -1);
    for (int i = 0; i < d.n; ++i) {
        need(1);
        // Round toward nearest int; labels are 0, 1, or -1 (unknown).
        float lab = v[p++];
        d.labels[i] = static_cast<int>(lab + (lab < 0.0f ? -0.5f : 0.5f));
        need(IMG_SIZE);
        for (int k = 0; k < IMG_SIZE; ++k)
            d.images[static_cast<std::size_t>(i) * IMG_SIZE + k] = v[p++];
    }
}

// ---------------------------------------------------------------------------
// make_builtin: the zero-argument fallback problem, identical to what
//   scripts/make_synthetic.py writes to data/sample. Keeping the generator in
//   BOTH places (C++ here, Python there) lets the demo run offline while the
//   committed sample stays human-inspectable. See make_synthetic.py for the
//   design rationale of the weights (a hand-built center-blob detector).
//
//   The batch has 4 images with KNOWN labels so batch accuracy is meaningful:
//     img 0,1 : a bright 4x4 blob in the center      -> label 1 (lesion)
//     img 2,3 : flat/low-contrast tissue             -> label 0 (normal)
// ---------------------------------------------------------------------------
void make_builtin(Weights& w, Dataset& d) {
    // --- Conv filters: 4 hand-designed detectors --------------------------
    // Filter 0 is a Laplacian-like center-surround (+8 center, -1 ring) that
    // fires strongly on a bright blob. Filters 1..3 are simple edge/blur taps
    // so the feature vector has some variety (as a real conv layer would).
    w.conv_w.assign(CONV_WSZ, 0.0f);
    w.conv_b.assign(NUM_F, 0.0f);
    const float lap[9] = {-1,-1,-1, -1, 8,-1, -1,-1,-1};   // filter 0
    const float sx [9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};   // filter 1: Sobel-x
    const float sy [9] = {-1,-2,-1,  0, 0, 0,  1, 2, 1};   // filter 2: Sobel-y
    const float bl [9] = { 1, 1, 1,  1, 1, 1,  1, 1, 1};   // filter 3: box blur
    for (int k = 0; k < 9; ++k) {
        w.conv_w[0 * 9 + k] = lap[k];
        w.conv_w[1 * 9 + k] = sx[k];
        w.conv_w[2 * 9 + k] = sy[k];
        w.conv_w[3 * 9 + k] = bl[k] * (1.0f / 9.0f);       // normalized blur
    }

    // --- Dense layer: weight the LAPLACIAN (filter 0) pooled features -------
    // A lesion (bright blob) makes filter 0's activations large, so class 1 is
    // the sum of filter-0 pooled features and class 0 is (roughly) its negation.
    w.dense_w.assign(static_cast<std::size_t>(NUM_CLS) * FLAT, 0.0f);
    w.dense_b.assign(NUM_CLS, 0.0f);
    for (int j = 0; j < POOL_H * POOL_W; ++j) {            // filter-0 block only
        w.dense_w[1 * FLAT + j] =  0.25f;   // class 1 (lesion): reward blob energy
        w.dense_w[0 * FLAT + j] = -0.25f;   // class 0 (normal): penalize it
    }
    w.dense_b[0] = 0.5f;   // small prior toward "normal" so a flat image -> 0

    // --- Images: 4 samples with known labels ------------------------------
    d.n = 4;
    d.images.assign(static_cast<std::size_t>(d.n) * IMG_SIZE, 0.05f);   // dim bg
    d.labels.assign(d.n, 0);

    auto put_blob = [&](int i, float bg, float fg) {
        float* im = &d.images[static_cast<std::size_t>(i) * IMG_SIZE];
        for (int k = 0; k < IMG_SIZE; ++k) im[k] = bg;
        for (int y = 6; y < 10; ++y)                        // centered 4x4 blob
            for (int x = 6; x < 10; ++x) im[y * IMG_W + x] = fg;
    };
    put_blob(0, 0.10f, 0.95f); d.labels[0] = 1;   // strong lesion
    put_blob(1, 0.15f, 0.80f); d.labels[1] = 1;   // weaker lesion
    for (int k = 0; k < IMG_SIZE; ++k)             // img 2: flat normal
        d.images[static_cast<std::size_t>(2) * IMG_SIZE + k] = 0.20f;
    d.labels[2] = 0;
    for (int k = 0; k < IMG_SIZE; ++k)             // img 3: gentle gradient
        d.images[static_cast<std::size_t>(3) * IMG_SIZE + k] =
            0.10f + 0.20f * (static_cast<float>(k % IMG_W) / IMG_W);
    d.labels[3] = 0;
}
