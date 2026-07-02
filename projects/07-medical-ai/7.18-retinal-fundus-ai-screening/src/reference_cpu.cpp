// ===========================================================================
// src/reference_cpu.cpp  --  Loader, fixed model, and the trusted CPU forward pass
// ---------------------------------------------------------------------------
// Project 7.18 : Retinal Fundus AI Screening
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- plain readable loops, no parallelism -- so that
//   when the GPU and CPU agree we believe the GPU. It calls the SAME per-pixel
//   math (cnn_core.h) that kernels.cu calls, so the only differences are float
//   summation order (see THEORY.md section 5).
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, cnn_core.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>       // std::exp
#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// load_fundus: read the tiny text sample (data/sample/fundus_sample.txt).
//   Format (see data/README.md):
//     line 1 :  C H W label      (label = ground-truth DR grade, -1 if unknown)
//     then   :  C*H*W floats in [0,1], channel-major then row-major.
//   Throws on any malformed input so the demo fails loudly, never silently on
//   an empty image.
// ---------------------------------------------------------------------------
FundusImage load_fundus(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open fundus file: " + path);
    FundusImage img;
    if (!(in >> img.C >> img.H >> img.W >> img.label))
        throw std::runtime_error("bad header (expected: C H W label) in " + path);
    if (img.C != CNN_C_IN)
        throw std::runtime_error("this teaching model expects 3 (RGB) channels in " + path);
    if (img.H <= 0 || img.W <= 0)
        throw std::runtime_error("non-positive image dimensions in " + path);
    const std::size_t n = static_cast<std::size_t>(img.C) * img.H * img.W;
    img.data.resize(n);
    for (std::size_t i = 0; i < n; ++i)
        if (!(in >> img.data[i]))
            throw std::runtime_error("pixel data truncated in " + path);
    return img;
}

// ---------------------------------------------------------------------------
// det_fill: deterministically fill a weight buffer with small, reproducible
//   values drawn from a cheap hash of the flat index. We use a fixed, closed-
//   form pseudo-pattern (NOT std::rand) so EVERY run -- and both the CPU and GPU
//   -- see byte-identical weights. This stands in for "trained parameters"; a
//   real model would load these from a checkpoint (THEORY.md section 7).
//     values land in roughly [-scale, +scale].
// ---------------------------------------------------------------------------
static void det_fill(std::vector<float>& w, int count, float scale, unsigned seed) {
    w.resize(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        // A tiny integer hash (splitmix-style) -> map to [-1,1] -> scale.
        unsigned h = (static_cast<unsigned>(i) + seed) * 2654435761u;
        h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
        // h in [0, 2^32); (h / 2^32) in [0,1); *2-1 -> [-1,1).
        const float u = (static_cast<float>(h) / 4294967296.0f) * 2.0f - 1.0f;
        w[static_cast<std::size_t>(i)] = u * scale;
    }
}

// ---------------------------------------------------------------------------
// make_fixed_model: build the deterministic teaching network.
//   The first conv layer's weights are seeded so that, combined, the feature
//   maps respond to edges, blobs, and colour contrast -- the visual cues a
//   clinician (and a trained DR model) keys on: microaneurysms (small red
//   blobs), haemorrhages, and exudates (bright spots). We keep the weights
//   SMALL so activations stay in a sane range through two layers without batch
//   normalisation (which we omit for clarity). See THEORY.md section 3.
// ---------------------------------------------------------------------------
CnnModel make_fixed_model() {
    CnnModel m;
    const int K2 = CNN_KSIZE * CNN_KSIZE;
    // Conv1: C1 output maps, each C_IN*K*K weights. Seed 1001.
    det_fill(m.conv1_w, CNN_C1 * CNN_C_IN * K2, 0.25f, 1001u);
    det_fill(m.conv1_b, CNN_C1,                 0.05f, 2002u);
    // Conv2: C2 output maps, each C1*K*K weights. Seed 3003.
    det_fill(m.conv2_w, CNN_C2 * CNN_C1 * K2,   0.15f, 3003u);
    det_fill(m.conv2_b, CNN_C2,                 0.05f, 4004u);
    // FC head: NUM_CLASSES x C2 weights mapping the C2 pooled features -> logits.
    det_fill(m.fc_w,    CNN_NUM_CLASSES * CNN_C2, 0.40f, 5005u);
    det_fill(m.fc_b,    CNN_NUM_CLASSES,          0.10f, 6006u);
    return m;
}

// ---------------------------------------------------------------------------
// conv_relu_pool_cpu: run one "conv -> ReLU -> 2x2 max-pool" block on the CPU.
//   in    : [C_in * H * W] input stack
//   out   : resized to [C_out * (H/2) * (W/2)], the pooled feature stack
//   The intermediate conv+ReLU map is built full-size, then pooled. This mirrors
//   what the GPU does in two kernels (conv+relu, then pool) so we can verify
//   each stage. Uses conv_at() and relu() and maxpool2x2_at() from cnn_core.h.
// ---------------------------------------------------------------------------
static void conv_relu_pool_cpu(const std::vector<float>& in, int C_in, int H, int W,
                               const std::vector<float>& weights,
                               const std::vector<float>& bias, int C_out,
                               std::vector<float>& out, int& Hout, int& Wout) {
    // 1) Convolution + ReLU into a full-resolution activation stack.
    std::vector<float> act(static_cast<std::size_t>(C_out) * H * W);
    for (int oc = 0; oc < C_out; ++oc)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const float pre = conv_at(in.data(), C_in, H, W,
                                          weights.data(), bias.data(), oc, y, x);
                act[static_cast<std::size_t>(oc) * H * W + y * W + x] = relu(pre);
            }
    // 2) 2x2 stride-2 max-pool -> halve each spatial dimension.
    Hout = H / 2; Wout = W / 2;
    out.assign(static_cast<std::size_t>(C_out) * Hout * Wout, 0.0f);
    for (int c = 0; c < C_out; ++c)
        for (int oy = 0; oy < Hout; ++oy)
            for (int ox = 0; ox < Wout; ++ox)
                out[static_cast<std::size_t>(c) * Hout * Wout + oy * Wout + ox] =
                    maxpool2x2_at(act.data(), c, H, W, oy, ox);
}

// ---------------------------------------------------------------------------
// softmax: numerically stable exp-normalise (subtract max before exp so we
//   never overflow exp() on a large logit). Shared by CPU and GPU report paths.
// ---------------------------------------------------------------------------
void softmax(const std::vector<float>& logits, std::vector<float>& probs) {
    probs.resize(logits.size());
    float mx = logits[0];
    for (float v : logits) if (v > mx) mx = v;
    double sum = 0.0;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        const double e = std::exp(static_cast<double>(logits[i] - mx));
        probs[i] = static_cast<float>(e);
        sum += e;
    }
    for (std::size_t i = 0; i < probs.size(); ++i)
        probs[i] = static_cast<float>(probs[i] / sum);
}

// ---------------------------------------------------------------------------
// forward_cpu: the whole reference forward pass, stage by stage.
//   conv1->relu->pool  (3->6 maps, H/2 x W/2)
//   conv2->relu->pool  (6->12 maps, H/4 x W/4)
//   global-average-pool -> 12 features
//   fc + bias -> 5 logits ; softmax -> probs ; argmax -> grade
//   Grad-CAM-style CAM: weight each layer-2 feature map by the winning class's
//   FC weight for that map, sum, ReLU -> a coarse "where the class looked" map.
// ---------------------------------------------------------------------------
void forward_cpu(const FundusImage& img, const CnnModel& model, ForwardResult& out) {
    // --- Layer 1: conv (3->C1) + ReLU + pool -------------------------------
    std::vector<float> f1; int H1, W1;
    conv_relu_pool_cpu(img.data, CNN_C_IN, img.H, img.W,
                       model.conv1_w, model.conv1_b, CNN_C1, f1, H1, W1);

    // --- Layer 2: conv (C1->C2) + ReLU + pool ------------------------------
    std::vector<float> f2; int H2, W2;
    conv_relu_pool_cpu(f1, CNN_C1, H1, W1,
                       model.conv2_w, model.conv2_b, CNN_C2, f2, H2, W2);

    // --- Global average pool: one number per layer-2 feature map -----------
    // gap[c] = mean over the H2 x W2 grid of feature map c. This is the standard
    // "collapse spatial dims into a feature vector" step before the classifier.
    std::vector<float> gap(CNN_C2, 0.0f);
    const int spatial = H2 * W2;
    for (int c = 0; c < CNN_C2; ++c) {
        double s = 0.0;
        for (int i = 0; i < spatial; ++i)
            s += f2[static_cast<std::size_t>(c) * spatial + i];
        gap[c] = static_cast<float>(s / spatial);
    }

    // --- Fully-connected classifier head: 5 logits -------------------------
    out.logits.assign(CNN_NUM_CLASSES, 0.0f);
    for (int k = 0; k < CNN_NUM_CLASSES; ++k) {
        float acc = model.fc_b[k];
        for (int c = 0; c < CNN_C2; ++c)
            acc += model.fc_w[static_cast<std::size_t>(k) * CNN_C2 + c] * gap[c];
        out.logits[k] = acc;
    }

    // --- Softmax + argmax --------------------------------------------------
    softmax(out.logits, out.probs);
    int best = 0;
    for (int k = 1; k < CNN_NUM_CLASSES; ++k)
        if (out.probs[k] > out.probs[best]) best = k;
    out.pred_grade = best;

    // --- Grad-CAM-style class activation map (lesion localisation) ---------
    // CAM(y,x) = ReLU( sum_c fc_w[best,c] * f2(c,y,x) ). Highlights the regions
    // of the pooled feature grid that pushed the winning class up. A didactic
    // stand-in for true Grad-CAM (which uses gradients); THEORY.md section 6.
    out.cam_h = H2; out.cam_w = W2;
    out.cam.assign(static_cast<std::size_t>(H2) * W2, 0.0f);
    for (int y = 0; y < H2; ++y)
        for (int x = 0; x < W2; ++x) {
            float s = 0.0f;
            for (int c = 0; c < CNN_C2; ++c)
                s += model.fc_w[static_cast<std::size_t>(best) * CNN_C2 + c]
                     * f2[static_cast<std::size_t>(c) * spatial + y * W2 + x];
            out.cam[static_cast<std::size_t>(y) * W2 + x] = relu(s);
        }
}
