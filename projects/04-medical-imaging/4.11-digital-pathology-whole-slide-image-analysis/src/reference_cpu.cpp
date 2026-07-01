// ===========================================================================
// src/reference_cpu.cpp  --  Loader, frozen model, serial attention-MIL reference
// ---------------------------------------------------------------------------
// Project 4.11 : Digital Pathology / Whole-Slide Image Analysis
// Compiled by the HOST compiler only. All per-tile arithmetic lives in wsi.h so
// this serial baseline and the GPU kernels (kernels.cu) compute the identical
// forward pass -- main.cu compares them and expects an EXACT match (fixed-point
// pooling makes that possible).
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>       // std::exp
#include <fstream>     // std::ifstream
#include <limits>      // std::numeric_limits
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// default_params: the frozen gated-attention head.
//   These numbers are HAND-SET (not learned) so the demo is reproducible and
//   interpretable. The design goal: hidden unit 0 of the attention network is a
//   crude "tumor detector" -- it fires (large positive content AND open gate)
//   when a tile's first two features are high, which is exactly the pattern the
//   synthetic "tumor" tiles carry (see make_synthetic.py). The slide classifier
//   w_c then reads the pooled embedding's tumor-associated features. The result:
//   attention concentrates on the planted tumor tiles and the slide probability
//   rises with the tumor fraction -- a faithful miniature of how CLAM behaves.
// ---------------------------------------------------------------------------
AttnParams default_params() {
    AttnParams p{};   // zero-initialise every field, then fill the non-zeros

    // --- Attention "content" matrix V [M x D] --------------------------------
    // Hidden unit 0 = tumor detector: strong positive weight on features 0 and 1
    // (the markers the synthetic tumor tiles elevate), small elsewhere. Units 1-3
    // give the head a little extra capacity but are weak, so unit 0 dominates.
    // Row m, column d is V[m*FEAT_DIM + d].
    p.V[0 * FEAT_DIM + 0] = 1.50;  p.V[0 * FEAT_DIM + 1] = 1.20;
    p.V[1 * FEAT_DIM + 2] = 0.40;
    p.V[2 * FEAT_DIM + 3] = 0.30;
    p.V[3 * FEAT_DIM + 4] = 0.20;

    // --- Attention "gate" matrix U [M x D] -----------------------------------
    // The gate for unit 0 opens (sigmoid -> ~1) when features 0/1 are high, so it
    // MULTIPLICATIVELY reinforces the content branch on tumor tiles and stays
    // near 0.5 (half-open) on background tiles. Other gates are mildly positive.
    p.U[0 * FEAT_DIM + 0] = 1.00;  p.U[0 * FEAT_DIM + 1] = 0.80;
    p.U[1 * FEAT_DIM + 2] = 0.20;
    p.U[2 * FEAT_DIM + 3] = 0.20;
    p.U[3 * FEAT_DIM + 4] = 0.20;

    // --- Attention combiner w [M] --------------------------------------------
    // Weight the tumor-detector hidden unit heavily (the others barely count).
    // A LARGE w[0] widens the logit gap between tumor and background tiles, which
    // makes the softmax SHARP -- attention piles onto the few tumor tiles instead
    // of spreading thin over the bag. That concentration is the whole point of
    // attention MIL (a diffuse attention map would tell a pathologist nothing).
    p.w[0] = 8.00;  p.w[1] = 0.20;  p.w[2] = 0.20;  p.w[3] = 0.20;

    // --- Slide classifier w_c [D], bias b_c ----------------------------------
    // The pooled embedding z is (attention-weighted) mostly the tumor tiles'
    // features, whose first two entries are high (~0.9). w_c reads those out; the
    // bias shifts the decision boundary so a tumor slide lands above p=0.5 and a
    // tumor-free slide (all-background bag) lands below it.
    p.w_c[0] = 2.00;  p.w_c[1] = 1.50;  p.w_c[2] = -0.20;
    p.b_c = -2.80;

    return p;
}

// ---------------------------------------------------------------------------
// load_slide: parse the text slide-bag format (see data/README.md).
//   Header: "N D label". D must equal FEAT_DIM (the model's feature width) or we
//   throw -- a shape mismatch is a bug, not something to silently truncate.
// ---------------------------------------------------------------------------
SlideBag load_slide(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open slide file: " + path);

    SlideBag bag;
    int D = 0, label = -1;
    if (!(in >> bag.N >> D >> label) || bag.N <= 0)
        throw std::runtime_error("bad header (expected 'N D label') in " + path);
    if (D != FEAT_DIM)
        throw std::runtime_error("feature dim in file does not equal FEAT_DIM in " + path);

    bag.true_label = label;
    bag.has_true_label = (label == 0 || label == 1);

    bag.features.resize(static_cast<std::size_t>(bag.N) * FEAT_DIM);
    for (std::size_t i = 0; i < bag.features.size(); ++i)
        if (!(in >> bag.features[i]))
            throw std::runtime_error("slide file truncated in " + path);
    return bag;
}

// ---------------------------------------------------------------------------
// mil_forward_cpu: the serial attention-MIL forward pass (ground truth).
//   The four steps mirror wsi.h's project/softmax/pool/classify exactly, and use
//   the SAME fixed-point pooling as the GPU so the two agree bit-for-bit.
// ---------------------------------------------------------------------------
MilResult mil_forward_cpu(const SlideBag& bag, const AttnParams& p) {
    const int N = bag.N;
    MilResult r;
    r.attn.resize(N);
    r.embedding.assign(FEAT_DIM, 0.0);

    // --- Step 1: per-tile attention logits e_i ------------------------------
    // Also track the maximum logit, needed for a numerically-stable softmax
    // (subtracting the max prevents exp() overflow -- standard practice).
    std::vector<double> logits(N);
    double max_logit = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < N; ++i) {
        const double e = wsi_attention_logit(&bag.features[static_cast<std::size_t>(i) * FEAT_DIM], p);
        logits[i] = e;
        if (e > max_logit) max_logit = e;
    }

    // --- Step 2: softmax over tiles -> attention weights a_i ------------------
    // a_i = exp(e_i - max) / sum_j exp(e_j - max). The shift by max_logit is
    // mathematically a no-op (it cancels) but keeps every exp() argument <= 0.
    double denom = 0.0;
    for (int i = 0; i < N; ++i) {
        const double ex = std::exp(logits[i] - max_logit);
        r.attn[i] = ex;      // temporarily hold the unnormalised numerator
        denom += ex;
    }
    for (int i = 0; i < N; ++i) r.attn[i] /= denom;   // normalise so sum a_i = 1

    // --- Step 3: fixed-point weighted pool  z = sum_i a_i * h_i --------------
    // We accumulate in 64-bit fixed-point integers, EXACTLY as the GPU kernel's
    // atomicAdd does, so the pooled embedding matches the device bit-for-bit
    // (integer addition is associative/commutative; float addition is not).
    std::vector<unsigned long long> fixed(FEAT_DIM, 0ull);
    for (int i = 0; i < N; ++i) {
        const double a = r.attn[i];
        const double* h = &bag.features[static_cast<std::size_t>(i) * FEAT_DIM];
        for (int d = 0; d < FEAT_DIM; ++d)
            fixed[d] += wsi_quantize(a * h[d]);   // same quantiser as the GPU
    }
    for (int d = 0; d < FEAT_DIM; ++d)
        r.embedding[d] = wsi_dequantize(fixed[d]);

    // --- Step 4: classify the pooled embedding ------------------------------
    r.slide_logit  = wsi_slide_logit(r.embedding.data(), p);
    r.probability  = wsi_sigmoid(r.slide_logit);

    // --- Extra: which tile did the model look at most? ----------------------
    // Deterministic argmax with a strict '>' (ties resolve to the lowest index).
    int best = 0;
    for (int i = 1; i < N; ++i) if (r.attn[i] > r.attn[best]) best = i;
    r.top_tile = best;

    return r;
}
