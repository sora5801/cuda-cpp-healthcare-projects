// ===========================================================================
// src/reference_cpu.h  --  Slide bag + attention-MIL CPU reference (pure C++)
// ---------------------------------------------------------------------------
// Project 4.11 : Digital Pathology / Whole-Slide Image Analysis
//
// Pure C++ (NO CUDA): compiled by cl.exe / g++. The per-tile math lives in
// wsi.h (shared __host__ __device__ so CPU and GPU agree exactly). This header
// declares the loaded slide, the RESULT bundle, the loader, and the serial
// attention-MIL reference that main.cu trusts as ground truth. kernels.cu reuses
// SlideBag + these declarations, so the GPU path and CPU path share one data type.
//
// READ THIS AFTER: wsi.h. Then kernels.cuh, then main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "wsi.h"   // FEAT_DIM, ATTN_HIDDEN, AttnParams, per-tile math

// ---------------------------------------------------------------------------
// SlideBag: one whole-slide image reduced to a BAG of tile feature vectors.
//   In production these features come from a frozen CNN/ViT encoder run over
//   each 224x224 tile; here they are provided by the sample file (as if the
//   encoder had already run -- we deliberately do not reimplement the encoder).
//     N        : number of tiles kept after tissue detection (the bag size).
//     features : [N * FEAT_DIM] tile features, row-major (tile i -> row i).
//     true_label / has_true_label : optional ground-truth slide label (0/1) that
//                the sample can carry so the demo can report accuracy; not used
//                by the computation itself.
// ---------------------------------------------------------------------------
struct SlideBag {
    int N = 0;                       // number of tiles in the bag
    std::vector<double> features;    // [N * FEAT_DIM] row-major tile features
    int true_label = -1;             // optional ground truth (0/1); -1 = unknown
    bool has_true_label = false;     // whether true_label is meaningful
};

// ---------------------------------------------------------------------------
// MilResult: everything the attention-MIL forward pass produces for a slide.
//   attn         : [N] attention weights a_i (>=0, sum to 1) -- the heat map.
//   embedding    : [FEAT_DIM] pooled slide embedding z = sum_i a_i * h_i.
//   slide_logit  : scalar s = w_c . z + b_c.
//   probability  : sigmoid(s) in (0,1) -- the slide's predicted tumor score.
//   top_tile     : index of the single highest-attention tile (the model's
//                  "most suspicious" region -- what a pathologist would review).
// ---------------------------------------------------------------------------
struct MilResult {
    std::vector<double> attn;        // [N] attention weights
    std::vector<double> embedding;   // [FEAT_DIM] pooled slide embedding
    double slide_logit = 0.0;        // pre-sigmoid slide score
    double probability = 0.0;        // sigmoid(slide_logit)
    int    top_tile = -1;            // argmax attention tile index
};

// ---------------------------------------------------------------------------
// default_params: the frozen gated-attention head shipped with this teaching
//   project. Returned by value (it is tiny). The SAME parameters are used by the
//   CPU reference and the GPU kernels, so the two forward passes are identical.
//   make_synthetic.py documents how the sample features were built to interact
//   with these weights (one hidden unit acts as a "tumor detector").
// ---------------------------------------------------------------------------
AttnParams default_params();

// load_slide: read a slide bag from the text format (see data/README.md):
//   line 1 : "N D label"   (label -1 if unknown; D MUST equal FEAT_DIM)
//   next N lines : D feature values per tile.
// Throws std::runtime_error on a missing/short/ill-shaped file so demos fail loudly.
SlideBag load_slide(const std::string& path);

// mil_forward_cpu: the SERIAL reference forward pass (the trusted baseline).
//   Runs the four steps (logits -> softmax -> fixed-point pool -> classify) using
//   the shared wsi.h math and the SAME fixed-point pooling the GPU uses, so the
//   result matches the GPU bit-for-bit. Fills and returns a MilResult.
//     bag : the input slide bag (N tiles x FEAT_DIM)
//     p   : the frozen attention model
MilResult mil_forward_cpu(const SlideBag& bag, const AttnParams& p);
