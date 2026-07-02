// ===========================================================================
// src/reference_cpu.h  --  Fundus image, fixed CNN model, and CPU forward pass
// ---------------------------------------------------------------------------
// Project 7.18 : Retinal Fundus AI Screening
//
// WHAT THIS PROJECT COMPUTES
//   The FORWARD PASS (inference) of a small convolutional neural network that
//   grades a colour retinal fundus photograph for diabetic-retinopathy (DR)
//   severity on the clinical 0..4 scale (none / mild / moderate / severe /
//   proliferative). The network is the same skeleton every vision CNN has --
//   stacked 3x3 convolutions + ReLU + max-pool, then a linear classifier and a
//   softmax -- but with FIXED, hand-designed filter weights so the demo is fully
//   deterministic and needs no training run or ML framework. See cnn_core.h for
//   the per-pixel math and THEORY.md for the science and the GPU mapping.
//
// WHY A GPU
//   The convolution layers dominate the cost: every output pixel of every
//   feature map is an independent K x K x C_in multiply-accumulate. That is
//   embarrassingly parallel -> one GPU thread per output pixel, with the input
//   tile staged in shared memory (kernels.cu). Real screening pipelines push
//   millions of 2048x2048 images/year through such backbones, so throughput is
//   an operational, GPU-bound concern (catalog deep-dive).
//
//   Pure C++ header (NO CUDA) so kernels.cu can reuse FundusImage and CnnModel.
//
// READ THIS AFTER: cnn_core.h (the shared conv/relu/pool math).
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "cnn_core.h"   // CNN_C_IN, CNN_C1, ... and the HD math helpers

// ---------------------------------------------------------------------------
// FundusImage: a loaded colour fundus photo (or synthetic stand-in).
//   Stored CHANNEL-MAJOR then row-major: data[c*H*W + y*W + x] is channel c
//   (0=R,1=G,2=B), row y, col x. Channel-major (a.k.a. "planar" / NCHW without
//   the N) is the layout cuDNN and most CNN kernels prefer because it makes each
//   channel a contiguous plane -> coalesced reads when a thread block sweeps one
//   channel. Pixel values are normalized to [0,1] floats.
// ---------------------------------------------------------------------------
struct FundusImage {
    int C = 0;                    // channels (always CNN_C_IN = 3 here)
    int H = 0;                    // height in pixels
    int W = 0;                    // width in pixels
    std::vector<float> data;      // [C*H*W] normalized pixels, channel-major
    int label = -1;               // ground-truth DR grade if known (-1 = unknown)
};

// ---------------------------------------------------------------------------
// CnnModel: all the fixed parameters of the teaching network.
//   Two conv layers (weights + biases) and one fully-connected classifier head.
//   Shapes follow the constants in cnn_core.h:
//     conv1_w : [C1 * C_IN * K * K]   conv1_b : [C1]
//     conv2_w : [C2 * C1   * K * K]   conv2_b : [C2]
//     fc_w    : [NUM_CLASSES * C2]    fc_b    : [NUM_CLASSES]
//   The classifier consumes the C2 global-average-pooled features (one number
//   per feature map) and produces NUM_CLASSES logits.
// ---------------------------------------------------------------------------
struct CnnModel {
    std::vector<float> conv1_w, conv1_b;
    std::vector<float> conv2_w, conv2_b;
    std::vector<float> fc_w,    fc_b;
};

// ---------------------------------------------------------------------------
// ForwardResult: everything the forward pass produces, so we can compare the
//   CPU and GPU paths field by field and print a deterministic report.
//     logits    : [NUM_CLASSES] raw classifier outputs (pre-softmax)
//     probs     : [NUM_CLASSES] softmax probabilities (sum to 1)
//     pred_grade: argmax(probs) = predicted DR severity 0..4
//     cam       : [Hf*Wf] class-activation map over the layer-2 feature grid,
//                 a Grad-CAM-style heatmap of "where the winning class looked"
//                 (lesion localisation, catalog "Grad-CAM"). Hf,Wf are the
//                 pooled spatial dims after two 2x2 pools.
//     cam_h, cam_w : the CAM grid size.
// ---------------------------------------------------------------------------
struct ForwardResult {
    std::vector<float> logits;
    std::vector<float> probs;
    int pred_grade = -1;
    std::vector<float> cam;
    int cam_h = 0, cam_w = 0;
};

// Load a fundus image from the text sample format (see data/README.md):
//   header line "C H W label" then C*H*W whitespace-separated floats in [0,1].
FundusImage load_fundus(const std::string& path);

// Build the FIXED teaching model (deterministic, hand-designed filters).
//   Deterministic so CPU and GPU see identical weights and the demo is
//   reproducible. THEORY.md section 7 describes what a trained model changes.
CnnModel make_fixed_model();

// CPU reference forward pass: the trusted baseline the GPU is checked against.
//   Runs conv1->relu->pool -> conv2->relu->pool -> global-avg-pool -> fc ->
//   softmax, and fills `out`. Uses the SAME cnn_core.h helpers as the GPU.
void forward_cpu(const FundusImage& img, const CnnModel& model, ForwardResult& out);

// Numeric-stable softmax of `logits` into `probs` (subtract max first). Shared
// by both paths so the reported probabilities match exactly given equal logits.
void softmax(const std::vector<float>& logits, std::vector<float>& probs);
