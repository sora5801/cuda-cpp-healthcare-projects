// ===========================================================================
// src/gnn.h  --  Shared (host + device) GNN / DTI primitives  (CPU/GPU parity)
// ---------------------------------------------------------------------------
// Project 7.2 : Drug-Target Interaction Prediction (GNN)
//
// WHAT THIS PROJECT COMPUTES  (reduced-scope teaching version -- see THEORY.md)
//   Drug-Target Interaction (DTI) prediction asks: will small molecule d bind
//   protein target p, and how strongly? Production systems (DeepPurpose,
//   TorchDrug) TRAIN a graph neural network (GNN) on the drug's molecular graph
//   and a transformer on the protein sequence, then score drug x protein pairs.
//
//   We implement the *inference* half of that pipeline with FIXED, DETERMINISTIC
//   weights (NOT trained -- see "HONESTY" below). This is deliberate: the goal is
//   to teach the CUDA data-flow of a GNN + pairwise scoring, not to reproduce a
//   trained model's accuracy. Concretely, for each drug graph we run:
//
//     1. MESSAGE PASSING (MPNN):  T rounds. In each round, every atom (node)
//        gathers the feature vectors of its bonded neighbours, sums them, pushes
//        the sum through a shared linear layer W (F x F) + bias, and applies
//        ReLU. This is the canonical GNN operation and the parallel bottleneck.
//     2. GRAPH POOLING:  sum the final node features into ONE drug embedding
//        (a length-F vector) -- "readout".
//     3. PROTEIN ENCODING:  project a length-F protein descriptor (here, a fixed
//        composition vector) through a linear layer into the SAME F-dim space.
//     4. DTI SCORE:  score(d,p) = sigmoid( dot(drug_emb, prot_emb) / F ). We
//        score EVERY drug x protein pair -> a dense D x P interaction matrix.
//
// WHY A GPU
//   Message passing is a GATHER over graph edges: output node i sums over its
//   neighbours. Each node is independent -> one GPU thread per (drug,node,round)
//   or, as we do it, one thread per node with the round loop inside. Pairwise
//   scoring is D*P independent dot products -> one thread per pair. Real virtual
//   screening scores MILLIONS of compounds x thousands of targets; the GPU's job
//   is throughput. See PATTERNS.md sec 1 ("gather" + "independent jobs").
//
// CPU/GPU PARITY (PATTERNS.md sec 2)
//   The per-element math -- the linear layer, ReLU, dot product, sigmoid -- lives
//   HERE as GNN_HD (__host__ __device__) inline functions. The CPU reference and
//   the GPU kernels both call these, so they run byte-for-byte identical math and
//   verification is near-exact (a tiny FP tolerance for FMA reordering only).
//
//   Keep CUDA-only constructs (no __global__) out of this header so the host
//   compiler (cl.exe) can include it from reference_cpu.cpp.
//
// HONESTY (CLAUDE.md sec 8, sec 13)
//   The weights are pseudo-random but FIXED (seeded), i.e. an UNTRAINED network.
//   The synthetic drugs/proteins are engineered so that a KNOWN pair scores
//   highest (an implanted "true" interaction), which is what we recover and
//   report -- this validates the *machinery*, not any clinical binding claim.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh.  BEFORE: kernels.cuh.
// ===========================================================================
#pragma once

#include <cmath>     // std::exp, std::tanh
#include <cstddef>   // std::size_t

// HD-macro idiom (PATTERNS.md sec 2): under nvcc (__CUDACC__ defined) these
// functions are compiled for BOTH host and device; under the plain host
// compiler the decorators simply vanish, so the same header is legal in .cpp.
#ifdef __CUDACC__
#define GNN_HD __host__ __device__
#else
#define GNN_HD
#endif

// ---------------------------------------------------------------------------
// Model dimensions. Kept small so the demo is fast and the numbers are
// inspectable; the code is written for any F / T (compile-time constants keep
// the inner loops unrollable and the register usage predictable).
//   F = node/embedding feature width (columns of the weight matrix W).
//   T = number of message-passing rounds (graph "receptive field" radius).
// ---------------------------------------------------------------------------
static const int GNN_F = 8;    // feature dimension (per node and per embedding)
static const int GNN_T = 2;    // message-passing rounds (2 => 2-hop neighbourhood)

// ---------------------------------------------------------------------------
// relu: the standard rectified-linear activation, max(0, x). Non-linearity is
// what lets stacked message-passing layers represent more than a single linear
// map. Braner-free form so it vectorizes cleanly on both host and device.
// ---------------------------------------------------------------------------
GNN_HD inline float gnn_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

// ---------------------------------------------------------------------------
// sigmoid: squashes a real score into (0,1) so it reads as an interaction
// "probability". We clamp the argument to avoid overflow of expf for large |z|
// (a standard numerically-stable guard); this keeps host and device identical.
// ---------------------------------------------------------------------------
GNN_HD inline float gnn_sigmoid(float z) {
    if (z >  30.0f) return 1.0f;      // expf(-30) ~ 1e-13 -> sigmoid ~ 1
    if (z < -30.0f) return 0.0f;      // symmetric guard on the low side
    return 1.0f / (1.0f + std::exp(-z));
}

// ---------------------------------------------------------------------------
// gnn_linear_relu: the shared MPNN update applied to ONE aggregated message.
//   out[c] = relu( bias[c] + sum_k in[k] * W[k*F + c] )     for c in [0,F)
//
//   in     : length-F aggregated neighbour features (the "message")
//   W      : F x F weight matrix, row-major (W[k*F + c] multiplies in[k] into
//            output channel c). Shared across all nodes and rounds (weight
//            tying) -- exactly how a real GNN layer works.
//   bias   : length-F bias added before the activation.
//   out    : length-F result (caller-owned).
//
// This is a tiny dense matrix-vector product + ReLU; writing it once here is
// what guarantees the CPU reference and the GPU kernel agree. Complexity O(F^2)
// per node per round.
// ---------------------------------------------------------------------------
GNN_HD inline void gnn_linear_relu(const float* in, const float* W,
                                   const float* bias, float* out) {
    for (int c = 0; c < GNN_F; ++c) {
        float acc = bias[c];                       // start from the bias term
        // Accumulate in a FIXED order (k = 0..F-1) so host and device sum the
        // same way; float add is non-associative, and a fixed order keeps the
        // result deterministic and cross-platform reproducible.
        for (int k = 0; k < GNN_F; ++k)
            acc += in[k] * W[k * GNN_F + c];
        out[c] = gnn_relu(acc);
    }
}

// ---------------------------------------------------------------------------
// gnn_dot: scaled dot product of a drug embedding and a protein embedding, the
// DTI logit. Dividing by F keeps the logit O(1) regardless of feature width (a
// simple stand-in for the learned bilinear/MLP head in a real DTI model).
//   Returns sum_c a[c]*b[c] / F.  O(F).
// ---------------------------------------------------------------------------
GNN_HD inline float gnn_dot(const float* a, const float* b) {
    float s = 0.0f;
    for (int c = 0; c < GNN_F; ++c) s += a[c] * b[c];
    return s / static_cast<float>(GNN_F);
}
