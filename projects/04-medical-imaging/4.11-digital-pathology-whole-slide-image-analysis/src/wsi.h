// ===========================================================================
// src/wsi.h  --  Shared (host + device) attention-MIL primitives for WSI slides
// ---------------------------------------------------------------------------
// Project 4.11 : Digital Pathology / Whole-Slide Image Analysis
//
// WHAT THIS PROJECT COMPUTES  (reduced-scope teaching version -- see THEORY.md
// "Where this sits in the real world" for the full production pipeline)
//   A whole-slide image (WSI) is a multi-gigapixel scan. Production tools (CLAM,
//   the standard baseline named in the catalog) turn one slide into a BAG of N
//   small tiles, run each tile through a pretrained CNN/ViT to get a D-dim
//   FEATURE VECTOR, and then classify the whole slide from that bag using
//   ATTENTION-BASED MULTIPLE-INSTANCE LEARNING (ABMIL / gated attention MIL).
//
//   We implement exactly that LAST stage -- the attention-MIL head -- on the GPU,
//   because it is the part that is a clean, self-contained GPU numerical kernel
//   (the CNN feature extractor is a cuDNN/PyTorch black box we deliberately do
//   NOT reimplement; instead the committed sample GIVES us the N x D features, as
//   if a frozen encoder had already run). Given a bag H = [h_0 ... h_{N-1}] of N
//   tile features (each D-dimensional), gated-attention MIL does four steps:
//
//     1. PROJECT   each tile through a tiny gated-attention network to a scalar
//                  attention LOGIT  e_i  (one number per tile).            [per-tile]
//     2. SOFTMAX   the logits over the N tiles -> attention WEIGHTS a_i>=0,
//                  sum a_i = 1  (which tiles the model "looks at").      [reduction]
//     3. POOL      the slide EMBEDDING z = sum_i a_i * h_i  (a weighted
//                  average of tile features).                    [weighted reduction]
//     4. CLASSIFY  slide logit s = w_c . z + b_c  ->  probability sigmoid(s).  [dot]
//
// WHY A GPU
//   A single 40x slide yields tens of thousands of tiles; a cohort (TCGA,
//   CAMELYON) is millions of tiles. Step 1 is embarrassingly parallel (one
//   thread per tile). Steps 2-3 are REDUCTIONS across tiles. This is the same
//   "independent per-item projection + reduction" shape as flagship 11.09
//   (k-means) and 1.12 (Tanimoto).
//
// THE ATTENTION NETWORK (gated attention, Ilse et al. 2018; used by CLAM)
//   For tile feature h (length D), with a hidden width M:
//       u = tanh( V h )            (V is M x D)   -- the "content" branch
//       g = sigmoid( U h )         (U is M x D)   -- the multiplicative "gate"
//       e = w . (u (elementwise*) g)   (w is length M) -- the attention logit
//   The gate lets the network suppress or amplify each hidden unit per tile;
//   this is what makes the attention SELECTIVE (a real diagnostic model puts
//   almost all its weight on the few tumor-bearing tiles).
//
// DETERMINISM TRICK  (identical idea to flagships 5.01 and 11.09)
//   The POOL step (3) is a scatter reduction: many tiles add their weighted
//   features into the same D accumulators. On the GPU that is atomicAdd, and a
//   FLOAT atomicAdd is order-dependent (float + is not associative) -> the sum
//   would vary run to run and would NOT match the CPU. We instead accumulate in
//   FIXED-POINT integers (atomicAdd on unsigned long long): integer adds commute,
//   so the GPU pooled embedding is bit-reproducible AND equals the CPU exactly.
//   Everything else (steps 1,2,4) is per-tile or a small host reduction, so it is
//   deterministic by construction.
//
//   All the per-tile math below is __host__ __device__ (WSI_HD) so the CPU
//   reference (reference_cpu.cpp) and the GPU kernels (kernels.cu) run the SAME
//   arithmetic and agree exactly. Keep CUDA-only types OUT of this header so the
//   host compiler can include it (CLAUDE.md PATTERNS.md section 2).
// ===========================================================================
#pragma once

#include <cmath>     // std::tanh, std::exp
#include <cstdint>   // fixed-width integer types

// WSI_HD expands to __host__ __device__ under nvcc, and to nothing under the
// plain host compiler (which has never heard of those decorators). This one
// macro is what lets a single definition compile for BOTH the CPU and the GPU.
#ifdef __CUDACC__
#define WSI_HD __host__ __device__
#else
#define WSI_HD
#endif

// Fixed dimensions of the tiny gated-attention head, chosen small so the whole
// model fits in registers/constant memory and the demo is legible. In a real
// CLAM model D is 512-1024 (the encoder's feature width) and M is 256-512.
//   FEAT_DIM   D : length of each tile feature vector.
//   ATTN_HIDDEN M: width of the attention network's hidden layer.
// These are compile-time constants so loops unroll and the code stays branch-free.
// (constexpr so these compile-time constants are usable in BOTH host and device
//  code -- a plain namespace-scope `static const` is a host global that nvcc will
//  not resolve inside a __device__ function.)
constexpr int FEAT_DIM    = 8;   // D: tile feature dimensionality (teaching size)
constexpr int ATTN_HIDDEN = 4;   // M: attention hidden width      (teaching size)

// Fixed-point scale for the deterministic POOL accumulation. Tile features and
// attention weights are bounded (features in a documented range, weights in
// [0,1] summing to 1), so a weighted feature a_i*h_i is small; 2^30 gives ~9
// significant decimal digits of headroom while the sum over N tiles stays far
// below the ~9.2e18 range of unsigned long long. We store a SIGNED fixed-point
// value in an unsigned container via two's-complement wraparound: adds still
// commute, and we reinterpret the final 64-bit sum as signed on the way out.
// (constexpr so it is a compile-time constant visible in device code.)
constexpr double WSI_FIXED_SCALE = 1073741824.0;  // 2^30

// -----------------------------------------------------------------------------
// wsi_quantize: real weighted-feature value -> fixed-point integer bucket.
//   We round to nearest (the +0.5 / -0.5 trick) so the CPU and GPU quantize a
//   given double to the exact same integer -- rounding must be identical on both
//   sides or the "exact match" verification would fail. The result is stored in
//   an unsigned long long via a signed->unsigned reinterpret so that negative
//   contributions (features can be negative) accumulate correctly under
//   commutative two's-complement addition.
// -----------------------------------------------------------------------------
WSI_HD inline unsigned long long wsi_quantize(double weighted_value) {
    const double scaled = weighted_value * WSI_FIXED_SCALE;
    // Round half away from zero, matching on host and device.
    const double rounded = (scaled >= 0.0) ? (scaled + 0.5) : (scaled - 0.5);
    const long long as_signed = static_cast<long long>(rounded);
    return static_cast<unsigned long long>(as_signed);  // bit-preserving reinterpret
}

// -----------------------------------------------------------------------------
// wsi_dequantize: fixed-point 64-bit accumulator -> real double.
//   Inverse of wsi_quantize for a SUM: reinterpret the unsigned accumulator as a
//   signed integer (undoing the two's-complement store) and divide by the scale.
// -----------------------------------------------------------------------------
WSI_HD inline double wsi_dequantize(unsigned long long fixed_sum) {
    const long long as_signed = static_cast<long long>(fixed_sum);
    return static_cast<double>(as_signed) / WSI_FIXED_SCALE;
}

// -----------------------------------------------------------------------------
// The attention MODEL parameters. These are the (frozen, pretrained) weights of
// the gated-attention head. In production they are LEARNED; here we ship a fixed,
// documented set (see reference_cpu.cpp / make_synthetic.py) so the demo is
// reproducible and the result is interpretable. Row-major layout throughout.
//   V : [M x D] "content" projection      (attention hidden = tanh(V h))
//   U : [M x D] "gate" projection         (gate            = sigmoid(U h))
//   w : [M]     attention combiner        (logit           = w . (content*gate))
//   w_c : [D]   slide classifier weights  (slide logit     = w_c . z + b_c)
//   b_c :       slide classifier bias
// -----------------------------------------------------------------------------
struct AttnParams {
    double V[ATTN_HIDDEN * FEAT_DIM];   // content branch weights, row-major [M][D]
    double U[ATTN_HIDDEN * FEAT_DIM];   // gate branch weights,    row-major [M][D]
    double w[ATTN_HIDDEN];              // attention combiner weights
    double w_c[FEAT_DIM];               // slide-level classifier weights
    double b_c;                         // slide-level classifier bias
};

// -----------------------------------------------------------------------------
// wsi_attention_logit: compute ONE tile's attention logit e_i.
//   This is step 1 of the pipeline for a single tile and is the per-item work
//   that both the CPU loop and the GPU (one-thread-per-tile) kernel call.
//   Parameters:
//     h : pointer to this tile's D feature values (FEAT_DIM long)
//     p : the attention model parameters (V, U, w)
//   Returns the scalar logit e = w . (tanh(V h) * sigmoid(U h)).
//   Complexity: O(M*D) multiply-adds -- tiny and fully in registers.
// -----------------------------------------------------------------------------
WSI_HD inline double wsi_attention_logit(const double* h, const AttnParams& p) {
    double e = 0.0;  // accumulates the final attention logit
    // One hidden unit m at a time: form content u_m and gate g_m, combine, weight.
    for (int m = 0; m < ATTN_HIDDEN; ++m) {
        double content_pre = 0.0;  // (V h)_m before the tanh nonlinearity
        double gate_pre    = 0.0;  // (U h)_m before the sigmoid nonlinearity
        // Dot the m-th rows of V and U with the tile feature h.
        for (int d = 0; d < FEAT_DIM; ++d) {
            const double hd = h[d];
            content_pre += p.V[m * FEAT_DIM + d] * hd;
            gate_pre    += p.U[m * FEAT_DIM + d] * hd;
        }
        const double u = std::tanh(content_pre);              // content in (-1,1)
        const double g = 1.0 / (1.0 + std::exp(-gate_pre));   // gate in (0,1) = sigmoid
        e += p.w[m] * (u * g);   // gated content, weighted by the combiner w_m
    }
    return e;
}

// -----------------------------------------------------------------------------
// wsi_slide_logit: classify a pooled slide embedding z into a slide-level logit.
//   Step 4 of the pipeline: s = w_c . z + b_c. Shared by CPU and GPU so the final
//   number matches exactly. Probability is sigmoid(s), computed by the caller.
//     z : pooled slide embedding (FEAT_DIM long)
//     p : model parameters (w_c, b_c)
// -----------------------------------------------------------------------------
WSI_HD inline double wsi_slide_logit(const double* z, const AttnParams& p) {
    double s = p.b_c;
    for (int d = 0; d < FEAT_DIM; ++d) s += p.w_c[d] * z[d];
    return s;
}

// wsi_sigmoid: the logistic squashing function, used to turn a logit into a
// probability in (0,1). Defined here so both sides use the identical formula.
WSI_HD inline double wsi_sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}
