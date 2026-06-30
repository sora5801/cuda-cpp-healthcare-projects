// ===========================================================================
// src/vep_model.h  --  The ONE TRUE variant-effect model, shared CPU<->GPU
// ---------------------------------------------------------------------------
// Project 3.19 : Variant Effect / Pathogenicity Prediction
//
// WHY THIS HEADER EXISTS  (PATTERNS.md sec 2: the __host__ __device__ core)
//   The single most important idiom in this repo: the *per-variant math* lives
//   in ONE header as __host__ __device__ inline functions, so the CPU reference
//   (reference_cpu.cpp, compiled by cl.exe / g++) and the GPU kernel (kernels.cu,
//   compiled by nvcc) call the EXACT SAME code. That makes verification a
//   near-bit-for-bit match (~1e-12 in double) instead of a fuzzy approximation:
//   if CPU and GPU disagree, it is a real bug, not a numerical artifact.
//
//   The HD-macro trick: under nvcc the symbol __CUDACC__ is defined, so VEP_HD
//   expands to `__host__ __device__` (the function is compiled for BOTH the CPU
//   and every GPU architecture). Under the plain host compiler those decorators
//   do not exist, so VEP_HD expands to nothing. Keep NO CUDA-only types here
//   (no __global__, no dim3) so the host compiler can include the file cleanly.
//
// WHAT THE MODEL IS  (read ../THEORY.md for the full derivation)
//   This is a deliberately small, FIXED-WEIGHT 1-D convolutional neural network
//   that plays the role of a genomic "variant effect" scorer (the toy analogue
//   of AlphaMissense / Enformer / a DNA language model). It takes a one-hot
//   encoded DNA sequence window and returns a single scalar "pathogenicity
//   logit" in (0,1) after a sigmoid.
//
//   The pipeline for ONE window x  (length L bases, 4 channels A/C/G/T):
//     1. CONV1D  : K filters of width W slide over the window, each producing
//                  (L-W+1) pre-activations; we ReLU them.   (the "motif scan")
//     2. GLOBAL MAX POOL : per filter, take the max activation over all
//                  positions -> a K-vector "does this motif appear anywhere?".
//     3. DENSE   : a linear layer maps the K-vector to one logit z.
//     4. SIGMOID : sigma(z) in (0,1) = a calibrated-looking "pathogenic" prob.
//
//   THE VARIANT EFFECT itself is a DELTA SCORE (in-silico mutagenesis):
//       effect = score(ALT window) - score(REF window)
//   exactly how real tools turn a per-sequence model into a per-variant call
//   (Enformer's "variant effect = ref/alt difference"; ESM-1v's log-odds ratio).
//   A positive delta => the alternate allele looks more pathogenic than the
//   reference. This is the headline number the demo ranks.
//
//   The weights are SYNTHETIC and FIXED (see init_model()), engineered so the
//   toy network responds to a couple of planted "deleterious" motifs. Nothing
//   here is trained on real data and NONE of it is clinically meaningful -- it
//   teaches the GPU *inference pattern*, not real genetics (CLAUDE.md sec 8).
//
// READ THIS BEFORE: reference_cpu.h, kernels.cuh.
// ===========================================================================
#pragma once

#include <cstdint>
#include <cmath>   // exp() -- on the host side; nvcc provides device exp() too

// --- The HD decorator macro (see the header comment for the reasoning) ------
#ifdef __CUDACC__
#define VEP_HD __host__ __device__   // nvcc: build for host AND device
// VEP_UNROLL: nvcc understands `#pragma unroll`; emit it via _Pragma so the
// directive is wrapped in a macro. The host compiler (cl.exe) does NOT know this
// pragma and would warn C4068 -- so under the plain compiler VEP_UNROLL is empty.
#define VEP_UNROLL _Pragma("unroll")
#else
#define VEP_HD                        // host compiler: decorators don't exist
#define VEP_UNROLL                    // host compiler: no unroll pragma (avoids C4068)
#endif

// ---------------------------------------------------------------------------
// Fixed model geometry. These are COMPILE-TIME constants so:
//   * the inner loops can be unrolled by the compiler, and
//   * the whole weight set fits in GPU __constant__ memory (a fixed-size struct,
//     uploaded once, broadcast to every thread -- like the query in 1.12).
// Sizes are tiny on purpose: this is a teaching model, not a foundation model.
// ---------------------------------------------------------------------------
constexpr int VEP_BASES   = 4;    // DNA alphabet channels: A, C, G, T
constexpr int VEP_WINDOW  = 21;   // sequence context length L (bases). Odd, so
                                  // the variant sits at the exact centre index 10.
constexpr int VEP_KERNELS = 8;    // K = number of convolution filters (motifs)
constexpr int VEP_KWIDTH  = 5;    // W = filter width in bases (a 5-mer motif)
constexpr int VEP_CONVOUT = VEP_WINDOW - VEP_KWIDTH + 1;  // = 17 conv positions

// One-hot base codes. A window is stored as VEP_WINDOW int8 codes in [0,3];
// code 0..3 selects which of the 4 input channels is "hot" at that position.
//   index of the variant base within a window = VEP_WINDOW/2 (integer) = 10.
constexpr int VEP_CENTER = VEP_WINDOW / 2;

// ---------------------------------------------------------------------------
// VepModel: the complete fixed-weight network as a flat, trivially-copyable
// struct (so cudaMemcpyToSymbol can shovel it into __constant__ memory in one
// shot). Layout, with double precision throughout for CPU/GPU parity:
//   conv_w[k][c][w] : weight of filter k for input channel c at offset w
//   conv_b[k]       : bias added to filter k's pre-activation
//   dense_w[k]      : dense-layer weight on filter k's pooled activation
//   dense_b         : dense-layer bias (the single output neuron's bias)
// The arrays are fixed-size, so sizeof(VepModel) is a compile-time constant.
// ---------------------------------------------------------------------------
struct VepModel {
    double conv_w[VEP_KERNELS][VEP_BASES][VEP_KWIDTH];  // [K][C][W]
    double conv_b[VEP_KERNELS];                          // [K]
    double dense_w[VEP_KERNELS];                         // [K]
    double dense_b;                                       // scalar
};

// ---------------------------------------------------------------------------
// vep_relu / vep_sigmoid  --  the activation functions, in double precision.
//   ReLU(x)    = max(x, 0)          (the conv non-linearity)
//   sigma(x)   = 1 / (1 + e^{-x})   (squashes the final logit into (0,1))
// We use a NUMERICALLY STABLE sigmoid: for x >= 0 compute 1/(1+e^{-x}) (the
// exponent is <= 0, so e^{-x} <= 1 and cannot overflow); for x < 0 compute
// e^{x}/(1+e^{x}) (again e^{x} <= 1). The naive single-branch form would
// overflow exp() for large |x| and make CPU/GPU diverge near the extremes.
// Branch-on-sign is deterministic and identical on both sides.
// ---------------------------------------------------------------------------
VEP_HD inline double vep_relu(double x) {
    return x > 0.0 ? x : 0.0;
}

VEP_HD inline double vep_sigmoid(double x) {
    if (x >= 0.0) {
        const double e = exp(-x);          // e^{-x} in (0,1]  -> no overflow
        return 1.0 / (1.0 + e);
    } else {
        const double e = exp(x);           // e^{x}  in (0,1)  -> no overflow
        return e / (1.0 + e);
    }
}

// ---------------------------------------------------------------------------
// vep_score_window  --  run ONE one-hot window through the whole network.
//
//   m      : the fixed model (read-only). On the GPU this is the __constant__
//            copy; on the CPU it is a plain struct -- same bytes, same math.
//   window : VEP_WINDOW base codes, each in [0,3] (A=0,C=1,G=2,T=3). The code
//            at position p says which input channel is hot there; this is the
//            one-hot encoding WITHOUT materialising the 4xL matrix (we just
//            index the conv weight by the hot channel -- the other 3 channels
//            contribute 0, so they are skipped entirely; that is the whole
//            point of one-hot sparsity and it keeps the inner loop cheap).
//   returns: sigma(logit) in (0,1), the model's "pathogenic-looking" score.
//
// Complexity: O(K * (L-W+1) * W) multiply-adds per window. With the constants
// above that is 8 * 17 * 5 = 680 FMAs -- trivial for one window, but we do it
// for 2 windows (ref+alt) per variant and for MANY variants, which is exactly
// the batched-inference bottleneck the GPU is for (THEORY "GPU mapping").
//
// This function is the SHARED CORE: reference_cpu.cpp loops it over all
// variants on the CPU; kernels.cu calls it once per thread on the GPU. Because
// it is the very same code, the two results agree to ~1e-12 (THEORY "verify").
// ---------------------------------------------------------------------------
VEP_HD inline double vep_score_window(const VepModel& m, const int8_t* window) {
    // GLOBAL-MAX-POOL accumulator: pooled[k] will hold the max ReLU activation
    // of filter k over all conv positions. Initialise to 0.0 because ReLU >= 0,
    // so 0 is a valid floor (a filter that never fires pools to 0).
    double pooled[VEP_KERNELS];
    VEP_UNROLL
    for (int k = 0; k < VEP_KERNELS; ++k) pooled[k] = 0.0;

    // --- CONV1D + ReLU + global-max-pool, fused into one sweep ---------------
    // For each filter k and each valid start position p, dot the W-wide patch
    // of the (one-hot) window with filter k's weights, add the bias, ReLU, and
    // keep the running max. Doing pooling inline avoids storing the (K x 17)
    // activation map -- a small but real memory saving that matters on the GPU
    // where this all lives in registers.
    VEP_UNROLL
    for (int k = 0; k < VEP_KERNELS; ++k) {
        double best = 0.0;                         // max-pool over positions
        for (int p = 0; p < VEP_CONVOUT; ++p) {    // conv start position
            double acc = m.conv_b[k];              // pre-activation starts at bias
            VEP_UNROLL
            for (int w = 0; w < VEP_KWIDTH; ++w) {
                // The base at window position p+w selects the hot input channel
                // c. Only that channel is 1 in the one-hot column, so the dot
                // product over channels collapses to a single weight lookup.
                const int c = window[p + w];       // hot channel in [0,3]
                acc += m.conv_w[k][c][w];          // 1.0 * weight (one-hot)
            }
            const double a = vep_relu(acc);        // filter k's response at p
            if (a > best) best = a;                // running global max
        }
        pooled[k] = best;
    }

    // --- DENSE (linear) layer: pooled K-vector -> single logit z -------------
    double z = m.dense_b;
    VEP_UNROLL
    for (int k = 0; k < VEP_KERNELS; ++k) z += m.dense_w[k] * pooled[k];

    // --- SIGMOID: squash the logit into a (0,1) pseudo-probability -----------
    return vep_sigmoid(z);
}

// ---------------------------------------------------------------------------
// vep_variant_effect  --  the DELTA SCORE for one variant (the teaching point).
//   ref_win / alt_win : the two windows that differ ONLY at the centre base
//                       (VEP_CENTER). In-silico mutagenesis: same context,
//                       flip the one variant position from reference to alternate.
//   returns: score(alt) - score(ref). > 0 means "alternate looks more
//            pathogenic"; this single number is what real variant-effect tools
//            (Enformer, ESM-1v log-odds ratio, AlphaMissense) report per variant.
//   Side effects: none (pure function). Complexity: two window forward passes.
// ---------------------------------------------------------------------------
VEP_HD inline double vep_variant_effect(const VepModel& m,
                                        const int8_t* ref_win,
                                        const int8_t* alt_win) {
    const double s_ref = vep_score_window(m, ref_win);
    const double s_alt = vep_score_window(m, alt_win);
    return s_alt - s_ref;                          // the per-variant delta score
}
