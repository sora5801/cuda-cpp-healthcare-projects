// ===========================================================================
// src/admet_core.h  --  The ONE-TRUE per-element ADMET math (CPU/GPU parity)
// ---------------------------------------------------------------------------
// Project 1.16 : ADMET / Toxicity Prediction  (reduced-scope teaching version)
//
// WHY THIS HEADER EXISTS  (PATTERNS.md sec.2 -- the "__host__ __device__ core")
//   The single most useful idiom in this repo: put the per-element physics in
//   ONE inline function marked `__host__ __device__`, so the CPU reference and
//   the GPU kernel run *byte-for-byte identical math*. Verification then becomes
//   an exact check (CPU == GPU to ~machine precision) instead of a fuzzy one.
//
//   reference_cpu.cpp includes this via the plain host compiler; kernels.cu
//   includes it via nvcc. The FOO_HD macro expands to `__host__ __device__`
//   under nvcc and to nothing under the host compiler -- so the SAME source
//   line compiles for both targets. Keep CUDA-only constructs OUT of this file
//   (no __global__, no <<<>>>); only the per-element scalar math lives here.
//
// THE MODEL  (see ../THEORY.md sec."The math")
//   A reduced-scope, teaching stand-in for a multi-task ADMET predictor. Each
//   molecule is a fixed-length real DESCRIPTOR vector x (length D). Each of the
//   M toxicity ENDPOINTS (e.g. hERG block, Ames mutagenicity, hepatotoxicity)
//   is a LOGISTIC-REGRESSION head: a weight vector w_t (length D) plus bias b_t.
//   The predicted probability that molecule i is "positive" (toxic / fails) for
//   endpoint t is the logistic sigmoid of the linear score:
//
//       z_{i,t} = b_t + sum_{d=0..D-1}  w_{t,d} * x_{i,d}        (the "logit")
//       p_{i,t} = sigma(z_{i,t}) = 1 / (1 + exp(-z_{i,t}))       in [0,1]
//
//   This linear head is exactly the classical baseline that production graph
//   neural networks (Chemprop / ADMET-AI, see README "Prior art") are measured
//   against; the GPU mapping (one independent dot-product per (molecule,endpoint)
//   pair) is the same shape a real multi-task head uses, just with the learned
//   message-passing features replaced by precomputed descriptors. See THEORY
//   sec."Where this sits in the real world" for the honest gap.
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cu (both call admet_logit/sigmoid).
// ===========================================================================
#pragma once

#include <cmath>     // std::exp  (host); nvcc maps this to the device exp() too

// ---------------------------------------------------------------------------
// HD: the host/device decorator. Under nvcc (__CUDACC__ defined) it marks the
// function as callable from BOTH host and device code; under the plain host
// compiler the CUDA keywords do not exist, so it expands to nothing.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// Problem dimensions are COMPILE-TIME constants so:
//   * the per-endpoint weight block fits in fixed-size GPU constant memory, and
//   * the inner dot-product loop can be fully unrolled by the compiler.
// D = descriptor length, M = number of toxicity endpoints (multi-task heads).
// These are deliberately small (teaching scale); production descriptors are
// hundreds-to-thousands long and models are GNNs (THEORY "real world").
constexpr int ADMET_D = 64;    // descriptor components per molecule
constexpr int ADMET_M = 12;    // toxicity endpoints (mirrors Tox21's 12 assays)

// ---------------------------------------------------------------------------
// admet_dot: the plain (non-fused) dot product  w . x  over D components.
//
//   WHY A HAND-ROLLED, LEFT-TO-RIGHT SUM (and not std::inner_product / FMA)?
//   Floating-point addition is NOT associative, and a fused multiply-add (FMA)
//   rounds once where a separate multiply-then-add rounds twice -- so the GPU
//   (which fuses aggressively) and the host compiler can disagree in the last
//   bits unless we pin down the EXACT operation sequence. By summing strictly
//   left-to-right in double precision with explicit `acc += w*x` (two roundings,
//   same order on both sides) we make CPU and GPU agree to ~1e-12. THEORY
//   sec."Numerical considerations" expands on this; PATTERNS.md sec.4 calls it
//   the machine-precision tolerance class.
//
//   acc is double: D ~ 64 products of O(1) magnitude sum without precision loss.
//   Returns the logit's linear part (the caller adds the bias).
// ---------------------------------------------------------------------------
HD inline double admet_dot(const double* w,   // [D] one endpoint's weights
                           const double* x,    // [D] one molecule's descriptor
                           int d)              // D (passed so this stays generic)
{
    double acc = 0.0;
    for (int k = 0; k < d; ++k) {
        acc += w[k] * x[k];   // two roundings (mul, add), strict left-to-right
    }
    return acc;
}

// ---------------------------------------------------------------------------
// admet_sigmoid: the logistic squashing function sigma(z) = 1/(1+e^-z).
//
//   It maps any real logit z to a probability in (0,1). We use the numerically
//   stable two-branch form so a large |z| never overflows exp():
//     * z >= 0 : 1/(1+e^-z)            (e^-z is in (0,1], no overflow)
//     * z <  0 : e^z/(1+e^z)           (e^z  is in (0,1), no overflow)
//   The naive single-branch 1/(1+exp(-z)) would compute exp(+big) for very
//   negative z and overflow to +inf -> NaN. Both branches are algebraically
//   identical, so CPU and GPU pick the SAME branch for the same z and agree.
// ---------------------------------------------------------------------------
HD inline double admet_sigmoid(double z) {
    if (z >= 0.0) {
        double e = std::exp(-z);     // in (0,1]
        return 1.0 / (1.0 + e);
    } else {
        double e = std::exp(z);      // in (0,1)
        return e / (1.0 + e);
    }
}

// ---------------------------------------------------------------------------
// admet_predict: the full per-(molecule,endpoint) prediction = sigmoid(logit).
//   This is THE function both the CPU reference and the GPU kernel call, so the
//   two implementations are identical by construction.
//     w : [D] endpoint t's weights      x : [D] molecule i's descriptor
//     b : endpoint t's bias             d : D
//   Returns p in [0,1] = predicted probability molecule i is positive for t.
// ---------------------------------------------------------------------------
HD inline double admet_predict(const double* w, const double* x, double b, int d) {
    return admet_sigmoid(admet_dot(w, x, d) + b);
}

// ---------------------------------------------------------------------------
// admet_flagged: turn a probability into a deterministic 0/1 "flag" by an
// inclusive threshold. We reduce over these INTEGER flags (counts per endpoint)
// rather than summing the floating-point probabilities, because integer adds
// commute -> the GPU's parallel/atomic reduction is bit-for-bit reproducible
// and matches the CPU exactly (PATTERNS.md sec.3, the determinism rule).
//   p   : predicted probability in [0,1]
//   thr : flagging threshold (e.g. 0.5)  -> returns 1 if p >= thr else 0.
// ---------------------------------------------------------------------------
HD inline int admet_flagged(double p, double thr) {
    return (p >= thr) ? 1 : 0;
}
