// ===========================================================================
// src/route_score.h  --  The ONE TRUE per-route scoring formula (CPU == GPU)
// ---------------------------------------------------------------------------
// Project 1.20 : Reaction Yield / Retrosynthesis Scoring   (reduced-scope, see
//                README "Limitations & honesty" and THEORY "real world").
//
// WHY THIS HEADER EXISTS  (PATTERNS.md sec.2 -- the HD-macro idiom)
//   The single most useful idiom in this repo: put the PER-ELEMENT physics in
//   ONE header as `__host__ __device__` inline functions, so the CPU reference
//   (reference_cpu.cpp, compiled by cl.exe) and the GPU kernel (kernels.cu,
//   compiled by nvcc) run the SAME source-level math. The two results then agree
//   to a few times 1e-8 -- not bit-identical, because expf() and FMA contraction
//   differ slightly between host and device (THEORY "Numerical considerations"),
//   but far inside our 1e-6 verification tolerance.
//
//   This file therefore contains NO CUDA-only constructs (no __global__, no
//   <<<>>>), only the macro-guarded `RS_HD` decorator, so the host compiler can
//   include it happily. kernels.cu and reference_cpu.cpp both include it.
//
// THE PROBLEM IN ONE PARAGRAPH  (full derivation in ../THEORY.md)
//   Retrosynthesis planning decomposes a target molecule into purchasable
//   building blocks via a SEQUENCE of known reactions (a "route"). A planner
//   (AiZynthFinder, ASKCOS) explores a tree of such routes and must SCORE each
//   candidate route so it can rank them. The score of a route is essentially
//   the probability it would actually work in the lab: the PRODUCT of the
//   per-step success/yield probabilities, boosted when the leaves are cheap,
//   in-stock building blocks. We compute that score for a whole BATCH of
//   candidate routes at once -- and every route is independent, so each route
//   gets its own GPU thread (exactly the 1.12 / 12.01 "independent jobs"
//   pattern).
//
//   The per-STEP yield is predicted from a few features of the reaction
//   (a learned template prior, reagent/condition penalties, ...). In a real
//   system that predictor is a transformer or GNN; here it is a small,
//   transparent LOGISTIC model so nothing is a black box (THEORY sec.real-world
//   explains exactly what we swapped out and why the GPU mapping is identical).
//
// READ THIS BEFORE: reference_cpu.h, kernels.cuh, kernels.cu.
// ===========================================================================
#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// RS_HD: expands to `__host__ __device__` under nvcc (so the function compiles
// for BOTH the CPU and every GPU thread), and to nothing under the plain host
// compiler (which has never heard of those keywords). This is the portability
// trick that lets reference_cpu.cpp include this very header.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define RS_HD __host__ __device__
// RS_UNROLL: nvcc understands `#pragma unroll` to fully unroll a fixed-length
// loop; the plain host compiler does NOT and would warn (C4068 "unknown
// pragma"). We therefore emit the pragma ONLY under nvcc, via the _Pragma
// operator (the token-string form of #pragma usable inside a macro). On the
// host side RS_UNROLL expands to nothing -- the tiny loops there are trivial.
#define RS_UNROLL _Pragma("unroll")
#else
#define RS_HD
#define RS_UNROLL
#endif

// ---------------------------------------------------------------------------
// PROBLEM-SIZE CONSTANTS (compile-time so loops unroll and layouts are fixed).
//
//   NUM_FEATURES : how many numeric features describe ONE reaction step. We use
//                  a deliberately small, INTERPRETABLE set (see THEORY sec.math
//                  for the meaning of each slot):
//                    [0] template_prior   -- historical reliability of the
//                                            reaction template, in [0,1]
//                    [1] precedent_count  -- log10(1+#literature precedents),
//                                            normalized to ~[0,1]
//                    [2] condition_penalty-- harshness of required conditions
//                                            (high temp / pressure / exotic
//                                            catalyst), in [0,1], BAD when high
//                    [3] selectivity      -- expected regio/stereo selectivity,
//                                            in [0,1], GOOD when high
//   MAX_STEPS    : the longest route we score (rows in a route's step matrix).
//                  Shorter routes pad unused steps with the SENTINEL below.
// ---------------------------------------------------------------------------
constexpr int NUM_FEATURES = 4;    // features per reaction step
constexpr int MAX_STEPS    = 6;    // max reaction steps in a route

// A step feature-vector whose first slot is < 0 marks "no step here" (padding
// for routes shorter than MAX_STEPS). We pick a sentinel that can never be a
// real, in-range feature value.
constexpr float STEP_ABSENT = -1.0f;

// ---------------------------------------------------------------------------
// Logistic (sigmoid) link.  sigma(z) = 1 / (1 + e^-z), mapping any real score
// z onto a probability in (0,1). This is the SAME function a logistic-regression
// or a neural net's output head uses; using it here keeps the per-step yield a
// genuine probability. We implement it with expf (single precision); the host
// and device expf agree to within a few ULPs, the dominant source of the tiny
// CPU-vs-GPU difference (THEORY sec.numerics discusses the FP determinism point).
// ---------------------------------------------------------------------------
RS_HD inline float rs_sigmoid(float z) {
    return 1.0f / (1.0f + expf(-z));
}

// ---------------------------------------------------------------------------
// step_yield: predicted SUCCESS PROBABILITY / yield of a SINGLE reaction step.
//
//   This is our stand-in for the transformer/GNN yield predictor. It is a plain
//   logistic model: z = w . x + b, then yield = sigma(z). The weights `w` encode
//   chemical intuition and are shared by every step (a "global" reaction-quality
//   model); they live in constant memory on the GPU (see kernels.cu) and in a
//   plain array on the CPU -- but the ARITHMETIC here is identical either way.
//
//   Parameters:
//     x  : pointer to this step's NUM_FEATURES features (see slot meanings above)
//     w  : pointer to NUM_FEATURES weights (one per feature)
//     b  : scalar bias (intercept)
//   Returns: the step yield in (0,1).
//
//   Note the SIGN handling is baked into the weights, not the formula: features
//   that are bad-when-high (condition_penalty) get a NEGATIVE weight. Keeping the
//   formula a pure dot-product keeps the CPU and GPU math identical at the source
//   level (they then agree to ~1e-8 -- see the file header on why not exactly).
// ---------------------------------------------------------------------------
RS_HD inline float step_yield(const float* x, const float* w, float b) {
    float z = b;
    // NUM_FEATURES is a compile-time constant, so under nvcc this loop fully
    // unrolls into a short straight-line dot product (no loop overhead).
    RS_UNROLL
    for (int f = 0; f < NUM_FEATURES; ++f) {
        z += w[f] * x[f];          // accumulate the weighted feature
    }
    return rs_sigmoid(z);          // squash to a probability in (0,1)
}

// ---------------------------------------------------------------------------
// route_score: the headline number -- the predicted overall success of a whole
// retrosynthetic ROUTE (a sequence of up to MAX_STEPS reaction steps).
//
//   A route succeeds end-to-end only if EVERY step succeeds, so under the
//   (teaching) independence assumption the route's success probability is the
//   PRODUCT of the per-step yields:
//        route_prob = PROD_over_steps  step_yield(step)
//   We then multiply by a BUILDING-BLOCK AVAILABILITY factor in [0,1]: a route
//   that bottoms out in cheap, in-stock starting materials is worth more than an
//   equally-probable route that needs an exotic precursor. (AiZynthFinder applies
//   exactly this kind of stock bonus.)
//
//   We return the product directly (a probability in [0,1]); a higher score is a
//   better, more synthesizable route. THEORY sec.numerics explains why we keep
//   the product in float and why we DO NOT switch to a log-sum here (we want the
//   demo's headline numbers to read as plain probabilities; the log form is an
//   exercise).
//
//   Parameters (all describe ONE route; this is what one GPU thread evaluates):
//     feats        : MAX_STEPS * NUM_FEATURES floats, row-major. Row s is the
//                    feature vector of step s; a row whose [0] == STEP_ABSENT is
//                    padding and is skipped (contributes a yield of 1, i.e. the
//                    multiplicative identity -- a "no-op" step).
//     availability : building-block availability factor for THIS route, in [0,1].
//     w, b         : the shared logistic weights + bias (same for all routes).
//   Returns: the route score in [0,1].
// ---------------------------------------------------------------------------
RS_HD inline float route_score(const float* feats, float availability,
                               const float* w, float b) {
    float prob = 1.0f;                       // multiplicative identity
    RS_UNROLL
    for (int s = 0; s < MAX_STEPS; ++s) {
        const float* x = feats + s * NUM_FEATURES;   // step s feature row
        // Skip padded steps: a missing step neither helps nor hurts (yield 1).
        if (x[0] == STEP_ABSENT) continue;
        prob *= step_yield(x, w, b);          // chain the independent yields
    }
    return prob * availability;               // reward in-stock leaves
}
