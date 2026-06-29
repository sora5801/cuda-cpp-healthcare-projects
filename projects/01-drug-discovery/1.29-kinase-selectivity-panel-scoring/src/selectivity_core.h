// ===========================================================================
// src/selectivity_core.h  --  The ONE TRUE per-kinase scoring physics
// ---------------------------------------------------------------------------
// Project 1.29 : Kinase Selectivity Panel Scoring
//
// WHY THIS HEADER EXISTS  (the HD-macro idiom, PATTERNS.md sec 2)
//   The single most useful pattern in this repo: put the *per-element math* in
//   ONE header, marked `__host__ __device__`, so that:
//       * reference_cpu.cpp (compiled by the host C++ compiler) and
//       * kernels.cu        (compiled by nvcc for the GPU)
//   run BYTE-FOR-BYTE identical arithmetic. That turns "GPU == CPU" verification
//   into an EXACT integer comparison instead of a fuzzy float tolerance.
//
//   The trick: when nvcc compiles a .cu it defines __CUDACC__ and understands the
//   `__host__ __device__` decorators (meaning "emit this function for BOTH the CPU
//   and the GPU"). The plain host compiler does NOT know those keywords, so we
//   #define them away to nothing. Same source text, two back-ends, one result.
//
//   KEEP CUDA-ONLY THINGS OUT OF THIS FILE: no `__global__`, no <<<>>>, no
//   cudaXxx(). Only POD structs and inline scalar functions, so the host compiler
//   is happy including it.
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cu (both call score_kinase()).
// The science/math/derivation is in ../THEORY.md.
// ===========================================================================
#pragma once

#include <cstdint>   // fixed-width integers for deterministic fixed-point math

// ---------------------------------------------------------------------------
// HD : the host/device portability macro.
//   * Under nvcc (__CUDACC__ defined) it expands to `__host__ __device__`, so the
//     compiler emits each function for the CPU AND the GPU.
//   * Under the plain host compiler it expands to nothing, so the same text is a
//     perfectly ordinary inline C++ function.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ---------------------------------------------------------------------------
// THE INTERACTION-FINGERPRINT MODEL  (a teaching simplification of KLIFS/IFP)
// ---------------------------------------------------------------------------
// A real kinase-ligand Interaction FingerPrint (IFP) records, for each of the
// ~85 KLIFS binding-site residues, which interaction TYPES are present (H-bond
// donor/acceptor, hydrophobic, aromatic, ionic, ...). KLIFS encodes this as a
// fixed-length bit/feature string per (kinase, ligand) pose.
//
// We model the same SHAPE with a fixed-length feature vector of NFEAT pharmacophore
// channels. There are two players:
//
//   * The COMPOUND (the query) carries, for each feature channel f, a small
//     non-negative integer `lig[f]` = "how strongly this ligand can satisfy a
//     pocket requirement of type f" (count of donors, acceptors, rings, ...).
//
//   * Each KINASE pocket carries, for each channel f, a "requirement" `req[f]`
//     = "how much of feature f this pocket wants to see to bind well" (from its
//     KLIFS pocket residues). It also carries an integer baseline `bias` that
//     captures pocket-independent affinity (size/electrostatics offset).
//
// The predicted binding strength of the compound for a kinase is the per-channel
// MATCH between what the ligand offers and what the pocket needs, summed over
// channels, plus the bias:
//
//        raw_score = bias + sum_f  min(lig[f], req[f]) * weight[f]
//
// `min(offer, need)` is the classic "you only score the overlap you can actually
// form" rule (you cannot make more H-bonds than either partner allows). Using
// INTEGER offers/needs/weights makes the whole sum an integer -> deterministic
// and identical on CPU and GPU (no float reordering, PATTERNS.md sec 3).
//
// We then map raw_score -> a predicted pIC50-like affinity in fixed-point
// milli-units (see predicted_pK_milli below) so the reported numbers look like
// the pK values a kinase panel reports, while staying exact integers.
// ---------------------------------------------------------------------------

// Number of pharmacophore feature channels in our toy IFP. 8 keeps the sample
// human-readable while still exercising the per-channel `min` overlap logic.
// (Real KLIFS IFPs are hundreds of bits; the GPU mapping is identical -- one
// thread still owns one kinase, see THEORY "GPU mapping".)
constexpr int NFEAT = 8;

// Per-channel weights (importance of each interaction type to affinity). These
// are fixed model constants shared by CPU and GPU. Chosen as small integers so
// products stay exact. Index meaning (toy pharmacophore channels):
//   0 = H-bond donor      1 = H-bond acceptor   2 = hydrophobic   3 = aromatic
//   4 = ionic +           5 = ionic -           6 = halogen bond  7 = hinge motif
// The hinge motif (channel 7) is weighted highest: nearly all ATP-competitive
// kinase inhibitors hydrogen-bond to the hinge, so it dominates affinity.
HD inline int feature_weight(int f) {
    // A tiny lookup table. `constexpr`-style data kept inline so it compiles for
    // both the host and the device without a separate translation unit.
    const int W[NFEAT] = { 3, 3, 2, 2, 4, 4, 2, 6 };
    return W[f];
}

// A kinase pocket: its per-channel requirement vector plus a scalar bias.
//   req  : [NFEAT] integer "how much of each feature this pocket wants"  (0..15)
//   bias : pocket-independent affinity offset                            (0..~30)
//   id   : stable index used only for reporting/ranking (NOT in the math)
// POD struct (trivially copyable) so we can memcpy an array of these to the GPU.
struct KinasePocket {
    int32_t req[NFEAT];
    int32_t bias;
    int32_t id;
};

// ---------------------------------------------------------------------------
// score_kinase : the per-kinase scoring physics. THE function that must be
// identical on CPU and GPU. Returns the raw integer match score.
//   lig    : [NFEAT] the compound's per-channel feature offers (read-only)
//   pocket : one kinase pocket (its req[] + bias)
//   returns: bias + sum_f min(lig[f], req[f]) * weight[f]   (an exact integer)
//
// Complexity: O(NFEAT) per kinase, fully unrollable. Called once per kinase by
// BOTH the serial CPU loop and the one-thread-per-kinase GPU kernel.
// ---------------------------------------------------------------------------
HD inline int32_t score_kinase(const int32_t* lig, const KinasePocket& pocket) {
    int32_t acc = pocket.bias;            // start from the pocket's baseline affinity
    for (int f = 0; f < NFEAT; ++f) {
        // Overlap you can actually form on channel f = min(offer, need).
        const int32_t offer = lig[f];
        const int32_t need  = pocket.req[f];
        const int32_t overlap = (offer < need) ? offer : need;   // integer min
        acc += overlap * feature_weight(f);   // weighted contribution (exact int)
    }
    return acc;   // raw match score, monotonically related to predicted affinity
}

// ---------------------------------------------------------------------------
// predicted_pK_milli : map the raw integer score to a predicted affinity that
// reads like a pIC50/pKd, expressed in fixed-point MILLI-units (pK * 1000) so it
// stays an exact integer (deterministic; no floats in the reduction).
//
//   We use a simple affine map  pK = PK_BASE + raw * PK_PER_POINT,  evaluated in
//   milli-units. With the constants below a raw score of 0 -> pK 4.000 (a weak/
//   non-binder floor) and each raw point adds 0.050 pK. This is a TEACHING map,
//   not a fitted QSAR model -- THEORY "real world" explains how production tools
//   (KinoML, machine-learned scorers) replace it.
// ---------------------------------------------------------------------------
constexpr int32_t PK_BASE_MILLI     = 4000;   // pK floor = 4.000 for a zero score
constexpr int32_t PK_PER_POINT_MILLI = 50;    // each raw point adds 0.050 pK

HD inline int32_t predicted_pK_milli(int32_t raw_score) {
    return PK_BASE_MILLI + raw_score * PK_PER_POINT_MILLI;   // exact integer pK*1000
}

// ---------------------------------------------------------------------------
// SELECTIVITY THRESHOLD for the S-score (Karaman et al., Nat. Biotechnol. 2008).
//   The kinome "S-score" S(x) = (# kinases bound with pK >= x) / (# kinases
//   tested). A SMALL S-score means a SELECTIVE compound (it hits few kinases).
//   We pick pK >= 6.000 ("<= 1 microMolar") as the "bound" threshold, the common
//   cutoff for a meaningful kinase hit. Stored in milli-units to compare against
//   predicted_pK_milli() with pure integer arithmetic.
// ---------------------------------------------------------------------------
constexpr int32_t SELECTIVITY_THRESHOLD_MILLI = 6000;   // pK >= 6.000  (<= 1 uM)

// is_hit : does this predicted affinity count as "bound" for the S-score?
//   Pure integer comparison -> the hit COUNT is exact and order-independent, so
//   the S-score is deterministic whether summed on the CPU or via the GPU's
//   per-thread integer flags (PATTERNS.md sec 3: integer reductions commute).
HD inline bool is_hit(int32_t pK_milli) {
    return pK_milli >= SELECTIVITY_THRESHOLD_MILLI;
}
