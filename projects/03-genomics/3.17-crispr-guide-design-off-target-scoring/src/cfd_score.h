// ===========================================================================
// src/cfd_score.h  --  The ONE TRUE per-window CRISPR off-target scorer
// ---------------------------------------------------------------------------
// Project 3.17 : CRISPR Guide Design & Off-Target Scoring
//
// WHY THIS HEADER EXISTS (the most important idea in the project)
//   The CPU reference (reference_cpu.cpp, host compiler) and the GPU kernel
//   (kernels.cu, nvcc) must produce BYTE-FOR-BYTE identical scores so the demo's
//   verification is exact, not "close enough". The classic way to guarantee that
//   (PATTERNS.md §2, the `__host__ __device__` core idiom) is to put the actual
//   per-window math in ONE place, compiled by BOTH toolchains:
//
//       #ifdef __CUDACC__  -> nvcc defines CFD_HD = __host__ __device__
//       #else              -> host compiler sees CFD_HD = (nothing)
//
//   reference_cpu.cpp loops these functions over every genome window; the GPU
//   kernel calls the SAME functions from one thread per window. Same inputs,
//   same operations, same IEEE-754 rounding -> identical outputs.
//
//   KEEP THIS HEADER CUDA-FREE except for the CFD_HD decorator: no __global__,
//   no <cuda_runtime.h>, no kernel launches. Only then can the host compiler
//   include it. (kernels.cu adds the device-only machinery around these calls.)
//
// THE SCIENCE IN ONE PARAGRAPH (full derivation in ../THEORY.md)
//   SpCas9 is targeted to DNA by a 20-nucleotide guide RNA "spacer". It cuts
//   wherever a genomic 20-mer "protospacer" matches the spacer AND is followed
//   by an "NGG" PAM (the two G's are the recognition signal). Crucially, Cas9
//   tolerates some mismatches -> OFF-TARGET cuts elsewhere in the genome. How
//   much a given mismatch reduces cutting depends on BOTH which position it is
//   at (mismatches near the PAM, the "seed", hurt cutting far more than distal
//   ones) AND the identity of the rNA:dNA pair. The Cutting Frequency
//   Determination (CFD) score models this as a PRODUCT of per-position penalty
//   weights in [0,1]: a perfect match scores 1.0; each mismatch multiplies in a
//   weight < 1. A near-PAM mismatch might cut the score in half; a distal one
//   barely dents it. The genome-wide off-target burden of a guide is then the
//   sum of CFD scores over all near-matching sites.
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cu. The data model
// (encoding, GuideJob struct) lives in reference_cpu.h.
// ===========================================================================
#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// CFD_HD: the host/device decorator. Under nvcc (__CUDACC__ is defined) we mark
// every inline below as callable from BOTH host and device. Under the plain C++
// compiler the decorator expands to nothing, so the same source is ordinary C++.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define CFD_HD __host__ __device__
#else
#define CFD_HD
#endif

// ---------------------------------------------------------------------------
// Problem constants. The spacer length (20) and PAM length (3) are biology, not
// tunables: SpCas9's guide is a 20-mer and its PAM is "NGG" (N = any base, then
// two guanines). A genome window we score therefore spans GUIDE_LEN + PAM_LEN
// = 23 bases: 20 protospacer bases then the 3-base PAM.
// ---------------------------------------------------------------------------
constexpr int GUIDE_LEN = 20;                   // SpCas9 spacer length (nt)
constexpr int PAM_LEN   = 3;                     // "NGG"
constexpr int WINDOW_LEN = GUIDE_LEN + PAM_LEN;  // 23-base genome window we test

// Bases are encoded as 2-bit codes so a position fits in a byte and comparisons
// are trivial integer equality. This mapping MUST match encode_base() in
// reference_cpu.cpp (the genome and guide are encoded with it before scoring).
//   A=0  C=1  G=2  T=3      (any other char -> 255 "invalid", never matches)
enum : uint8_t { BASE_A = 0, BASE_C = 1, BASE_G = 2, BASE_T = 3, BASE_INVALID = 255 };

// ---------------------------------------------------------------------------
// cfd_position_weight: the penalty multiplier for a mismatch at spacer position
// `pos` (0 = PAM-distal 5' end ... GUIDE_LEN-1 = PAM-proximal 3' end, i.e. the
// base nearest the PAM). Returns 1.0 for no penalty, down toward ~0 for a
// cutting-abolishing mismatch.
//
//   *** TEACHING MODEL -- NOT the published Doench-2016 table. ***
//   The real CFD score uses an experimentally-measured 20x(mismatch-type) weight
//   matrix (Doench et al., Nat. Biotechnol. 2016) which we do NOT redistribute
//   here. Instead we use a SYNTHETIC, monotone position model that captures the
//   one biologically essential fact -- the SEED EFFECT: mismatches close to the
//   PAM (high `pos`) are far more disruptive than distal ones (low `pos`). The
//   weight rises smoothly from a small seed value to nearly 1.0 at the 5' end:
//
//       w(pos) = W_DISTAL - (W_DISTAL - W_SEED) * ((GUIDE_LEN-1-pos)/(GUIDE_LEN-1))^2
//
//   so pos = GUIDE_LEN-1 (right at the PAM) -> W_SEED, and pos = 0 -> W_DISTAL.
//   THEORY.md §"real world" explains how to swap in the true table; the code is
//   structured so only this one function changes.
//
//   We compute in DOUBLE and use only multiply/subtract/divide in a FIXED order,
//   so the host and device evaluate bit-identical IEEE-754 arithmetic.
// ---------------------------------------------------------------------------
CFD_HD inline double cfd_position_weight(int pos) {
    // Seed (PAM-proximal) mismatches retain only ~5% of cutting; the most distal
    // mismatch retains ~95%. These two anchors define the synthetic curve.
    const double W_SEED   = 0.05;   // weight for a mismatch AT the PAM-proximal end
    const double W_DISTAL = 0.95;   // weight for a mismatch at the far 5' end
    // Distance from the PAM, normalized to [0,1]: 0 at the seed, 1 at the 5' end.
    const double d = static_cast<double>(GUIDE_LEN - 1 - pos)
                   / static_cast<double>(GUIDE_LEN - 1);
    // Square it so the penalty stays severe across the whole seed region and
    // only relaxes near the distal end (a crude stand-in for the real curve).
    return W_DISTAL - (W_DISTAL - W_SEED) * (1.0 - d * d) ;
}

// ---------------------------------------------------------------------------
// WindowScore: everything we learn about one candidate genome window. Returned
// by value from score_window() so the CPU loop and the GPU thread fill the same
// struct. (POD: trivially copyable to/from the device.)
// ---------------------------------------------------------------------------
struct WindowScore {
    int    mismatches;  // # of guide/protospacer mismatches (0..GUIDE_LEN); -1 = no PAM
    double cfd;         // CFD off-target score in [0,1]; 0.0 if no valid PAM here
};

// ---------------------------------------------------------------------------
// has_ngg_pam: does the 3-base window [w0,w1,w2] match the SpCas9 "NGG" PAM?
//   w0 is "N" (any base is fine); w1 and w2 must both be G. The genome is the
//   forward strand here (teaching simplification; THEORY notes the reverse
//   strand). Invalid bases (255) never equal BASE_G, so an N-run fails cleanly.
// ---------------------------------------------------------------------------
CFD_HD inline bool has_ngg_pam(uint8_t p0, uint8_t p1, uint8_t p2) {
    (void)p0;                                   // "N": unconstrained by design
    return (p1 == BASE_G) && (p2 == BASE_G);    // the two recognition guanines
}

// ---------------------------------------------------------------------------
// score_window: THE core computation, shared verbatim by CPU and GPU.
//   Inputs (all 2-bit base codes):
//     guide  : pointer to GUIDE_LEN spacer bases (5'->3')
//     proto  : pointer to GUIDE_LEN genomic protospacer bases (same orientation)
//     pam    : pointer to PAM_LEN bases immediately 3' of the protospacer
//   Returns a WindowScore:
//     * If the PAM is not NGG -> {mismatches = -1, cfd = 0.0} (not a target site).
//     * Else -> count mismatches and fold the per-position weights into a
//       product. A position that MATCHES contributes a factor of exactly 1.0
//       (so it leaves `cfd` unchanged); a MISMATCH multiplies in its weight.
//
//   Determinism: we iterate pos = 0..GUIDE_LEN-1 in a FIXED order and multiply
//   in that order, so CPU and GPU build the identical floating-point product.
//   There is no add next to these multiplies, so no FMA contraction can differ
//   between host and device (THEORY §"Numerical considerations").
// ---------------------------------------------------------------------------
CFD_HD inline WindowScore score_window(const uint8_t* guide,
                                       const uint8_t* proto,
                                       const uint8_t* pam) {
    WindowScore r;
    r.mismatches = -1;
    r.cfd = 0.0;

    // Gate on the PAM first: no NGG, no Cas9 cut, nothing to score.
    if (!has_ngg_pam(pam[0], pam[1], pam[2])) return r;

    int mm = 0;            // running mismatch count (exact integer)
    double cfd = 1.0;      // running product of per-position weights
    for (int pos = 0; pos < GUIDE_LEN; ++pos) {
        if (guide[pos] != proto[pos]) {        // a mismatch at this position
            ++mm;
            cfd *= cfd_position_weight(pos);   // fold in its penalty (<1.0)
        }
        // else: a match contributes factor 1.0 -> intentionally do nothing.
    }
    r.mismatches = mm;
    r.cfd = cfd;          // == 1.0 for a perfect (on-target) match
    return r;
}
