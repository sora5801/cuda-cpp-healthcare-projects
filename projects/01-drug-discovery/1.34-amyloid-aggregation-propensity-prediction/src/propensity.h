// ===========================================================================
// src/propensity.h  --  The ONE TRUE per-residue aggregation physics
//                       (shared by the CPU reference AND the GPU kernel)
// ---------------------------------------------------------------------------
// Project 1.34 : Amyloid / Aggregation Propensity Prediction
//
// WHY THIS HEADER EXISTS  (PATTERNS.md §2 -- the CPU/GPU parity idiom)
//   This is the single most important file for VERIFICATION. The amino-acid ->
//   propensity lookup and the sliding-window mean are written here ONCE, as
//   `__host__ __device__` inline functions, so:
//      * reference_cpu.cpp (compiled by the host compiler) and
//      * kernels.cu        (compiled by nvcc for the GPU)
//   both call the *identical* arithmetic in the *identical* order. The GPU
//   result then matches the CPU result to ~float epsilon -- not approximately,
//   but because they literally execute the same code. (If the two diverged we
//   would not know whether a mismatch was a GPU bug or a different formula.)
//
//   The HD macro below expands to `__host__ __device__` under nvcc and to
//   nothing under the host compiler (which has never heard of those keywords).
//   We keep CUDA-only constructs (__global__, <<<>>>, shared memory) OUT of this
//   header so the plain C++ compiler can include it without complaint.
//
// READ THIS AFTER: reference_cpu.h (defines Protein/Dataset/AggResult).
// ===========================================================================
#pragma once

#include "reference_cpu.h"   // PAD_CODE (the padding sentinel)

// --- The HD ("host/device") decorator macro --------------------------------
// Under nvcc, __CUDACC__ is defined and we want both a host copy (for the CPU
// reference) and a device copy (for the kernel) of each function. Under the
// host compiler the decorators do not exist, so HD must vanish.
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ---------------------------------------------------------------------------
// AA_ORDER / the propensity scale
// ---------------------------------------------------------------------------
// We index the 20 standard amino acids in this fixed order; any other symbol
// (B, Z, X, '-', ...) maps to index 20, the "other" bucket. `code_of_char`
// (in reference_cpu.cpp) does the char -> index mapping ONCE at load time, so
// the hot loops here only ever see small ints.
//
//   index:   0    1    2    3    4    5    6    7    8    9
//   AA   :   A    R    N    D    C    Q    E    G    H    I
//   index:  10   11   12   13   14   15   16   17   18   19   20
//   AA   :   L    K    M    F    P    S    T    W    Y    V   (other)
//
// AA_PROPENSITY[k] is the INTRINSIC beta-aggregation propensity of amino acid k.
// Higher = more aggregation-prone. The values are a transparent, didactic scale
// in the spirit of published intrinsic-aggregation scales (e.g. Pawar et al.
// 2005, the basis of Zyggregator/TANGO-style predictors): aliphatic/aromatic
// hydrophobics (I, L, V, F, Y, W) and the beta-branched/Cys residues score
// HIGH; charged (D, E, K, R) and the helix/turn breakers (P, G) score LOW. The
// exact numbers are illustrative -- the POINT is the pipeline, not the scale --
// and are documented as such in THEORY.md and data/README.md. The "other"
// bucket (index 20) scores 0 so unknown symbols never create a false hot spot.
//
// Stored in a plain constexpr array so the SAME literal table is compiled into
// the host code and (via cudaMemcpyToSymbol from kernels.cu) onto the device.
static constexpr int   AA_COUNT     = 21;     // 20 standard + 1 "other"
static constexpr float AA_PROPENSITY[AA_COUNT] = {
    //  A      R      N      D      C      Q      E      G      H      I
    0.50f, 0.10f, 0.30f, 0.05f, 0.90f, 0.30f, 0.05f, 0.15f, 0.35f, 1.00f,
    //  L      K      M      F      P      S      T      W      Y      V
    0.95f, 0.10f, 0.70f, 0.95f, 0.00f, 0.35f, 0.45f, 0.90f, 0.85f, 1.00f,
    // other
    0.00f
};

// ---------------------------------------------------------------------------
// propensity_of_code: map an amino-acid index to its intrinsic propensity.
//   code : 0..20  (a valid amino-acid index), or PAD_CODE (-1) for padding.
//   returns the scale value, or 0 for padding/out-of-range (so padded tail
//           residues contribute nothing to a window mean).
//   This is a pure table lookup -- O(1), no branches on the hot path beyond the
//   padding guard -- and is identical on host and device.
// ---------------------------------------------------------------------------
HD inline float propensity_of_code(int code) {
    if (code < 0 || code >= AA_COUNT) return 0.0f;   // PAD_CODE or junk -> 0
    return AA_PROPENSITY[code];
}

// ---------------------------------------------------------------------------
// windowed_mean: the centered sliding-window mean at one residue.
//   This is the heart of every sequence aggregation predictor: a residue is
//   dangerous only if it sits inside a *contiguous* aggregation-prone stretch,
//   so we average the intrinsic propensities over a window centered on it.
//
//   codes  : pointer to THIS protein's residues (length `len`; for the GPU this
//            points into the shared-memory tile, for the CPU into the sequence).
//   len    : number of REAL residues in this protein (padding excluded).
//   i      : the residue whose smoothed score we want (0 <= i < len).
//   half   : half-window; the full window is W = 2*half + 1 residues.
//   returns mean propensity over [i-half, i+half] CLAMPED to [0, len).
//
//   We divide by the number of residues ACTUALLY inside the chain (not by W),
//   so residues near the termini -- and the padded tail -- are handled
//   correctly without biasing the score toward zero. Summation runs left-to-
//   right in a single float accumulator, and BOTH the host and device call this
//   exact function, so the rounding is bit-for-bit the same on both sides.
// ---------------------------------------------------------------------------
HD inline float windowed_mean(const float* propensities, int len, int i, int half) {
    int lo = i - half;  if (lo < 0)      lo = 0;        // clamp left edge
    int hi = i + half;  if (hi > len - 1) hi = len - 1; // clamp right edge
    float sum = 0.0f;                                   // running window sum
    for (int j = lo; j <= hi; ++j) sum += propensities[j];
    int count = hi - lo + 1;                            // real residues averaged
    return sum / static_cast<float>(count);             // count >= 1 always
}

#undef HD   // keep the macro local to this header (PATTERNS.md §2 hygiene)
