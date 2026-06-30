// ===========================================================================
// src/antibody.h  --  Shared (host + device) antibody-screening core
// ---------------------------------------------------------------------------
// Project 2.15 : Antibody Structure Prediction  (REDUCED-SCOPE teaching version)
//
// WHAT THE FULL PROBLEM IS (and why we reduce it -- CLAUDE.md §13)
//   "Antibody structure prediction" in the wild (IgFold, ABodyBuilder3) is an
//   attention-based DEEP-LEARNING pipeline: an ESM-2 / IgLM language model embeds
//   the sequence, then an Evoformer/structure module folds the Fv region, with
//   special care for the hypervariable CDR-H3 loop. That is hundreds of MB of
//   trained weights and a multi-kernel transformer -- not a single didactic
//   CUDA kernel. So this project teaches the *load-bearing biology and the GPU
//   pattern* that sits UNDER library-scale antibody work, honestly labelled:
//
//     ANTIBODY LIBRARY SCREENING BY CDR SIMILARITY.
//
//   Given ONE query antibody (its six CDR loops) and a LIBRARY of N antibodies,
//   score how similar each library antibody's CDRs are to the query's, weighting
//   CDR-H3 most (it dominates antigen specificity). This is exactly the
//   high-throughput screening step ABodyBuilder3 accelerates ("thousands of
//   sequences per GPU-hour") -- we do the *similarity scoring*, not the folding.
//
// THE BIOLOGY (see ../THEORY.md "The science")
//   An antibody Fv has two chains (heavy, light). Most of each chain is a
//   conserved beta-sandwich FRAMEWORK; grafted onto it are three hypervariable
//   COMPLEMENTARITY-DETERMINING REGIONS (CDR1/2/3) per chain -- six loops total.
//   The CDRs form the paratope that grips the antigen. CDR-H3 (heavy chain, loop
//   3) is the most variable in length and sequence and contributes most binding
//   energy, so we weight it highest. The IMGT numbering scheme assigns each
//   residue a canonical position so CDRs are delimited consistently across
//   antibodies -- here we pre-delimit the CDRs in the dataset (data/README.md).
//
// THE MATH (see ../THEORY.md "The math")
//   We score two aligned CDR strings a,b of equal length L with a SUBSTITUTION
//   MATRIX S (BLOSUM-like): score = sum_i S[a_i][b_i]. Identical, length-matched
//   CDRs score high; divergent ones score low/negative. The antibody-level score
//   is a CDR-weighted sum (CDR-H3 weight 3, others weight 1). All entries are
//   small integers, so the whole computation is EXACT INTEGER ARITHMETIC -> the
//   CPU reference and the GPU kernel agree BIT-FOR-BIT (tolerance 0).
//
// THE HD-CORE IDIOM (PATTERNS.md §2)
//   The per-pair scoring lives in ONE __host__ __device__ function, ab_cdr_score,
//   included by BOTH reference_cpu.cpp (host compiler) AND kernels.cu (nvcc), so
//   there is exactly one definition of "the score" and the two paths cannot drift.
//
// READ THIS BEFORE: reference_cpu.h, kernels.cuh. Then kernels.cu / reference_cpu.cpp.
// ===========================================================================
#pragma once

#include <cstdint>

// AB_HD expands to __host__ __device__ under nvcc, and to nothing under the host
// C++ compiler (which has never heard of those decorators). This is the standard
// "one formula, two compilers" trick (PATTERNS.md §2).
#ifdef __CUDACC__
#define AB_HD __host__ __device__
#else
#define AB_HD
#endif

// --- Antibody / CDR geometry constants -------------------------------------
// We model the SIX CDR loops of an Fv: heavy H1,H2,H3 then light L1,L2,L3.
// Indexing this way keeps H3 (the specificity-driving loop) at a known slot.
static const int AB_NUM_CDRS = 6;          // H1,H2,H3,L1,L2,L3

// CDR-H3 lives at index 2 (0=H1,1=H2,2=H3,3=L1,4=L2,5=L3). We weight it 3x to
// reflect its outsized role in antigen recognition; the other five weigh 1x.
// (These are TEACHING weights, not a calibrated model -- see THEORY "real world".)
AB_HD inline int ab_cdr_weight(int cdr_index) {
    return (cdr_index == 2) ? 3 : 1;       // index 2 == CDR-H3
}

// Each CDR is stored as a fixed-width, right-padded field so every antibody has
// the SAME memory footprint -> simple row-major device layout, coalesced reads,
// and no per-antibody pointer chasing. 24 covers essentially all natural CDR
// lengths (CDR-H3 is the longest, ~3-25 residues). Padding uses the gap symbol.
static const int AB_CDR_LEN = 24;          // padded residues per CDR field
// Total residues per antibody record = 6 CDRs * 24 = 144 chars (row stride).
static const int AB_RECORD_LEN = AB_NUM_CDRS * AB_CDR_LEN;   // = 144

// --- The amino-acid alphabet ------------------------------------------------
// We encode residues as small integers 0..20: the 20 standard amino acids in the
// canonical BLOSUM order, plus index 20 = gap/pad ('-'). Encoding to ints up
// front means the hot scoring loop indexes a flat matrix with no char branching.
static const int AB_ALPHABET = 21;         // 20 amino acids + gap
static const int AB_GAP = 20;              // index of the gap/pad symbol

// ab_encode_residue: map an ASCII residue letter to its 0..20 index.
//   Returns AB_GAP (20) for '-' OR any unrecognized character, so malformed
//   input degrades to "gap" rather than reading out of bounds. The 26-entry
//   lookup table is laid out A..Z; -1 marks letters that are not amino acids
//   (B,J,O,U,X,Z) which we also fold to gap. We keep this __host__ __device__ so
//   the dataset loader (host) and any device-side parsing share one mapping.
AB_HD inline int ab_encode_residue(char c) {
    // BLOSUM62 row order: A R N D C Q E G H I L K M F P S T W Y V.
    // This table maps ASCII 'A'..'Z' to that 0..19 index, or -1 if not an AA.
    // (constexpr-style flat array; index by (c - 'A').)
    const int LUT[26] = {
        /*A*/  0, /*B*/ -1, /*C*/  4, /*D*/  3, /*E*/  6, /*F*/ 13,
        /*G*/  7, /*H*/  8, /*I*/  9, /*J*/ -1, /*K*/ 11, /*L*/ 10,
        /*M*/ 12, /*N*/  2, /*O*/ -1, /*P*/ 14, /*Q*/  5, /*R*/  1,
        /*S*/ 15, /*T*/ 16, /*U*/ -1, /*V*/ 19, /*W*/ 17, /*X*/ -1,
        /*Y*/ 18, /*Z*/ -1
    };
    if (c >= 'A' && c <= 'Z') {
        const int v = LUT[c - 'A'];
        return (v >= 0) ? v : AB_GAP;      // unknown letter -> gap
    }
    return AB_GAP;                          // '-', lowercase, anything else -> gap
}

// --- The substitution matrix (BLOSUM62, integer) ---------------------------
// A 21x21 matrix S where S[i][j] is the log-odds score of aligning residue i
// with residue j. We use the classic BLOSUM62 integer values for the 20x20 amino
// acid block; the gap row/column (index 20) scores AB_GAP_SCORE so that padding
// vs padding is neutral-ish and a residue vs a gap is penalized. We return it via
// a function (not a global array) so the SAME table is visible to host and device
// without needing __constant__ plumbing for the matrix itself (it is tiny and the
// compiler keeps it in fast memory). See THEORY "The math".
//
// ab_blosum62: score for aligning encoded residues i,j (each 0..20).
//   Symmetric: S[i][j] == S[j][i]. Diagonal (identity) is positive; chemically
//   dissimilar pairs are negative. Integer-valued -> exact, deterministic sums.
AB_HD inline int ab_blosum62(int i, int j) {
    // Row-major BLOSUM62 for the 20 AAs in the order encoded above, then a final
    // gap row/col. Values are the standard NCBI BLOSUM62 integers. Laid out as a
    // flat 21*21 array so both compilers see one identical copy.
    static const signed char S[AB_ALPHABET * AB_ALPHABET] = {
        // A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  -
         4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0,-4, // A
        -1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3,-4, // R
        -2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3,-4, // N
        -2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3,-4, // D
         0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,-4, // C
        -1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2,-4, // Q
        -1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2,-4, // E
         0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3,-4, // G
        -2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3,-4, // H
        -1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3,-4, // I
        -1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1,-4, // L
        -1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2,-4, // K
        -1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1,-4, // M
        -2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1,-4, // F
        -1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2,-4, // P
         1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2,-4, // S
         0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0,-4, // T
        -3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3,-4, // W
        -2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1,-4, // Y
         0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4,-4, // V
        -4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4, 1  // - (gap)
    };
    return S[i * AB_ALPHABET + j];
}

// --- The per-CDR score ------------------------------------------------------
// ab_score_one_cdr: substitution-matrix score of one query CDR field against one
// library CDR field. Both are AB_CDR_LEN encoded residues (0..20). We sum the
// matrix score column by column. This is *ungapped* scoring of length-matched,
// pre-padded fields -- the simplest correct teaching version; THEORY explains
// the gapped (Needleman-Wunsch) version production tools would use.
//   q   : pointer to AB_CDR_LEN encoded residues (the query CDR)
//   l   : pointer to AB_CDR_LEN encoded residues (a library CDR)
//   returns: integer column-sum score (can be negative)
AB_HD inline int ab_score_one_cdr(const uint8_t* q, const uint8_t* l) {
    int s = 0;
    for (int p = 0; p < AB_CDR_LEN; ++p) {
        // Both sides being gap (padding aligned to padding) is the common case
        // for short CDRs; ab_blosum62(GAP,GAP) is +1 so equal-length pads add a
        // small constant to every comparison -- harmless and uniform.
        s += ab_blosum62(q[p], l[p]);
    }
    return s;
}

// --- The antibody-level score ----------------------------------------------
// ab_cdr_score: the full query-vs-one-library-antibody score = CDR-weighted sum
// of the six per-CDR scores, with CDR-H3 (index 2) weighted 3x. This is the ONE
// function the CPU reference loops over all N library antibodies and the GPU
// kernel calls once per thread -- guaranteeing identical results (PATTERNS.md §2).
//   query : AB_RECORD_LEN encoded residues (6 CDR fields of AB_CDR_LEN)
//   lib   : AB_RECORD_LEN encoded residues for one library antibody
//   returns: total integer score; higher = more similar to the query's CDRs
AB_HD inline int ab_cdr_score(const uint8_t* query, const uint8_t* lib) {
    int total = 0;
    for (int c = 0; c < AB_NUM_CDRS; ++c) {
        const int off = c * AB_CDR_LEN;                 // start of CDR c in a record
        const int cdr = ab_score_one_cdr(query + off, lib + off);
        total += ab_cdr_weight(c) * cdr;                // CDR-H3 counts triple
    }
    return total;
}
