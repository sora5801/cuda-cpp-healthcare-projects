// ===========================================================================
// src/reference_cpu.h  --  Data model, pairing rules, CPU reference & traceback
// ---------------------------------------------------------------------------
// Project 3.10 : RNA Secondary-Structure Prediction  (Nussinov base-pair DP)
//
// WHAT THIS PROJECT COMPUTES
//   Given a single RNA sequence over the alphabet {A,C,G,U}, we predict its
//   SECONDARY STRUCTURE: which bases pair up (A-U, G-C, and the wobble G-U) to
//   form the stems and hairpin loops that the molecule folds into. We use the
//   NUSSINOV algorithm, the classic teaching model of RNA folding: it finds the
//   structure with the MAXIMUM NUMBER OF (non-crossing) BASE PAIRS.
//
//   Let M[i][j] = the most base pairs achievable in the sub-sequence s[i..j]
//   (inclusive, 0-based, i <= j). The recurrence (this is the whole algorithm):
//
//     M[i][j] = max of:
//        (a) M[i+1][j]                          // i is left UNPAIRED
//        (b) M[i][j-1]                          // j is left UNPAIRED
//        (c) M[i+1][j-1] + pair(i, j)           // i pairs with j (if allowed)
//        (d) max over i<=k<j of                 // BIFURCATION: split into two
//                M[i][k] + M[k+1][j]            //   independent sub-structures
//
//   pair(i,j) is 1 if bases s[i],s[j] can form a pair AND are far enough apart
//   to bend back (a minimum hairpin loop of MIN_LOOP unpaired bases), else 0.
//   The answer is M[0][n-1]: the maximum number of base pairs for the whole
//   sequence. TRACEBACK recovers one optimal structure as a dot-bracket string.
//
// WHY A GPU
//   M[i][j] depends on cells with a SMALLER span L = j - i (its neighbours
//   M[i+1][j], M[i][j-1], M[i+1][j-1] all have span L-1 or L-2, and the
//   bifurcation reads only cells of span < L). So every cell on one "span
//   diagonal" L = j - i is independent of the others on that same diagonal and
//   can be filled in PARALLEL. We sweep spans L = 1, 2, ... , n-1 -- the
//   anti-diagonal WAVEFRONT (kernels.cu). This is the same dependency structure
//   as Smith-Waterman (project 3.01), here in upper-triangular form.
//
//   This pure-C++ header is shared by reference_cpu.cpp, main.cu, and kernels.cu.
//   The pairing rule and the per-cell recurrence live here as HD-decorated
//   inline functions (the "__host__ __device__ core", docs/PATTERNS.md §2) so the
//   CPU reference and the GPU kernel run BIT-IDENTICAL integer math -- making
//   verification exact (max_abs_diff == 0), not approximate. No __global__ here.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// HD: the host/device decorator switch (docs/PATTERNS.md §2).
//   When this header is pulled in by nvcc (kernels.cu, main.cu), __CUDACC__ is
//   defined and we tag the shared functions __host__ __device__ so the SAME
//   source compiles for BOTH the CPU and the GPU. When the plain host compiler
//   builds reference_cpu.cpp, __CUDACC__ is absent and HD expands to nothing,
//   so cl.exe / g++ sees ordinary inline functions. One formula, two backends,
//   identical results.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// Minimum number of UNPAIRED bases enclosed by a base pair (the smallest hairpin
// loop). Sterics forbid a strand bending back on itself too tightly; the Vienna
// model uses 3. So bases i and j may pair only if (j - i) > MIN_LOOP. Integer
// constant => shared exactly by CPU and GPU.
constexpr int MIN_LOOP = 3;

constexpr char ALPHABET[] = "ACGU";   // index 0..3 <-> ribonucleotide base

// ---------------------------------------------------------------------------
// can_pair: the Watson-Crick + wobble pairing rule, as integer base codes 0..3.
//   Codes: A=0, C=1, G=2, U=3 (matches ALPHABET above).
//   Canonical RNA pairs: A-U, G-C, and the weaker "wobble" G-U. We return 1 if
//   the two codes form any of those (in either order), else 0. Pure integer
//   comparisons -> identical on host and device. This is the ONLY chemistry in
//   the model; everything else is counting.
// ---------------------------------------------------------------------------
HD inline int can_pair(uint8_t a, uint8_t b) {
    // A(0)-U(3)
    if ((a == 0 && b == 3) || (a == 3 && b == 0)) return 1;
    // C(1)-G(2)
    if ((a == 1 && b == 2) || (a == 2 && b == 1)) return 1;
    // G(2)-U(3) wobble
    if ((a == 2 && b == 3) || (a == 3 && b == 2)) return 1;
    return 0;
}

// ---------------------------------------------------------------------------
// pair_score: the score contribution of pairing positions i and j.
//   Returns 1 only if the bases can pair AND the loop is long enough
//   (j - i > MIN_LOOP). Centralising this here guarantees the CPU and GPU apply
//   exactly the same constraint. (s is the encoded sequence.)
// ---------------------------------------------------------------------------
HD inline int pair_score(const uint8_t* s, int i, int j) {
    if (j - i <= MIN_LOOP) return 0;     // hairpin too tight to close
    return can_pair(s[i], s[j]);
}

// ---------------------------------------------------------------------------
// nussinov_cell: the per-cell recurrence (a)-(d) above, factored out so the CPU
//   loop and the GPU kernel call the IDENTICAL function for cell (i, j).
//   Inputs:
//     s   : [n] encoded sequence (codes 0..3)
//     M   : the DP matrix so far, row-major with row stride `n` (M[i*n + j]).
//           All cells of SMALLER span (j-i) it reads are already finalised.
//     i,j : the cell to compute (0 <= i < j < n; diagonal i==j is the 0 base case)
//     n   : sequence length (= row stride)
//   Returns M[i][j]. Reads only: M[i+1][j], M[i][j-1], M[i+1][j-1] (span-1/-2),
//   and the bifurcation pairs M[i][k], M[k+1][j] for i<=k<j (all span < j-i).
//   Because it touches no cell of its OWN span, an entire span-diagonal can be
//   evaluated in parallel with no synchronisation (see kernels.cu).
// ---------------------------------------------------------------------------
HD inline int nussinov_cell(const uint8_t* s, const int* M, int i, int j, int n) {
    // (a) leave base i unpaired: inherit the best of the shorter interval [i+1,j].
    int best = M[(i + 1) * n + j];

    // (b) leave base j unpaired: best of [i, j-1].
    int v = M[i * n + (j - 1)];
    if (v > best) best = v;

    // (c) pair i with j: the inside [i+1, j-1] plus this pair's score.
    v = M[(i + 1) * n + (j - 1)] + pair_score(s, i, j);
    if (v > best) best = v;

    // (d) BIFURCATION: try every split point k, joining two independent optimal
    //     sub-structures [i,k] and [k+1,j]. This O(n) inner loop is what makes
    //     the whole algorithm O(n^3); it also handles multi-loops / branches.
    for (int k = i; k < j; ++k) {
        v = M[i * n + k] + M[(k + 1) * n + j];
        if (v > best) best = v;
    }
    return best;
}

// A loaded problem: one RNA sequence, encoded as 0..3 indices into ALPHABET.
struct RnaSeq {
    int n = 0;                  // sequence length
    std::vector<uint8_t> s;     // [n] encoded bases (0..3)
    std::string raw;            // [n] original letters (for pretty-printing)
};

// One recovered structure: the max pair count and a dot-bracket rendering, e.g.
//   GGGAAAUCCC
//   (((...)))   <- '(' i pairs with a later ')' ; '.' is unpaired
struct Structure {
    int pairs = 0;              // maximum number of base pairs (= M[0][n-1])
    std::string dot_bracket;    // [n] '(' / ')' / '.' for the recovered folding
};

// Load one RNA sequence (first non-empty, non-'>' line) from a text file.
// Accepts A/C/G/U (and T, treated as U, since many files store DNA letters).
// Throws std::runtime_error on a bad/empty input so demos fail loudly.
RnaSeq load_rna(const std::string& path);

// CPU reference: fill the Nussinov matrix M (n*n ints, row-major M[i*n+j]).
// Only the upper triangle (i <= j) is meaningful; the diagonal and lower
// triangle stay 0. This is the trusted serial baseline the GPU wavefront is
// checked against -- every cell of the upper triangle must match exactly.
void nussinov_cpu(const RnaSeq& r, std::vector<int>& M);

// Recover one optimal structure from a filled matrix M (host-side; the traceback
// is inherently serial and is NOT the GPU teaching point, so we do it once on
// whichever matrix we display). Deterministic tie-breaking: at each cell we test
// the cases in the fixed order unpaired-i, unpaired-j, pair(i,j), bifurcation,
// and take the FIRST that reproduces M[i][j]. Returns pairs + dot-bracket.
Structure traceback(const RnaSeq& r, const std::vector<int>& M);
