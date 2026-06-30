// ===========================================================================
// src/reference_cpu.h  --  CPU reference API + shared data types (pure C++)
// ---------------------------------------------------------------------------
// Project 3.27 : Suffix Array / BWT / FM-Index Construction
//
// WHY A SEPARATE PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the plain host compiler (cl.exe / g++) and
//   must NEVER see CUDA syntax (__global__, etc.). So all the types and function
//   signatures shared between the CPU baseline (reference_cpu.cpp) and the GPU
//   entry point (main.cu) live HERE, in a CUDA-free header. kernels.cuh includes
//   this header too, so the GPU side sees the same SaResult struct.
//
// WHAT THIS PROJECT COMPUTES  (the real algorithm, not a placeholder)
//   Given a DNA text T over {A,C,G,T} we:
//     1. append a unique sentinel '$' (smallest symbol) -> guarantees all
//        suffixes are distinct, so the suffix array is unique;
//     2. build the SUFFIX ARRAY  SA[0..n-1]: the starting positions of all
//        suffixes of T$, sorted lexicographically (via prefix doubling);
//     3. apply the BURROWS-WHEELER transform: BWT[i] = T$[(SA[i]-1+n) mod n];
//     4. build a tiny FM-INDEX (the C[] table + per-symbol occurrence counts)
//        and run BACKWARD SEARCH (LF mapping) to COUNT occurrences of a query.
//   The CPU does this serially and obviously-correctly; the GPU does the same
//   with a parallel prefix-doubling sort. main.cu compares the two SAs exactly.
//
// READ THIS BEFORE: reference_cpu.cpp (implements these) and main.cu (calls them).
// ===========================================================================
#pragma once

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// SaResult: everything one run produces. Returned by both the CPU and the GPU
//   paths so main.cu can compare them field by field.
// ---------------------------------------------------------------------------
struct SaResult {
    int                 n      = 0;   // text length INCLUDING the '$' sentinel
    std::vector<int>    sa;           // [n] suffix array: sorted suffix start positions
    std::string         bwt;          // [n] Burrows-Wheeler transform of T$ (chars)
    int                 pattern_count = 0;  // FM backward-search occurrences of the query
    int                 doubling_rounds = 0;// prefix-doubling iterations actually run
};

// ---------------------------------------------------------------------------
// load_text: read the committed DNA sample (one line of A/C/G/T) and append '$'.
//   path : file containing the raw DNA string (no sentinel, no header).
//   Returns the text WITH the '$' sentinel appended (so length = bases + 1).
//   Throws std::runtime_error if the file is missing or contains a non-ACGT char
//   -> the demo fails loudly rather than silently mis-parsing.
// ---------------------------------------------------------------------------
std::string load_text(const std::string& path);

// ---------------------------------------------------------------------------
// suffix_array_cpu: build SA + BWT + FM backward-search count, all on the CPU.
//   text    : the input text T$ (sentinel already appended), length n.
//   pattern : the query string to count via FM-index backward search.
//   Returns a fully-populated SaResult. This is the trusted baseline.
//   Complexity: O(n log^2 n) (log n doubling rounds, each an O(n log n) sort).
// ---------------------------------------------------------------------------
SaResult suffix_array_cpu(const std::string& text, const std::string& pattern);

// ---------------------------------------------------------------------------
// bwt_from_sa: derive the Burrows-Wheeler transform string from a suffix array.
//   Shared by CPU and GPU paths (pure host code) so both produce the same BWT
//   from their (identical) suffix arrays. BWT[i] = text[(SA[i]-1+n) mod n].
// ---------------------------------------------------------------------------
std::string bwt_from_sa(const std::string& text, const std::vector<int>& sa);

// ---------------------------------------------------------------------------
// fm_count: count occurrences of `pattern` in `text` using an FM-index built
//   from its BWT (the C[] table + occurrence counts) via backward search.
//   This is pure host code reused by both paths to validate the SA downstream:
//   a correct SA -> correct BWT -> correct backward-search count.
//   Returns the number of occurrences (0 if the pattern is absent).
// ---------------------------------------------------------------------------
int fm_count(const std::string& text, const std::vector<int>& sa, const std::string& pattern);
