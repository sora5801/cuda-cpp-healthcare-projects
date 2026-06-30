// ===========================================================================
// src/reference_cpu.cpp  --  The plain-C++ baseline we trust
// ---------------------------------------------------------------------------
// Project 3.27 : Suffix Array / BWT / FM-Index Construction
//
// ROLE IN THE PROJECT
//   The "ground truth". It builds the suffix array by PREFIX DOUBLING in a way
//   that is obviously correct -- a readable serial loop, std::stable_sort, and a
//   straightforward renumbering -- so that when the GPU agrees with it, we
//   believe the GPU. It uses the SAME packed-key math (sa_core.h) the GPU uses,
//   which is precisely what makes the two suffix arrays bit-for-bit identical.
//
//   Compiled by the host C++ compiler only (no CUDA here).
//
// THE ALGORITHM (prefix doubling -- see ../THEORY.md section "The algorithm")
//   rank[i] starts as the code of T[i]. In round k = 1,2,4,..., we sort all n
//   suffixes by the pair (rank[i], rank[i+k]) -- equivalently by pack_key() --
//   then RENUMBER: suffixes with equal keys get equal new ranks, others get a
//   fresh increasing rank. After O(log n) rounds every rank is unique and the
//   sorted order IS the suffix array. Each round doubles the prefix length whose
//   order we know (1, 2, 4, 8, ... characters), hence "prefix doubling".
//
// READ THIS AFTER: reference_cpu.h, sa_core.h. Compare with kernels.cu (GPU twin).
// ===========================================================================
#include "reference_cpu.h"
#include "sa_core.h"          // pack_key, char_to_code (shared with the GPU)

#include <algorithm>          // std::stable_sort, std::min
#include <cctype>             // std::toupper
#include <numeric>            // std::iota
#include <stdexcept>          // std::runtime_error
#include "util/io.hpp"        // util::read_floats is NOT used here; text loader below

// ---------------------------------------------------------------------------
// load_text: read one line of DNA, uppercase it, validate, append the sentinel.
//   We read the whole file and keep only A/C/G/T characters' positions by
//   validating each one; any other non-whitespace symbol is an error. The '$'
//   sentinel (strictly smallest, see sa_core.h) is appended so the suffix array
//   is unique and well defined.
// ---------------------------------------------------------------------------
std::string load_text(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open input file: " + path);
    std::string text;
    char c;
    // Read character by character so we can skip line breaks/spaces and validate.
    while (in.get(c)) {
        if (c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;  // skip layout
        char up = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        if (up != 'A' && up != 'C' && up != 'G' && up != 'T')
            throw std::runtime_error(std::string("non-ACGT character in input: '") + c + "'");
        text.push_back(up);
    }
    if (text.empty()) throw std::runtime_error("input text is empty: " + path);
    text.push_back('$');   // the sentinel: smallest symbol, makes suffixes distinct
    return text;
}

// ---------------------------------------------------------------------------
// renumber: given suffixes already sorted into `order` by their packed keys,
//   assign each suffix a new rank. Adjacent suffixes with EQUAL keys share a
//   rank; a strictly larger key starts the next rank value. Returns the number
//   of DISTINCT ranks produced (when this equals n, every suffix is unique and
//   the doubling is finished).
//
//   key[order[0]] <= key[order[1]] <= ... is assumed (the caller sorted by key).
//   We walk the sorted order and bump the rank counter only on a key change.
// ---------------------------------------------------------------------------
static int renumber(const std::vector<int>& order, const std::vector<std::uint64_t>& key,
                    std::vector<int>& rank) {
    const int n = static_cast<int>(order.size());
    int r = 0;                       // rank value handed to the first (smallest) suffix
    rank[order[0]] = 0;
    for (int p = 1; p < n; ++p) {
        // New rank only if this suffix's key differs from the previous one.
        if (key[order[p]] != key[order[p - 1]]) ++r;
        rank[order[p]] = r;
    }
    return r + 1;                    // count of distinct ranks = max rank + 1
}

// ---------------------------------------------------------------------------
// suffix_array_cpu: the trusted serial suffix-array builder (prefix doubling).
// ---------------------------------------------------------------------------
SaResult suffix_array_cpu(const std::string& text, const std::string& pattern) {
    const int n = static_cast<int>(text.size());
    SaResult res;
    res.n = n;

    // order[]  : the running permutation of suffix indices, eventually the SA.
    // rank[]   : rank[i] = lexicographic rank of suffix i among the prefixes
    //            seen so far (length doubles each round).
    // key[]    : the packed sort key for each suffix this round.
    std::vector<int>            order(n);
    std::vector<int>            rank(n);
    std::vector<std::uint64_t>  key(n);

    std::iota(order.begin(), order.end(), 0);   // order = 0,1,2,...,n-1

    // ---- Round k = 0: initial ranks are the single-character codes ----------
    for (int i = 0; i < n; ++i) rank[i] = char_to_code(text[i]);

    int rounds = 0;
    // ---- Doubling rounds: k = 1, 2, 4, ... ----------------------------------
    // We stop as soon as every suffix has a unique rank (distinct == n) because
    // at that point the order is fully determined -- more rounds cannot change it.
    for (int k = 1; k < n; k <<= 1) {
        // Build this round's packed key for every suffix using the SHARED packer
        // (sa_core.h) -- the exact same function the GPU calls per thread.
        for (int i = 0; i < n; ++i) key[i] = pack_key(i, k, n, rank.data());

        // Sort suffix indices by their key. stable_sort keeps equal-key suffixes
        // in their previous relative order; the SA of T$ is unique anyway, but a
        // stable sort makes intermediate states reproducible and easy to reason
        // about. Comparator: smaller packed key sorts first.
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return key[a] < key[b]; });

        // Renumber ranks from the freshly sorted order.
        const int distinct = renumber(order, key, rank);
        ++rounds;

        // All ranks unique -> 'order' is the final suffix array. Done early.
        if (distinct == n) break;
    }

    res.sa = order;             // the suffix array
    res.doubling_rounds = rounds;
    res.bwt = bwt_from_sa(text, res.sa);                 // derive the BWT
    res.pattern_count = fm_count(text, res.sa, pattern); // FM backward-search count
    return res;
}

// ---------------------------------------------------------------------------
// bwt_from_sa: BWT[i] = the character JUST BEFORE suffix SA[i] (cyclically).
//   Intuition: the BWT is the last column of the sorted matrix of all cyclic
//   rotations; equivalently, for each suffix in sorted order, take the character
//   preceding its start position (wrapping past 0 to the '$' at the end).
// ---------------------------------------------------------------------------
std::string bwt_from_sa(const std::string& text, const std::vector<int>& sa) {
    const int n = static_cast<int>(text.size());
    std::string bwt(n, '\0');
    for (int i = 0; i < n; ++i) {
        const int j = sa[i];
        // (j - 1 + n) % n wraps index 0 around to the last character ('$').
        bwt[i] = text[(j - 1 + n) % n];
    }
    return bwt;
}

// ---------------------------------------------------------------------------
// fm_count: count occurrences of `pattern` via FM-index BACKWARD SEARCH.
// ---------------------------------------------------------------------------
//   The FM-index needs two things derived from the BWT L:
//     * C[c] = number of characters in the text STRICTLY SMALLER than c. This is
//       the offset of the first row beginning with c in the sorted matrix.
//     * Occ(c, i) = number of occurrences of c in L[0..i-1] (a rank query).
//   Backward search maintains a half-open range [lo, hi) of SA rows whose
//   suffixes start with the current pattern suffix. Processing the pattern from
//   RIGHT to LEFT, each step is the LF-mapping update:
//       lo = C[c] + Occ(c, lo)
//       hi = C[c] + Occ(c, hi)
//   The final (hi - lo) is the number of occurrences. This is the same backward
//   step the GPU could run per query; here we do it serially for the reference.
//   We compute Occ() by a simple count over the BWT, which is O(n) per step --
//   fine for teaching sizes; production uses a wavelet tree / rank dictionary
//   (see ../THEORY.md "Where this sits in the real world").
int fm_count(const std::string& text, const std::vector<int>& sa, const std::string& pattern) {
    const int n = static_cast<int>(text.size());
    if (pattern.empty()) return 0;

    const std::string L = bwt_from_sa(text, sa);   // the BWT (last column)

    // Build C[] over the 256-char space (only A/C/G/T/'$' are non-zero here).
    // First count each symbol in the text, then prefix-sum into "strictly less".
    int count[256] = {0};
    for (char ch : text) count[static_cast<unsigned char>(ch)]++;
    int C[256] = {0};
    int running = 0;
    for (int ch = 0; ch < 256; ++ch) { C[ch] = running; running += count[ch]; }

    // Occ(c, i): occurrences of c in L[0..i-1]. Small helper (linear scan).
    auto occ = [&](char c, int i) {
        int t = 0;
        for (int p = 0; p < i; ++p) if (L[p] == c) ++t;
        return t;
    };

    // Backward search over the pattern, right to left.
    int lo = 0, hi = n;                  // start with the full SA range [0, n)
    for (int p = static_cast<int>(pattern.size()) - 1; p >= 0; --p) {
        const char c = static_cast<char>(std::toupper(static_cast<unsigned char>(pattern[p])));
        lo = C[static_cast<unsigned char>(c)] + occ(c, lo);
        hi = C[static_cast<unsigned char>(c)] + occ(c, hi);
        if (lo >= hi) return 0;          // range emptied -> pattern absent
    }
    return hi - lo;                      // width of the final range = #occurrences
}
