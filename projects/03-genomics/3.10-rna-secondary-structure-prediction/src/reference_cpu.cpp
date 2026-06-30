// ===========================================================================
// src/reference_cpu.cpp  --  Trusted serial Nussinov DP + loader + traceback
// ---------------------------------------------------------------------------
// Project 3.10 : RNA Secondary-Structure Prediction  (Nussinov base-pair DP)
//
//   (1) load_rna()       : parse the one-line sequence sample (data/README.md).
//   (2) nussinov_cpu()   : the obviously-correct serial DP the GPU is checked on.
//                          It fills the SAME matrix the GPU wavefront fills, but
//                          one cell at a time, in order of increasing span L.
//   (3) traceback()      : recover one optimal dot-bracket structure (host only;
//                          inherently serial, so not the GPU teaching point).
//
//   The actual per-cell math (the recurrence) lives in reference_cpu.h as the
//   shared __host__ __device__ function nussinov_cell(), so this serial loop and
//   the GPU kernel compute byte-identical integers. Compiled by the host C++
//   compiler only -- no CUDA syntax here. See reference_cpu.h for the model.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <stdexcept>

// Map an RNA base character to its 0..3 code (or throw on an invalid letter).
// We accept 'T' as a synonym for 'U' so the loader also reads DNA-style files;
// lowercase is tolerated. Codes match ALPHABET = "ACGU".
static uint8_t encode(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'U': case 'u': return 3;
        case 'T': case 't': return 3;   // DNA 'T' <-> RNA 'U'
        default:
            throw std::runtime_error(std::string("non-ACGU character in sequence: '") + c + "'");
    }
}

// Read the first usable line of a (possibly FASTA-ish) file as the RNA sequence.
// We skip blank lines and any '>' header line, then encode the residues. Stray
// whitespace / carriage returns are tolerated so Windows and Unix files both work.
RnaSeq load_rna(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open sequence file: " + path);

    std::string line;
    while (std::getline(in, line)) {
        // Trim a trailing '\r' (Windows line endings) and skip empties/headers.
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '>') continue;

        RnaSeq r;
        for (char c : line) {
            if (c == ' ' || c == '\t') continue;     // tolerate stray whitespace
            r.s.push_back(encode(c));                // throws on a bad letter
            r.raw.push_back(ALPHABET[r.s.back()]);   // canonical upper-case A/C/G/U
        }
        r.n = static_cast<int>(r.s.size());
        if (r.n == 0) throw std::runtime_error("sequence line was empty in " + path);
        return r;
    }
    throw std::runtime_error("no sequence line found in " + path);
}

// ---------------------------------------------------------------------------
// nussinov_cpu: fill M in order of INCREASING span L = j - i.
//   The base cases (span 0: single bases; the empty interval) score 0 pairs, so
//   we start the matrix all-zero. Then for L = 1, 2, ..., n-1 and each i, the
//   cell (i, j=i+L) reads only cells of smaller span -- already filled -- so a
//   single forward pass is correct. This deliberate span-by-span order MIRRORS
//   the GPU's wavefront (kernels.cu launches one kernel per span L), which is
//   why the two matrices come out identical cell for cell.
//   Complexity: O(n^2) cells * O(n) bifurcation loop = O(n^3) time, O(n^2) space.
// ---------------------------------------------------------------------------
void nussinov_cpu(const RnaSeq& r, std::vector<int>& M) {
    const int n = r.n;
    M.assign(static_cast<std::size_t>(n) * n, 0);   // all spans start at 0 pairs
    const uint8_t* s = r.s.data();

    for (int L = 1; L < n; ++L) {                   // span = j - i, grows outward
        for (int i = 0; i + L < n; ++i) {           // every interval of this span
            const int j = i + L;
            // One call to the SHARED recurrence -> the same integer the GPU
            // thread for (i, j) will compute. See nussinov_cell in the header.
            M[static_cast<std::size_t>(i) * n + j] = nussinov_cell(s, M.data(), i, j, n);
        }
    }
}

// ---------------------------------------------------------------------------
// traceback: reconstruct ONE optimal structure from the filled matrix M.
//   We push the whole interval [0, n-1] on a stack and repeatedly resolve the
//   top interval [i, j] by asking "which case of the recurrence produced
//   M[i][j]?" -- testing them in the SAME fixed order as nussinov_cell so ties
//   break deterministically. When i pairs with j we mark '(' at i and ')' at j
//   and recurse on the inside [i+1, j-1]; a bifurcation pushes the two halves.
//   This is O(n) work along the optimal path -- serial and cheap, so we keep it
//   on the host. Output: a dot-bracket string of length n.
// ---------------------------------------------------------------------------
Structure traceback(const RnaSeq& r, const std::vector<int>& M) {
    const int n = r.n;
    const uint8_t* s = r.s.data();

    Structure out;
    out.pairs = (n > 0) ? M[0 * n + (n - 1)] : 0;   // M[0][n-1] is the answer
    out.dot_bracket.assign(static_cast<std::size_t>(n), '.');   // default: unpaired

    // An explicit work-stack of intervals avoids deep recursion on long RNAs.
    std::vector<std::pair<int, int>> stack;
    if (n > 0) stack.emplace_back(0, n - 1);

    while (!stack.empty()) {
        const auto [i, j] = stack.back();
        stack.pop_back();
        if (i >= j) continue;                       // empty / single-base interval

        const int here = M[static_cast<std::size_t>(i) * n + j];

        // Case (a): base i unpaired -> the rest of the structure is in [i+1, j].
        if (here == M[static_cast<std::size_t>(i + 1) * n + j]) {
            stack.emplace_back(i + 1, j);
            continue;
        }
        // Case (b): base j unpaired -> structure in [i, j-1].
        if (here == M[static_cast<std::size_t>(i) * n + (j - 1)]) {
            stack.emplace_back(i, j - 1);
            continue;
        }
        // Case (c): i pairs with j -> record the bracket, recurse on the inside.
        if (here == M[static_cast<std::size_t>(i + 1) * n + (j - 1)] + pair_score(s, i, j)) {
            out.dot_bracket[static_cast<std::size_t>(i)] = '(';
            out.dot_bracket[static_cast<std::size_t>(j)] = ')';
            stack.emplace_back(i + 1, j - 1);
            continue;
        }
        // Case (d): bifurcation -> find the split k and push both halves.
        for (int k = i; k < j; ++k) {
            if (here == M[static_cast<std::size_t>(i) * n + k]
                      + M[static_cast<std::size_t>(k + 1) * n + j]) {
                stack.emplace_back(i, k);
                stack.emplace_back(k + 1, j);
                break;
            }
        }
    }
    return out;
}
