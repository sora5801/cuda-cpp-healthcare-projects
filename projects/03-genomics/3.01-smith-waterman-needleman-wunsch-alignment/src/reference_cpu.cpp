// ===========================================================================
// src/reference_cpu.cpp  --  Trusted serial Smith-Waterman + loader + traceback
// ---------------------------------------------------------------------------
// Project 3.01 : Smith-Waterman / Needleman-Wunsch Alignment
//
// (1) load_sequences()  : parse the two-line FASTA-ish sample (data/README.md).
// (2) sw_cpu()          : the obviously-correct serial DP the GPU is checked on.
// (3) traceback()       : recover the alignment from a filled matrix (host only;
//                         not the GPU teaching point, so done once on the host).
//
// Compiled by the host C++ compiler only. See reference_cpu.h.
// ===========================================================================
#include "reference_cpu.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>

// Map a nucleotide character to its 0..3 code (or throw on an invalid letter).
static uint8_t encode(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default:
            throw std::runtime_error(std::string("non-ACGT character in sequence: '") + c + "'");
    }
}

SeqPair load_sequences(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open sequence file: " + path);
    std::string qline, tline;
    if (!std::getline(in, qline) || !std::getline(in, tline))
        throw std::runtime_error("expected two sequence lines (query, target) in " + path);

    auto encode_seq = [](const std::string& s) {
        std::vector<uint8_t> v;
        for (char c : s) {
            if (c == '\r' || c == ' ' || c == '\t') continue;  // tolerate stray whitespace
            v.push_back(encode(c));
        }
        return v;
    };
    SeqPair sp;
    sp.q = encode_seq(qline);
    sp.t = encode_seq(tline);
    sp.m = static_cast<int>(sp.q.size());
    sp.n = static_cast<int>(sp.t.size());
    if (sp.m == 0 || sp.n == 0) throw std::runtime_error("empty sequence in " + path);
    return sp;
}

void sw_cpu(const SeqPair& sp, std::vector<int>& H) {
    const int m = sp.m, n = sp.n;
    const int W = n + 1;                          // row stride (columns incl. the 0-column)
    H.assign(static_cast<std::size_t>(m + 1) * W, 0);   // row 0 and col 0 start at 0

    // Fill row by row, left to right. Each cell takes the best of: align the two
    // residues (diagonal), a gap in the target (up), a gap in the query (left),
    // or restart a fresh local alignment (the 0 floor -- this is what makes it
    // LOCAL rather than global).
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            const int s = (sp.q[i - 1] == sp.t[j - 1]) ? MATCH : MISMATCH;
            const int diag = H[(i - 1) * W + (j - 1)] + s;
            const int up   = H[(i - 1) * W + j]       + GAP;
            const int left = H[i * W + (j - 1)]       + GAP;
            int v = 0;
            if (diag > v) v = diag;
            if (up   > v) v = up;
            if (left > v) v = left;
            H[i * W + j] = v;
        }
    }
}

Alignment traceback(const SeqPair& sp, const std::vector<int>& H) {
    const int m = sp.m, n = sp.n, W = n + 1;

    // Find the max cell (the local-alignment endpoint). Deterministic: the first
    // cell achieving the max in row-major scan order.
    Alignment a;
    for (int i = 1; i <= m; ++i)
        for (int j = 1; j <= n; ++j)
            if (H[i * W + j] > a.score) { a.score = H[i * W + j]; a.end_i = i; a.end_j = j; }

    // Walk back from the max cell until we hit a 0, reconstructing columns.
    // Preference order diagonal > up > left makes the path deterministic.
    int i = a.end_i, j = a.end_j;
    std::string q, mk, t;
    while (i > 0 && j > 0 && H[i * W + j] > 0) {
        const int s = (sp.q[i - 1] == sp.t[j - 1]) ? MATCH : MISMATCH;
        if (H[i * W + j] == H[(i - 1) * W + (j - 1)] + s) {        // diagonal
            const char qc = ALPHABET[sp.q[i - 1]], tc = ALPHABET[sp.t[j - 1]];
            q += qc; t += tc; mk += (qc == tc) ? '|' : '.';
            --i; --j;
        } else if (H[i * W + j] == H[(i - 1) * W + j] + GAP) {      // up: gap in target
            q += ALPHABET[sp.q[i - 1]]; t += '-'; mk += ' ';
            --i;
        } else {                                                    // left: gap in query
            q += '-'; t += ALPHABET[sp.t[j - 1]]; mk += ' ';
            --j;
        }
    }
    // We built the alignment end-to-start; reverse to read 5'->3'.
    std::reverse(q.begin(), q.end());
    std::reverse(mk.begin(), mk.end());
    std::reverse(t.begin(), t.end());
    a.q_line = q; a.m_line = mk; a.t_line = t;
    a.length = static_cast<int>(q.size());
    a.identities = static_cast<int>(std::count(mk.begin(), mk.end(), '|'));
    return a;
}
