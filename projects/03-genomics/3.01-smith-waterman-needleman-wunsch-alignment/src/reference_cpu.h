// ===========================================================================
// src/reference_cpu.h  --  Data model, scoring, CPU reference & traceback
// ---------------------------------------------------------------------------
// Project 3.01 : Smith-Waterman / Needleman-Wunsch Alignment
//
// WHAT THIS PROJECT COMPUTES
//   The optimal LOCAL alignment of two sequences (Smith-Waterman). We fill a
//   dynamic-programming score matrix H, where H[i][j] is the best local-
//   alignment score ending at query position i and target position j:
//
//     H[i][j] = max( 0,
//                    H[i-1][j-1] + s(q_i, t_j),   // align q_i with t_j
//                    H[i-1][j]   + GAP,           // gap in target (skip q_i)
//                    H[i][j-1]   + GAP )          // gap in query  (skip t_j)
//
//   The alignment SCORE is the maximum cell; the alignment itself is recovered
//   by TRACEBACK from that cell back to the first 0. (Needleman-Wunsch global
//   alignment is the same recurrence WITHOUT the max-with-0 and with the score
//   read from the bottom-right corner -- see THEORY.md.)
//
// WHY A GPU
//   H[i][j] depends on its top, left, and top-left neighbours, so it looks
//   serial. But all cells on one ANTI-DIAGONAL (i+j constant) depend only on
//   the two previous anti-diagonals, so they are mutually independent and can
//   be computed in parallel -- the "wavefront" the GPU exploits (kernels.cu).
//
//   This pure-C++ header is shared by reference_cpu.cpp, main.cu, and kernels.cu
//   (the constants are used in the device kernel). No CUDA syntax here.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Linear-gap scoring (integers keep CPU and GPU bit-identical). DNA alphabet.
constexpr int MATCH    =  2;   // reward for aligning identical residues
constexpr int MISMATCH = -1;   // penalty for aligning different residues
constexpr int GAP      = -2;   // penalty for a single-residue insertion/deletion
constexpr char ALPHABET[] = "ACGT";  // index 0..3 <-> nucleotide

// A loaded problem: two sequences, encoded as 0..3 indices into ALPHABET.
struct SeqPair {
    int m = 0, n = 0;             // query length (m), target length (n)
    std::vector<uint8_t> q;       // [m] encoded query
    std::vector<uint8_t> t;       // [n] encoded target
};

// One recovered local alignment, with display strings (query / match / target).
struct Alignment {
    int score = 0;                // best local-alignment score (= max cell)
    int end_i = 0, end_j = 0;     // 1-based matrix cell where the max sits
    int length = 0;               // number of alignment columns
    int identities = 0;           // columns where the two residues match
    std::string q_line, m_line, t_line;   // aligned query / markers / target
};

// Load two sequences (query on line 1, target on line 2) from a text file.
// Non-ACGT characters are rejected. Throws std::runtime_error on error.
SeqPair load_sequences(const std::string& path);

// CPU reference: fill the Smith-Waterman matrix H (size (m+1)*(n+1), row-major,
// H[i*(n+1)+j]). Row 0 and column 0 are 0. This is the trusted baseline the GPU
// wavefront is checked against (every cell must match).
void sw_cpu(const SeqPair& sp, std::vector<int>& H);

// Recover the optimal local alignment from a filled matrix H (host-side; done
// once, on whichever matrix we display). Deterministic tie-breaking: scan picks
// the first max cell (lowest i, then j); traceback prefers diagonal > up > left.
Alignment traceback(const SeqPair& sp, const std::vector<int>& H);
