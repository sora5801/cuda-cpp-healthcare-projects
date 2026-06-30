// ===========================================================================
// src/reference_cpu.h  --  Data model, scoring, CPU reference for MSA
// ---------------------------------------------------------------------------
// Project 3.8 : Multiple Sequence Alignment (MSA)
//
// WHAT THIS PROJECT COMPUTES
//   Given N short DNA sequences, build a MULTIPLE alignment of all of them at
//   once. We use the classic three-stage PROGRESSIVE recipe (the ClustalW /
//   MAFFT family), specialised to the simplest didactic variant:
//
//     STAGE 1  pairwise distance matrix  D[a][b]  for every pair (a,b):
//              D[a][b] = 1 - (NW global-alignment score / self-alignment score),
//              i.e. how DISSIMILAR two sequences are. Each entry needs one
//              Needleman-Wunsch (NW) global alignment -> O(N^2) DP problems.
//              *** This O(N^2) phase is the GPU teaching point (kernels.cu):
//                  the pairwise alignments are mutually INDEPENDENT, so we give
//                  each pair its own CUDA thread block. ***
//
//     STAGE 2  guide order via CENTER-STAR: pick the "center" sequence c whose
//              total distance to all others is smallest; it is the most
//              representative sequence. (Full ClustalW builds a neighbor-joining
//              tree; center-star is the simplest correct guide and is exactly the
//              reduction CUK-Band (2024) puts on the GPU -- see THEORY.md.)
//
//     STAGE 3  progressive assembly: align every other sequence to the center
//              with NW, then merge all those pairwise alignments into one matrix
//              using the center as the common coordinate frame ("once a gap,
//              always a gap"). That is the multiple alignment; we grade it with
//              the Sum-of-Pairs (SP) column score.
//
//   The per-cell DP recurrence and the integer score it produces live in
//   nw_core.h as `__host__ __device__` code, so the CPU reference and the GPU
//   kernel run the SAME math (PATTERNS.md §2). THIS header is pure C++ (no CUDA),
//   so the plain host compiler can build reference_cpu.cpp from it.
//
// WHY A GPU
//   STAGE 1 dominates for large N: N(N-1)/2 independent NW alignments, each an
//   O(L^2) DP fill. That is "embarrassingly parallel over pairs" -- the catalog
//   pattern "one CUDA thread block per pairwise alignment". STAGES 2-3 are cheap
//   host bookkeeping and are not the GPU lesson.
//
// READ THIS AFTER: nw_core.h (the shared recurrence). Then kernels.cuh.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "nw_core.h"   // NW_MATCH/MISMATCH/GAP, nw_score_core() -- shared by CPU & GPU

// ---------------------------------------------------------------------------
// SeqSet: a loaded problem -- N sequences over the DNA alphabet, encoded 0..3.
//   We store every residue in ONE flat buffer `data` (the sequences concatenated
//   back to back), with `off[i]` the start index of sequence i and `len[i]` its
//   length. This "ragged array as flat buffer + offsets" layout is exactly what
//   we upload to the GPU: a single contiguous device allocation is far friendlier
//   (one cudaMalloc, one cudaMemcpy, coalesced access) than N separate ones.
// ---------------------------------------------------------------------------
struct SeqSet {
    int n = 0;                       // number of sequences
    int max_len = 0;                 // longest sequence length (DP buffer sizing)
    std::vector<std::string> names;  // [n] FASTA header text (display only)
    std::vector<uint8_t> data;       // concatenated residues, each encoded 0..3
    std::vector<int> off;            // [n] start index of sequence i in `data`
    std::vector<int> len;            // [n] length of sequence i

    // Convenience: pointer to the first residue of sequence i in the flat buffer.
    const uint8_t* seq(int i) const { return data.data() + off[i]; }
};

// ---------------------------------------------------------------------------
// MSA: one fully-built multiple alignment -- `n` rows of equal `width`, each a
// string over {A,C,G,T,-}. Plus the Sum-of-Pairs score and the chosen center.
// ---------------------------------------------------------------------------
struct MSA {
    int n = 0;                       // number of rows (= number of sequences)
    int width = 0;                   // number of alignment columns (all rows equal)
    int center = 0;                  // index of the center-star sequence
    long long sp_score = 0;          // Sum-of-Pairs score of the whole alignment
    std::vector<std::string> rows;   // [n] aligned rows; '-' marks a gap
};

// The DNA alphabet: index 0..3 <-> nucleotide character (for decoding to text).
extern const char DNA_ALPHABET[5];   // "ACGT" + NUL

// Load N sequences from a tiny multi-FASTA file (">name" line, then sequence
// line(s) until the next '>' or EOF). Non-ACGT letters are rejected. Throws
// std::runtime_error on any problem so demos fail loudly, not silently.
SeqSet load_fasta(const std::string& path);

// CPU reference, STAGE 1: fill the full pairwise score + distance matrices.
//   raw_score : [n*n] flat, row-major; raw_score[a*n+b] = NW global score of
//               sequences a vs b. Symmetric; diagonal is each self-score. This is
//               the EXACT integer quantity the GPU also computes, so the GPU/CPU
//               check is over integers (bit-exact), not the derived float.
//   D         : [n*n] flat; D[a*n+b] = 1 - score/self in [0,1] (0 = identical).
void distance_matrix_cpu(const SeqSet& s,
                         std::vector<int>& raw_score,
                         std::vector<double>& D);

// CPU reference, STAGES 2-3: from the distance matrix choose the center-star
// sequence and progressively assemble the full multiple alignment. Pure host
// bookkeeping (deterministic). Shared by the CPU and GPU paths because the matrix
// it consumes is identical on both. Returns the assembled MSA (sp_score filled).
MSA build_msa(const SeqSet& s, const std::vector<double>& D);

// Sum-of-Pairs score of an assembled alignment: for each column, sum the pairwise
// residue scores over all row pairs (a gap-vs-gap pair scores 0, gap-vs-residue
// scores NW_GAP, else nw_subst). The headline "alignment quality" number.
long long sum_of_pairs(const MSA& m);
