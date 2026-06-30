// ===========================================================================
// src/coevolution.h  --  Shared (host + device) coevolution primitives
// ---------------------------------------------------------------------------
// Project 2.25 : Coevolutionary Contact Prediction & MSA Transformer
//
// WHAT THIS PROJECT COMPUTES
//   Proteins that fold into a 3-D structure place pairs of residues (amino-acid
//   positions) physically close to each other. Over evolution, two residues that
//   TOUCH in 3-D tend to mutate in a CORRELATED way: a destabilizing change at
//   position i is "compensated" by a change at the contacting position j. If we
//   line up many homologous sequences in a Multiple Sequence Alignment (MSA) --
//   N rows (sequences) x L columns (alignment positions) -- then COLUMNS i and j
//   that coevolve are statistically dependent. Quantifying that dependence for
//   EVERY pair (i, j) yields an L x L "coevolution score" matrix; its largest
//   off-diagonal entries predict which residues are in CONTACT, which in turn
//   drives protein-structure prediction (the idea behind EVcouplings / CCMpred,
//   and a core signal exploited by AlphaFold's MSA representation).
//
//   This teaching version uses the foundational, exact estimator:
//     * MUTUAL INFORMATION (MI) between two columns, computed from integer counts
//       of how often each (amino-acid a in col i, amino-acid b in col j) pair
//       co-occurs across the N sequences. MI(i,j) = sum_ab p_ab log( p_ab /
//       (p_i(a) p_j(b)) ), measured in nats.
//     * AVERAGE PRODUCT CORRECTION (APC), the standard background subtraction
//       that removes phylogenetic / entropic bias so the TRUE coevolution signal
//       stands out (Dunn et al. 2008). This is the same correction CCMpred and
//       EVcouplings apply on top of their (fancier) raw scores.
//
//   THEORY.md walks the science -> math -> algorithm -> GPU mapping in full. The
//   "real world" section there explains how PLMC/DCA (EVcouplings) and the MSA
//   Transformer (ESM-MSA-1b) go beyond pairwise MI; here we build the exact,
//   verifiable pairwise core that every one of those methods starts from.
//
// WHY A GPU  (the pattern: "score all L x L column pairs, each independent")
//   There are L*(L-1)/2 column pairs and each MI is independent of the others.
//   For a real protein L is a few hundred to >1000, so that is 10^4 - 10^6
//   independent reductions over N sequences -- the textbook "many independent
//   jobs" GPU pattern (PATTERNS.md section 1; exemplars 1.12 Tanimoto, 12.01
//   spectral search). We give each column PAIR its own GPU thread. The catalog's
//   CCMpred reference uses custom CUDA kernels for exactly this per-pair work.
//
// DETERMINISM  (PATTERNS.md sections 2-4)
//   The heavy lifting is INTEGER COUNTING (how many sequences have residue a in
//   column i and b in column j). Integer adds commute, so the counts are
//   bit-identical on CPU and GPU regardless of thread order. MI is then a fixed
//   arithmetic function of those exact counts, evaluated by the SAME
//   __host__ __device__ code (the CV_HD functions below) on both sides -> the
//   GPU MI matrix equals the CPU MI matrix to ~machine precision (we verify to
//   1e-9; see THEORY.md "How we verify correctness"). No floating-point atomics
//   are used anywhere, so there is nothing order-dependent to diverge.
//
//   Keep CUDA-only constructs (e.g. __global__) OUT of this header so the plain
//   host compiler can include it for reference_cpu.cpp (PATTERNS.md section 2).
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cu, main.cu.
// ===========================================================================
#pragma once

#include <cmath>      // std::log
#include <cstddef>    // std::size_t
#include <cstdint>    // fixed-width integer types

// CV_HD marks a function as callable from BOTH host and device when compiled by
// nvcc, and as an ordinary inline function when compiled by the host compiler
// (cl.exe / g++) for reference_cpu.cpp. This is the HD-macro idiom that gives us
// byte-for-byte CPU/GPU parity (PATTERNS.md section 2).
#ifdef __CUDACC__
#define CV_HD __host__ __device__
#else
#define CV_HD
#endif

// ---------------------------------------------------------------------------
// THE ALPHABET
//   We model an MSA over a fixed alphabet of Q symbols: the 20 standard amino
//   acids plus a gap '-'. So Q = 21. Each MSA cell is stored as a small integer
//   "token" in [0, Q). Gap is token 20. We treat the gap as just another symbol
//   (the simplest convention); production tools often down-weight or mask gaps,
//   which THEORY.md notes as an exercise.
// ---------------------------------------------------------------------------
static const int CV_Q = 21;   // alphabet size: 20 amino acids + gap

// ---------------------------------------------------------------------------
// cv_token_of_aa: map a single FASTA amino-acid character to a token in [0,CV_Q).
//   The canonical order is the standard 1-letter code "ACDEFGHIKLMNPQRSTVWY".
//   Anything not in that set -- a gap '-', '.', 'X' (unknown), 'B'/'Z'/'U'/'O'
//   ambiguity codes, or lowercase -- maps to the GAP token (CV_Q-1 = 20). This
//   keeps the loader simple and deterministic; THEORY.md discusses why gap
//   handling matters for real coevolution.
//
//   It is CV_HD so the loader (host) and any device-side tokenizer share it; in
//   practice we tokenize on the host once and upload integer tokens.
// ---------------------------------------------------------------------------
CV_HD inline int cv_token_of_aa(char c) {
    // Branch-free-ish lookup written as an explicit switch for READABILITY: a
    // learner can see exactly which letter maps to which index. The compiler
    // turns this into a jump table; clarity costs us nothing here.
    switch (c) {
        case 'A': return 0;   case 'C': return 1;   case 'D': return 2;
        case 'E': return 3;   case 'F': return 4;   case 'G': return 5;
        case 'H': return 6;   case 'I': return 7;   case 'K': return 8;
        case 'L': return 9;   case 'M': return 10;  case 'N': return 11;
        case 'P': return 12;  case 'Q': return 13;  case 'R': return 14;
        case 'S': return 15;  case 'T': return 16;  case 'V': return 17;
        case 'W': return 18;  case 'Y': return 19;
        default:  return CV_Q - 1;   // gap / unknown / ambiguity -> token 20
    }
}

// ---------------------------------------------------------------------------
// cv_mi_from_counts: Mutual Information (in NATS) of one column pair (i, j),
//   computed from EXACT INTEGER COUNTS gathered over the N sequences.
//
//   INPUTS (all integer tallies over the N aligned sequences):
//     pair  : Q*Q matrix, pair[a*Q + b] = #sequences with token a in column i
//             AND token b in column j.                       (the joint counts)
//     ci    : length-Q vector, ci[a] = #sequences with token a in column i.
//     cj    : length-Q vector, cj[b] = #sequences with token b in column j.
//     N     : number of sequences (rows of the MSA). N > 0.
//
//   MATH:
//     Let p_ab = pair[a,b]/N        (joint probability of (a in i, b in j))
//         p_i(a) = ci[a]/N,  p_j(b) = cj[b]/N   (the two marginals)
//     MI(i,j) = sum over a,b of  p_ab * ln( p_ab / (p_i(a) * p_j(b)) ),
//     summing ONLY over cells with p_ab > 0 (the 0*ln(0)=0 convention).
//   MI >= 0; MI = 0 iff the columns are statistically independent (no
//   coevolution). Larger MI = stronger statistical coupling.
//
//   WHY THIS FORM IS DETERMINISTIC AND CPU/GPU-IDENTICAL:
//     Every input is an exact integer. We accumulate the sum in DOUBLE precision
//     and -- crucially -- iterate the (a, b) cells in the SAME fixed order on
//     host and device (a outer, b inner). Floating-point addition is not
//     associative, but it IS deterministic for a FIXED order, so both sides
//     produce identical bits. (We verify to 1e-9 to allow for the host compiler
//     vs. nvcc evaluating std::log with a 1-ulp difference -- see THEORY.md.)
//
//   COMPLEXITY: O(Q*Q) = O(441) per pair, independent of N once counts exist.
//   Returns MI in nats (natural log). Caller may convert to bits (/ln 2).
// ---------------------------------------------------------------------------
CV_HD inline double cv_mi_from_counts(const uint32_t* pair,
                                      const uint32_t* ci,
                                      const uint32_t* cj,
                                      int N) {
    const double invN = 1.0 / static_cast<double>(N);   // 1/N, computed once
    double mi = 0.0;                                     // running MI accumulator
    // Iterate the joint table in a FIXED (a outer, b inner) order so the partial
    // sums add up identically on CPU and GPU.
    for (int a = 0; a < CV_Q; ++a) {
        const uint32_t ca = ci[a];          // count of symbol a in column i
        if (ca == 0u) continue;             // symbol absent -> contributes nothing
        const double pia = static_cast<double>(ca) * invN;   // marginal p_i(a)
        for (int b = 0; b < CV_Q; ++b) {
            const uint32_t nab = pair[a * CV_Q + b];   // joint count of (a,b)
            if (nab == 0u) continue;        // empty cell: 0 * ln(0) := 0
            const double pab = static_cast<double>(nab) * invN;          // p_ab
            const double pjb = static_cast<double>(cj[b]) * invN;        // p_j(b)
            // p_ab * ln( p_ab / (p_i(a) p_j(b)) ). Because nab>0, both ca>0 and
            // cj[b]>0 here (a joint occurrence implies each marginal occurs), so
            // the argument of log is finite and positive -- no NaN/Inf possible.
            mi += pab * std::log(pab / (pia * pjb));
        }
    }
    return mi;   // nats; guaranteed >= 0 up to rounding
}
