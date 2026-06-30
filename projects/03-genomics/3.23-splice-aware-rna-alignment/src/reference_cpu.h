// ===========================================================================
// src/reference_cpu.h  --  Data model, scoring, and the SHARED splice-aware
//                          alignment recurrence (CPU reference + GPU twin)
// ---------------------------------------------------------------------------
// Project 3.23 : Splice-Aware RNA Alignment   (REDUCED-SCOPE teaching version)
//
// WHAT THIS PROJECT COMPUTES (and what it deliberately does NOT)
//   A splice-aware aligner (STAR, HISAT2, minimap2 -ax splice) maps an RNA-seq
//   read onto a genome so that the read can JUMP OVER an intron: the mature
//   mRNA the read came from has had its introns spliced out, so a single read
//   often spans an exon-exon junction. Where a plain aligner would be forced to
//   pay a huge per-base gap penalty across the (kilobase-long) intron, a
//   splice-aware aligner instead pays ONE fixed "intron-open" penalty and emits
//   a CIGAR 'N' operation ("skipped region from the reference"), rewarded when
//   the skipped region begins with the canonical GT donor and ends with the AG
//   acceptor dinucleotides (the "GT-AG rule" of ~99% of human introns).
//
//   The full production problem -- suffix-array seeding over a 3 Gb genome, a
//   graph FM-index, chaining of many seeds, long-read wavefront extension -- is
//   research-grade and far beyond one didactic file. So, per CLAUDE.md §13, we
//   ship the SCIENTIFIC HEART of it as a self-contained, verifiable kernel:
//
//     >>> the SPLICED dynamic-programming alignment of a read against a short
//         reference "gene" with one or more introns, scoring the intron jump
//         with a canonical-splice-site bonus and emitting an N-aware CIGAR. <<<
//
//   This is exactly the "banded SW for exon extension + CIGAR with N (intron)
//   operations + splice-site scoring" the catalog names as the GPU target. We
//   align MANY reads at once -- the GPU pattern is "one independent alignment
//   per thread block" (batched jobs), which is how real spliced aligners scale.
//
// THE RECURRENCE (defined ONCE, below, as a __host__ __device__ core so the CPU
//   reference and the GPU kernel compute byte-for-byte identical integer scores;
//   see docs/PATTERNS.md §2). For a read q[1..M] and reference r[1..N]:
//
//     H[i][j] = max(
//        0,                                   // local restart (Smith-Waterman)
//        H[i-1][j-1] + s(q_i, r_j),           // (M)atch / mismatch  -- diagonal
//        H[i-1][j]   + GAP,                    // (I)nsertion in read -- up
//        H[i][j-1]   + GAP,                    // (D)eletion / small ref gap -- left
//        E[i][j] )                             // (N) intron-spliced match
//
//   where the intron term E[i][j] aligns read base i to reference base j as a
//   MATCH, with a spliced-out intron r[k+1..j-1] immediately preceding column j
//   (so read base i-1 was aligned at the earlier column k):
//
//     E[i][j] = max over donor columns k of
//                 H[i-1][k] + s(q_i, r_j) + INTRON_OPEN + canonical_bonus(r,k,j)
//
//   We bound k to a window (MAX_INTRON) for tractability, exactly as real
//   aligners cap intron length. canonical_bonus rewards r[k+1..k+2]=="GT"
//   (donor) and r[j-2..j-1]=="AG" (acceptor). See THEORY.md for the derivation.
//
// WHY A GPU
//   Each read is aligned independently against the reference, so a batch of R
//   reads is R independent DP problems -- embarrassingly parallel across reads.
//   We give each read its own thread block (one block fills that read's whole DP
//   table). Across a sequencing run that is millions of independent blocks.
//
//   This pure-C++ header is shared by reference_cpu.cpp (host compiler) AND
//   kernels.cu (nvcc): the HD-decorated functions below are the single source of
//   truth for the math, so CPU and GPU agree exactly (integer scores).
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// HD: the host/device decorator macro (docs/PATTERNS.md §2). When this header
// is pulled into a .cu file, nvcc defines __CUDACC__ and we mark the shared
// math as callable from BOTH the host and a kernel. When the plain C++ compiler
// builds reference_cpu.cpp, __CUDACC__ is undefined and HD expands to nothing,
// so the very same source compiles as ordinary C++. One formula, two backends.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ---------------------------------------------------------------------------
// Scoring (all INTEGER, so CPU and GPU sums are associative and bit-identical;
// docs/PATTERNS.md §3/§4 -- this lets us verify with an EXACT, == 0 tolerance).
// The values are the usual didactic SW DNA scheme plus splice terms.
// ---------------------------------------------------------------------------
constexpr int MATCH        =  2;   // reward for aligning identical bases
constexpr int MISMATCH     = -1;   // penalty for a substitution
constexpr int GAP          = -2;   // penalty for a one-base indel (I or D)
constexpr int INTRON_OPEN  = -6;   // FLAT cost of one intron jump (an 'N' run),
                                   //   independent of intron length -- this is
                                   //   the whole point of splice-aware scoring:
                                   //   skipping a 40-base intron costs this ONCE,
                                   //   versus ~40 * GAP = -80 for a base-by-base
                                   //   gap. A SINGLE canonical jump nets
                                   //   INTRON_OPEN + CANON_BONUS = -2, so an
                                   //   intron is a (small) PENALTY, never a
                                   //   reward -- otherwise the optimiser would
                                   //   fabricate introns to harvest the bonus.
constexpr int CANON_BONUS  =  4;   // reward when the jump uses a canonical
                                   //   GT...AG intron (donor GT + acceptor AG).
                                   //   Tuned so canonical < 0 (see above) but a
                                   //   canonical jump beats the non-canonical
                                   //   one (-2 vs -6), steering the aligner to
                                   //   the biologically correct splice site.
constexpr int MIN_INTRON   =  4;   // shortest gap we will call an intron (below
                                   //   this, ordinary gaps/mismatches win).
constexpr int MAX_INTRON   = 64;   // longest intron we search for (a "band" on
                                   //   the N move; real tools cap this too, e.g.
                                   //   STAR --alignIntronMax). Keeps it O(N·band).

constexpr char ALPHABET[] = "ACGT";  // base code 0..3 <-> nucleotide letter

// ---------------------------------------------------------------------------
// sub_score(a, b): substitution score for aligning base code a against code b.
//   Shared by CPU and GPU. Bases are 0..3; equal -> MATCH, else MISMATCH.
// ---------------------------------------------------------------------------
HD inline int sub_score(uint8_t a, uint8_t b) {
    return (a == b) ? MATCH : MISMATCH;
}

// ---------------------------------------------------------------------------
// is_canonical_intron(r, n, k, j):
//   Geometry (the SAME everywhere): an intron move connects "read base i-1
//   aligned at reference column k" to "read base i aligned at reference column
//   j", skipping the reference substring r[k+1 .. j-1] (1-based columns). The
//   canonical "GT-AG rule": that skipped intron STARTS with the GT donor
//   dinucleotide (columns k+1, k+2) and ENDS with the AG acceptor (columns
//   j-2, j-1). ~99% of human introns obey it.
//
//   r[] is 0-based of length n; 1-based column c maps to r[c-1]. Returns true
//   iff both dinucleotides are canonical AND the intron is long enough to be
//   plausible (>= MIN_INTRON).
// ---------------------------------------------------------------------------
HD inline bool is_canonical_intron(const uint8_t* r, int n, int k, int j) {
    const int intron_len = (j - 1) - (k + 1) + 1;   // bases r[k+1..j-1] removed
    if (intron_len < MIN_INTRON) return false;
    // Donor GT = (G=2, T=3) at intron columns k+1, k+2 -> 0-based r[k], r[k+1].
    const int dk0 = k;          // 0-based index of 1-based column k+1
    const int dk1 = k + 1;      // 0-based index of 1-based column k+2
    // Acceptor AG = (A=0, G=2) at intron columns j-2, j-1 -> 0-based r[j-3], r[j-2].
    const int ak0 = j - 3;      // 0-based index of 1-based column j-2
    const int ak1 = j - 2;      // 0-based index of 1-based column j-1
    if (dk0 < 0 || ak1 >= n || ak0 < 0) return false;   // bounds guard
    const bool donor    = (r[dk0] == 2 /*G*/) && (r[dk1] == 3 /*T*/);
    const bool acceptor = (r[ak0] == 0 /*A*/) && (r[ak1] == 2 /*G*/);
    return donor && acceptor;
}

// ---------------------------------------------------------------------------
// intron_score(r, n, k, j):
//   The score CONTRIBUTION of crossing an intron r[k+1..j-1] (NOT counting the
//   H[i-1][k] we come from nor the substitution at column j). It is the flat
//   INTRON_OPEN, plus CANON_BONUS iff the skipped region is a canonical GT-AG
//   intron. Returns a large-negative sentinel for a span too short to be an
//   intron, so the caller can cheaply reject it.
// ---------------------------------------------------------------------------
HD inline int intron_score(const uint8_t* r, int n, int k, int j) {
    const int intron_len = (j - 1) - (k + 1) + 1;
    if (intron_len < MIN_INTRON) return -1000000;   // "not an intron" sentinel
    int sc = INTRON_OPEN;
    if (is_canonical_intron(r, n, k, j)) sc += CANON_BONUS;
    return sc;
}

// ---------------------------------------------------------------------------
// cell_recurrence(...): compute ONE DP cell H[i][j] given its already-computed
//   neighbours and the reference window for the intron (N) move. This is the
//   single place the recurrence lives -- the CPU loop and the GPU kernel both
//   call it, so they cannot drift apart.
//
//   Inputs (all 1-based i,j into read q[1..M] and reference r[1..N]):
//     qi       : encoded read base at position i  (q[i-1])
//     rj       : encoded ref  base at position j  (r[j-1])
//     h_diag   : H[i-1][j-1]   (already final)
//     h_up     : H[i-1][j]     (already final)
//     h_left   : H[i][j-1]     (already final)
//     prev_row : pointer to the PREVIOUS row H[i-1][*] (length N+1), so the
//                intron move can read H[i-1][k] for every candidate donor k.
//     r        : the encoded reference array (length n), 0-based
//     n        : reference length N
//     j        : this cell's 1-based column
//   Returns H[i][j].
//
//   The intron (N) move is on the DIAGONAL: read base i MATCHES reference base
//   j (the +s), and immediately before column j there is an intron r[k+1..j-1]
//   whose left side connects to read base i-1 aligned at column k. So the
//   candidate is  H[i-1][k] + s(q_i, r_j) + intron_score(k, j).  We scan donor
//   columns k in [j-1-MAX_INTRON, j-1-MIN_INTRON] -- a bounded "band" exactly as
//   real aligners cap intron length (e.g. STAR --alignIntronMax). Every cell
//   read (the three classic neighbours, finalised; and prev_row[k] from the
//   already-complete row i-1) is final, so the cell is a pure function of
//   finished state -- which is why a block can fill row i once row i-1 is done.
// ---------------------------------------------------------------------------
HD inline int cell_recurrence(uint8_t qi, uint8_t rj,
                              int h_diag, int h_up, int h_left,
                              const int* prev_row, const uint8_t* r, int n, int j) {
    const int s = sub_score(qi, rj);             // match/mismatch at column j
    int v = 0;                                   // local restart floor (SW)
    int cand;
    cand = h_diag + s;   if (cand > v) v = cand;   // M (diagonal match/mismatch)
    cand = h_up   + GAP; if (cand > v) v = cand;   // I (read insertion -- up)
    cand = h_left + GAP; if (cand > v) v = cand;   // D (one-base ref del -- left)

    // N (intron) move: q_i matches r_j, preceded by a spliced-out intron.
    const int k_hi = j - 1 - MIN_INTRON;         // closest allowed donor column
    int k_lo = j - 1 - MAX_INTRON;               // farthest allowed donor column
    if (k_lo < 0) k_lo = 0;
    for (int k = k_lo; k <= k_hi; ++k) {
        const int is = intron_score(r, n, k, j);
        if (is <= -1000000) continue;            // too short to be an intron
        cand = prev_row[k] + s + is;             // H[i-1][k] + match + intron
        if (cand > v) v = cand;
    }
    return v;
}

// ===========================================================================
//  Host-side data model and CPU reference API (NOT shared with the device; the
//  device only needs the HD math above + the raw arrays).
// ===========================================================================

// A batch of reads to align against a single reference "gene model".
struct ReadBatch {
    int n = 0;                          // reference length (N)
    std::vector<uint8_t> ref;           // [n] encoded reference bases (0..3)
    int num_reads = 0;                  // number of reads R
    int read_len = 0;                   // max read length M (rows in each table)
    std::vector<uint8_t> reads;         // [R*read_len] encoded reads, row-major
    std::vector<int> read_lens;         // [R] true length of each read (<= read_len)
};

// The result of aligning ONE read: the best spliced-local-alignment score and
// the endpoint cell, enough to reconstruct a CIGAR on the host.
struct AlignResult {
    int score = 0;                      // best cell value in the read's DP table
    int end_i = 0, end_j = 0;           // 1-based (read, ref) cell of that best
};

// Load a batch: line 1 is the reference; each subsequent line is one read.
// Bases must be ACGT (RNA 'U' accepted as 'T'). Throws on malformed input.
ReadBatch load_batch(const std::string& path);

// CPU reference: align EVERY read in the batch, filling out[r] with its result.
// This is the trusted serial baseline the GPU is checked against. The full
// per-read DP tables are returned via H_all (size R*(M+1)*(N+1)) so the host can
// run a deterministic traceback for display (see main.cu).
void align_batch_cpu(const ReadBatch& b,
                     std::vector<AlignResult>& out,
                     std::vector<int>& H_all);

// Reconstruct a CIGAR-with-N string for one read from its filled DP table.
// Host-only (traceback is serial and is not the GPU teaching point). Produces
// e.g. "12M48N9M": 12 matches, a 48-base intron skip, then 9 matches. Also
// reports the number of introns crossed and matched columns via out-params.
std::string traceback_cigar(const ReadBatch& b, int read_index,
                            const std::vector<int>& H_all,
                            const AlignResult& res,
                            int& out_introns, int& out_matched);
