// ===========================================================================
// src/hmm_core.h  --  The shared profile-HMM model + the ONE TRUE Viterbi row
// ---------------------------------------------------------------------------
// Project 2.13 : MSA Generation Acceleration
//
// WHY THIS HEADER EXISTS (the most important idiom in the repo -- PATTERNS.md §2)
//   Multiple-sequence-alignment generation (HHblits / Jackhmmer in the AlphaFold2
//   pipeline) is, at its core, ONE expensive inner loop run BILLIONS of times:
//   "score a profile hidden Markov model (HMM) against a database sequence with
//   the Viterbi dynamic program." Both our CPU reference (reference_cpu.cpp) and
//   our GPU kernel (kernels.cu) must perform the *byte-for-byte identical* DP so
//   that verification is EXACT, not approximate. The cleanest way to guarantee
//   that is to write the per-row recurrence exactly once, here, as a
//   `__host__ __device__` inline function, and have both sides call it.
//
//   This header therefore contains:
//     * the data model        (Plan7-style profile HMM + a packed database)
//     * the scoring constants  (fixed-point scale, the integer NEG_INF floor)
//     * the per-row recurrence  viterbi_step()   <-- the one true formula
//   ...and NOTHING CUDA-specific (no __global__, no kernel launches), so the
//   plain host compiler can include it for reference_cpu.cpp.
//
// THE FIXED-POINT TRICK (why integers, not floats)
//   Viterbi scores are sums of log-odds. If we accumulated them in float, the GPU
//   (which fuses multiply-adds and may reorder) and the host compiler could
//   diverge by ~1e-6 per cell, growing over a long sequence -- so CPU != GPU and
//   the demo's stdout would not be reproducible (PATTERNS.md §3, §4). Instead we
//   pre-quantise every log-odds to a SCALED INTEGER (log-odds * SCORE_SCALE,
//   rounded). The whole DP is then integer max/add: associative, deterministic,
//   and identical on both sides -> we can verify with tolerance == 0.
//
// READ THIS BEFORE: reference_cpu.h, kernels.cuh. The science/math is in
// ../THEORY.md ("The algorithm", "GPU mapping", "Numerical considerations").
// ===========================================================================
#pragma once

#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// HD: the host/device decoration macro.
//   When this header is pulled in by nvcc (which defines __CUDACC__), we tag the
//   shared functions `__host__ __device__` so the SAME source compiles for both
//   the CPU and the GPU. When pulled in by the plain host compiler (for
//   reference_cpu.cpp), those keywords do not exist, so HD expands to nothing.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ---------------------------------------------------------------------------
// The amino-acid alphabet. Proteins are strings over 20 standard residues; we
// add one slot (index 20) for "unknown / non-standard" (X, B, Z, gaps, ...) so a
// messy database sequence can never index out of bounds. ALPHABET_SIZE = 21.
// ---------------------------------------------------------------------------
constexpr int ALPHABET_SIZE = 21;   // 20 amino acids + 1 catch-all "X"

// ---------------------------------------------------------------------------
// Fixed-point scoring.
//   SCORE_SCALE : multiply a (dimensionless) log-odds score in nats by this and
//                 round to the nearest int to get our integer score unit. 1000
//                 keeps ~3 decimal digits of the log-odds -- ample for ranking
//                 hits, and small enough that a few-hundred-residue path sum
//                 (< ~10^6) never overflows a 32-bit int.
//   NEG_INF     : the integer stand-in for "impossible" (log 0). Chosen far below
//                 any reachable score but far enough from INT_MIN that adding a
//                 transition penalty to it cannot underflow/wrap a 32-bit int.
//                 We test `<= NEG_INF/2` to detect "still impossible" robustly.
// ---------------------------------------------------------------------------
constexpr int SCORE_SCALE = 1000;          // log-odds (nats) -> integer units
constexpr int NEG_INF     = -1000000000;   // "impossible state" (~ -1e9)

// ---------------------------------------------------------------------------
// ProfileHMM -- a simplified Plan7 profile HMM (the HMMER/HHblits model).
//
//   A profile HMM describes a protein FAMILY column by column. For each of L
//   "match" columns it stores:
//     * emit[k][a]  : the integer log-odds of emitting residue `a` from match
//                     state k -- i.e. log( P(a | column k) / P_background(a) ),
//                     scaled by SCORE_SCALE. A positive value means "residue a is
//                     MORE common in this family column than by chance" (a good
//                     match); negative means "rarer than chance".
//   Plus the state-transition log-odds (also integer, scaled), shared across
//   columns here for didactic simplicity (a real HMM has per-column transitions):
//     * t_mm : match  k   -> match  k+1   (consume a query column AND a residue)
//     * t_mi : match  k   -> insert k     (begin an insertion: extra residues)
//     * t_im : insert k   -> match  k+1   (end an insertion)
//     * t_ii : insert k   -> insert k     (extend an insertion)
//     * t_md : match  k   -> delete k+1   (skip a query column: a deletion)
//     * t_dm : delete k   -> match  k+1   (end a deletion)
//     * t_dd : delete k   -> delete k+1   (extend a deletion)
//
//   We keep insert states NON-emitting in log-odds terms (their emission cancels
//   against the background, the usual HMMER convention) so an insertion costs only
//   its transition penalties -- this keeps the teaching DP to three state rows.
//
//   Layout: emit is stored ROW-MAJOR as a flat vector of length L*ALPHABET_SIZE,
//   so match column k (1-based), residue a lives at emit[(k-1)*ALPHABET_SIZE + a].
//   A flat array is what we upload to the GPU, so we use the same layout on the
//   CPU -- identical indexing on both sides keeps the DP bit-for-bit equal.
// ---------------------------------------------------------------------------
struct ProfileHMM {
    int L = 0;                    // number of match columns (the profile length)
    std::vector<int> emit;        // [L * ALPHABET_SIZE] scaled log-odds emissions
    // Shared scaled-integer transition log-odds (see field list above):
    int t_mm = 0, t_mi = 0, t_im = 0, t_ii = 0, t_md = 0, t_dm = 0, t_dd = 0;
};

// ---------------------------------------------------------------------------
// SeqDB -- the search database: N protein sequences packed into one flat buffer.
//   Sequences vary in length, so instead of a 2-D array we store:
//     * res    : every sequence's residues concatenated (each residue is an
//                amino-acid index 0..ALPHABET_SIZE-1, one byte).
//     * offset : [N+1] prefix offsets, so sequence i is res[offset[i] .. offset[i+1]).
//                offset[N] == res.size(). (The classic CSR / "ragged array" trick.)
//     * length : [N] convenience copy of each sequence length.
//   This is exactly how a GPU wants ragged data: one contiguous buffer + an
//   index, so a block can find "its" sequence by a single offset lookup.
// ---------------------------------------------------------------------------
struct SeqDB {
    int N = 0;                       // number of database sequences
    std::vector<uint8_t> res;        // concatenated residues (amino-acid indices)
    std::vector<int>     offset;     // [N+1] start offsets into res (CSR style)
    std::vector<int>     length;     // [N] convenience: length of each sequence
};

// imax: a tiny integer max helper. We avoid std::max so the function is usable in
// device code without pulling in <algorithm>. Marked HD so it too is shared.
HD inline int imax(int x, int y) { return x > y ? x : y; }

// ---------------------------------------------------------------------------
// viterbi_step : advance ONE database residue across ALL L match columns.
//
//   THIS IS THE ONE TRUE RECURRENCE that the CPU reference and the GPU kernel
//   both call, so their results are provably identical (max integer diff == 0).
//
//   The full Viterbi alignment of a length-T database sequence against the
//   length-L profile fills a (T+1) x (L+1) grid of three states (M,I,D). We sweep
//   it ROW BY ROW (one row per database residue), keeping only the previous row
//   and the current row -- O(L) memory instead of O(T*L), which is what lets a
//   single GPU block hold its working set in fast shared memory.
//
//   For database residue `a` (its amino-acid index), this function reads the
//   previous row (prev*) and writes the current row (cur*):
//     M[k] : best score of a path ENDING in match  state of column k at this row,
//            having just consumed residue a.        (consumes a residue)
//     I[k] : best score ending in insert state of column k.   (consumes a residue)
//     D[k] : best score ending in delete state of column k.   (consumes NOTHING,
//            so it propagates WITHIN the current row -> a single forward pass)
//
//   Parameters:
//     L         : profile length (number of match columns)
//     hmm_emit  : flat [L*ALPHABET_SIZE] emission log-odds table (scaled int)
//     a         : this residue's amino-acid index (0..ALPHABET_SIZE-1)
//     t_*       : the shared transition log-odds (scaled int)
//     prevM/I/D : the M/I/D score arrays after the PREVIOUS residue (length L+1)
//     curM/I/D  : output arrays for THIS residue (length L+1)
//
//   Column 0 is the "begin" state: starting at the profile's start is free
//   (score 0) at every row, which lets a high-scoring local alignment begin
//   anywhere in the database sequence -- a glocal/Smith-Waterman-flavoured search.
// ---------------------------------------------------------------------------
HD inline void viterbi_step(int L, const int* hmm_emit, int a,
                            int t_mm, int t_mi, int t_im, int t_ii,
                            int t_md, int t_dm, int t_dd,
                            const int* prevM, const int* prevI, const int* prevD,
                            int* curM, int* curI, int* curD) {
    // Begin state: free entry at the profile start, for every database row.
    curM[0] = 0;
    curI[0] = NEG_INF;   // no insert "before" column 1 in this teaching model
    curD[0] = NEG_INF;   // no delete at the begin state

    // Sweep match columns 1..L left to right. Because D[k] depends on D[k-1] of
    // the CURRENT row (deletes don't consume a residue), one forward pass resolves
    // the whole delete chain correctly.
    for (int k = 1; k <= L; ++k) {
        // Emission log-odds of residue a in match column k (row-major lookup).
        const int e = hmm_emit[(k - 1) * ALPHABET_SIZE + a];

        // M[k]: ARRIVE in match k by consuming residue a from the best of M/I/D
        // of column k-1 at the PREVIOUS row, then add the emission log-odds. If
        // every predecessor is unreachable, M[k] stays NEG_INF (don't add e to a
        // ~-1e9 sentinel, which could otherwise drift the floor).
        const int inM = imax(imax(prevM[k - 1] + t_mm, prevI[k - 1] + t_im),
                             prevD[k - 1] + t_dm);
        curM[k] = (inM <= NEG_INF / 2) ? NEG_INF : inM + e;

        // I[k]: emit an EXTRA residue while staying on column k. Enter from match
        // k of the PREVIOUS row (begin an insertion) or extend an existing insert
        // (self-loop). Inserts are non-emitting in log-odds, so no emission term.
        const int inI = imax(prevM[k] + t_mi, prevI[k] + t_ii);
        curI[k] = (inI <= NEG_INF / 2) ? NEG_INF : inI;

        // D[k]: SKIP column k without consuming a residue. Comes from match k-1 or
        // delete k-1 of the CURRENT row (deletes move within a row -> read cur*).
        const int inD = imax(curM[k - 1] + t_md, curD[k - 1] + t_dd);
        curD[k] = (inD <= NEG_INF / 2) ? NEG_INF : inD;
    }
}

// ---------------------------------------------------------------------------
// best_in_row : the best MATCH score anywhere in a finished row.
//   After consuming each database residue we ask "could the alignment END here?"
//   The hit score for a sequence is the maximum M[k] over all columns and all
//   rows (a local-style score: the best partial path that has matched the profile
//   up to some column). Both CPU and GPU call this so the reported score matches.
//   Returns the max of curM[1..L] (column 0 is the begin state, excluded).
// ---------------------------------------------------------------------------
HD inline int best_in_row(int L, const int* curM) {
    int best = NEG_INF;
    for (int k = 1; k <= L; ++k) best = imax(best, curM[k]);
    return best;
}

#undef HD   // keep the macro local to this header (avoid leaking into includers)
