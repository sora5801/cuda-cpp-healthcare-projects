// ===========================================================================
// src/reference_cpu.cpp  --  The plain-C++ coevolution baseline we trust
// ---------------------------------------------------------------------------
// Project 2.25 : Coevolutionary Contact Prediction & MSA Transformer
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- a single readable triple loop over column pairs,
//   no parallelism, no cleverness -- so that when the GPU and CPU agree, we
//   believe the GPU. It computes the SAME quantity the kernels do (the raw MI
//   matrix) using the SAME shared math (cv_mi_from_counts in coevolution.h), so
//   agreement is exact up to a 1-ulp log() difference (THEORY.md verification).
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, coevolution.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"
#include "coevolution.h"     // CV_Q, cv_token_of_aa, cv_mi_from_counts

#include <algorithm>         // std::fill
#include <fstream>           // std::ifstream
#include <stdexcept>         // std::runtime_error
#include <string>

// ---------------------------------------------------------------------------
// load_msa: parse a FASTA alignment into a dense token matrix.
//   FASTA structure (simplified): lines starting with '>' are headers and begin
//   a new sequence; all following non-header lines are that sequence's residues
//   (possibly wrapped across multiple lines, which we concatenate). For an
//   ALIGNMENT every sequence must end up the same length L; we enforce that.
// ---------------------------------------------------------------------------
Msa load_msa(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open MSA file: " + path);

    // First gather the raw sequences as strings of tokens, one per record. We do
    // not know L until we have read the first sequence, so collect then validate.
    std::vector<std::vector<uint8_t>> seqs;   // one inner vector per sequence
    std::string line;
    bool in_record = false;                   // have we seen a '>' header yet?
    while (std::getline(in, line)) {
        // Strip a trailing '\r' so Windows CRLF files parse the same as LF ones.
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;           // skip blank lines
        if (line[0] == '>') {                 // header -> start a fresh sequence
            seqs.emplace_back();
            in_record = true;
            continue;
        }
        if (!in_record) continue;             // residues before any header: ignore
        // Append this line's residues (tokenized) to the current sequence.
        std::vector<uint8_t>& cur = seqs.back();
        for (char c : line) cur.push_back(static_cast<uint8_t>(cv_token_of_aa(c)));
    }
    if (seqs.empty()) throw std::runtime_error("MSA file has no sequences: " + path);

    // Validate rectangularity: every sequence must have length == L (= length of
    // the first). A ragged MSA is a user error we refuse to guess about.
    const std::size_t L = seqs.front().size();
    if (L == 0) throw std::runtime_error("first MSA sequence is empty: " + path);
    for (std::size_t r = 0; r < seqs.size(); ++r) {
        if (seqs[r].size() != L)
            throw std::runtime_error("MSA is not rectangular: sequence " +
                                     std::to_string(r) + " has length " +
                                     std::to_string(seqs[r].size()) + " != " +
                                     std::to_string(L));
    }

    // Pack into the dense row-major token matrix.
    Msa msa;
    msa.N = static_cast<int>(seqs.size());
    msa.L = static_cast<int>(L);
    msa.token.resize(static_cast<std::size_t>(msa.N) * msa.L);
    for (int r = 0; r < msa.N; ++r)
        for (int c = 0; c < msa.L; ++c)
            msa.token[static_cast<std::size_t>(r) * msa.L + c] = seqs[r][c];
    return msa;
}

// ---------------------------------------------------------------------------
// column_counts: tally how often each token appears in EACH column.
//   Returns a flat [L*CV_Q] array of integer counts, single[c*CV_Q + a] =
//   #sequences whose column c holds token a. These are the MARGINALS used by
//   every MI computation, so we build them ONCE (O(N*L)) and reuse them.
// ---------------------------------------------------------------------------
static std::vector<uint32_t> column_counts(const Msa& msa) {
    std::vector<uint32_t> single(static_cast<std::size_t>(msa.L) * CV_Q, 0u);
    for (int r = 0; r < msa.N; ++r) {
        const uint8_t* row = &msa.token[static_cast<std::size_t>(r) * msa.L];
        for (int c = 0; c < msa.L; ++c)
            single[static_cast<std::size_t>(c) * CV_Q + row[c]] += 1u;
    }
    return single;
}

// ---------------------------------------------------------------------------
// coevolution_cpu: the full reference pipeline (raw MI matrix, then APC).
//   STAGE 1 -- for every column pair (i, j) with i < j: build the Q*Q JOINT
//   count table over the N sequences, then call the shared cv_mi_from_counts to
//   get MI(i, j). The matrix is symmetric, so we fill (i,j) and (j,i). Diagonal
//   is 0. Complexity: O(L^2 * (N + Q^2)) -- the N term is the joint counting,
//   the Q^2 term is the MI sum. This O(L^2) independent-pair loop is exactly
//   what the GPU parallelizes (one thread per pair).
//   STAGE 2 -- apc_correct() turns raw MI into the corrected contact score.
// ---------------------------------------------------------------------------
void coevolution_cpu(const Msa& msa,
                     std::vector<double>& mi,
                     std::vector<double>& score) {
    const int N = msa.N, L = msa.L;
    const std::vector<uint32_t> single = column_counts(msa);   // [L*CV_Q] marginals

    mi.assign(static_cast<std::size_t>(L) * L, 0.0);   // L*L, diagonal stays 0

    // Scratch joint-count table, reused per pair (zeroed each time). CV_Q*CV_Q.
    std::vector<uint32_t> pair(static_cast<std::size_t>(CV_Q) * CV_Q);

    for (int i = 0; i < L; ++i) {
        const uint32_t* ci = &single[static_cast<std::size_t>(i) * CV_Q];   // marginal of col i
        for (int j = i + 1; j < L; ++j) {
            const uint32_t* cj = &single[static_cast<std::size_t>(j) * CV_Q];  // marginal of col j

            // Build the joint counts for THIS pair: walk all N sequences once,
            // bumping pair[tok_i, tok_j]. Integer counting -> exact, order-free.
            std::fill(pair.begin(), pair.end(), 0u);
            for (int r = 0; r < N; ++r) {
                const uint8_t* row = &msa.token[static_cast<std::size_t>(r) * L];
                const int a = row[i];   // token of this sequence in column i
                const int b = row[j];   // token of this sequence in column j
                pair[static_cast<std::size_t>(a) * CV_Q + b] += 1u;
            }

            // MI from the exact counts (shared host/device math).
            const double m = cv_mi_from_counts(pair.data(), ci, cj, N);
            mi[static_cast<std::size_t>(i) * L + j] = m;   // symmetric: fill both
            mi[static_cast<std::size_t>(j) * L + i] = m;
        }
    }

    apc_correct(mi, L, score);   // STAGE 2: background subtraction -> contact score
}

// ---------------------------------------------------------------------------
// apc_correct: Average Product Correction (Dunn et al., Bioinformatics 2008).
//   score(i,j) = MI(i,j) - MIcol(i)*MIcol(j)/MImean,  where
//     MIcol(i) = mean of off-diagonal MI in row i,
//     MImean   = mean of all off-diagonal MI.
//   Intuition: columns with high entropy (lots of variation) accumulate spurious
//   MI with EVERYONE -- a per-column "background brightness". The product term
//   estimates how much of MI(i,j) is just that background (proportional to both
//   columns' average coupling), and subtracts it. What remains is the SPECIFIC
//   coevolution between i and j -- a far better contact predictor than raw MI.
//
//   Shared by the CPU reference and the GPU path (main.cu) so corrected scores
//   match exactly. Deterministic: fixed summation order, double precision.
// ---------------------------------------------------------------------------
void apc_correct(const std::vector<double>& mi, int L, std::vector<double>& score) {
    score.assign(static_cast<std::size_t>(L) * L, 0.0);
    if (L < 2) return;   // a 1-column MSA has no pairs to correct

    // Per-column mean MI over OFF-DIAGONAL entries (L-1 of them per row), and the
    // global off-diagonal mean. Summing in a fixed (i,j) order keeps it
    // deterministic and identical to the GPU path's host-side reduction.
    std::vector<double> mi_col(static_cast<std::size_t>(L), 0.0);   // MIcol(i)
    double mi_sum_all = 0.0;                                        // for MImean
    for (int i = 0; i < L; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < L; ++j) {
            if (j == i) continue;                       // skip the diagonal
            row_sum += mi[static_cast<std::size_t>(i) * L + j];
        }
        mi_col[i] = row_sum / static_cast<double>(L - 1);   // mean over L-1 entries
        mi_sum_all += row_sum;
    }
    // Total off-diagonal cells = L*(L-1); their mean is the APC denominator.
    const double mi_mean = mi_sum_all / (static_cast<double>(L) * (L - 1));

    // Subtract the product-of-means background from every off-diagonal cell.
    // Guard mi_mean == 0 (a degenerate, fully-conserved MSA): then all MI is 0
    // and the corrected score is trivially 0 too.
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < L; ++j) {
            if (j == i) continue;
            const double bg = (mi_mean > 0.0) ? (mi_col[i] * mi_col[j] / mi_mean) : 0.0;
            score[static_cast<std::size_t>(i) * L + j] =
                mi[static_cast<std::size_t>(i) * L + j] - bg;
        }
    }
}
