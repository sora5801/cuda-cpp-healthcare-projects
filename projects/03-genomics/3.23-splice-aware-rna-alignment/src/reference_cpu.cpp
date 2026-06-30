// ===========================================================================
// src/reference_cpu.cpp  --  Trusted serial splice-aware aligner: loader,
//                            per-read DP, and N-aware CIGAR traceback
// ---------------------------------------------------------------------------
// Project 3.23 : Splice-Aware RNA Alignment   (REDUCED-SCOPE teaching version)
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU is checked against. It is written to be OBVIOUSLY
//   correct -- a single readable triple loop per read, no parallelism -- so that
//   when the GPU and CPU agree to the integer, we believe the GPU. The actual
//   per-cell math lives in cell_recurrence() in reference_cpu.h, SHARED with the
//   kernel, so this file is just the serial control flow around it.
//
//   (1) load_batch()       : parse the reference + reads sample (data/README.md).
//   (2) align_batch_cpu()  : fill each read's DP table and find its best cell.
//   (3) traceback_cigar()  : turn one filled table into a "12M48N9M" CIGAR.
//
//   Compiled by the host C++ compiler only (no __global__ here). See
//   reference_cpu.h. Compare against kernels.cu (the GPU twin of step 2).
// ===========================================================================
#include "reference_cpu.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// encode(c): map a nucleotide character to its 0..3 code (or throw). RNA reads
//   use 'U' where DNA uses 'T'; we fold U->T so the same integer alphabet works
//   for the genomic reference and the transcribed read.
// ---------------------------------------------------------------------------
static uint8_t encode(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        case 'U': case 'u': return 3;   // RNA uracil aligns like thymine here
        default:
            throw std::runtime_error(std::string("non-ACGTU base in sequence: '") + c + "'");
    }
}

// encode_seq: turn one text line into a vector of 0..3 codes, skipping stray
//   whitespace/CR so Windows and Unix line endings both load cleanly.
static std::vector<uint8_t> encode_seq(const std::string& s) {
    std::vector<uint8_t> v;
    v.reserve(s.size());
    for (char c : s) {
        if (c == '\r' || c == ' ' || c == '\t') continue;
        v.push_back(encode(c));
    }
    return v;
}

ReadBatch load_batch(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open batch file: " + path);

    // Helper: is this line blank or a '#' comment? (so the sample can be annotated)
    auto is_skippable = [](const std::string& s) {
        std::string t = s;
        while (!t.empty() && (t.back() == '\r' || t.back() == ' ' || t.back() == '\t'))
            t.pop_back();
        return t.empty() || t[0] == '#';
    };

    ReadBatch b;
    std::string line;

    // FIRST non-skippable line = the reference "gene model" (genomic order:
    // exons + introns). Skip any leading comment/blank header lines first.
    bool have_ref = false;
    while (std::getline(in, line)) {
        if (is_skippable(line)) continue;
        b.ref = encode_seq(line);
        b.n   = static_cast<int>(b.ref.size());
        have_ref = true;
        break;
    }
    if (!have_ref || b.n == 0)
        throw std::runtime_error("no reference line found in " + path);

    // Remaining non-skippable lines = reads (each a spliced fragment of the mRNA).
    std::vector<std::vector<uint8_t>> reads;
    while (std::getline(in, line)) {
        if (is_skippable(line)) continue;
        reads.push_back(encode_seq(line));
    }
    if (reads.empty()) throw std::runtime_error("no reads found in " + path);

    // Reads can differ in length; we store them in a padded [R*M] matrix (M =
    // the longest read) so the device sees a regular 2-D layout, and remember
    // each read's TRUE length so padding never participates in the alignment.
    b.num_reads = static_cast<int>(reads.size());
    b.read_len  = 0;
    for (const auto& rd : reads) b.read_len = std::max(b.read_len, (int)rd.size());
    b.reads.assign(static_cast<std::size_t>(b.num_reads) * b.read_len, 0);
    b.read_lens.assign(b.num_reads, 0);
    for (int r = 0; r < b.num_reads; ++r) {
        b.read_lens[r] = static_cast<int>(reads[r].size());
        for (int i = 0; i < b.read_lens[r]; ++i)
            b.reads[static_cast<std::size_t>(r) * b.read_len + i] = reads[r][i];
    }
    return b;
}

// ---------------------------------------------------------------------------
// align_one_cpu: fill ONE read's DP table H (size (M+1)*(N+1), row-major) and
//   return its best cell. Row 0 and column 0 are 0 (local-alignment init). We
//   sweep rows top-to-bottom and, within a row, columns left-to-right, so every
//   neighbour cell_recurrence() reads (diag/up/left and the earlier columns of
//   THIS row used by the intron move) is already final. Identical traversal
//   order to the kernel => identical integers.
//     b         : the batch (for the reference)
//     q         : pointer to this read's encoded bases (length m, then padding)
//     m         : this read's TRUE length
//     H         : output table, caller-sized to (m+1)*(N+1), filled here
//   Returns the AlignResult (best score + its 1-based endpoint cell).
// ---------------------------------------------------------------------------
static AlignResult align_one_cpu(const ReadBatch& b, const uint8_t* q, int m,
                                 std::vector<int>& H) {
    const int N = b.n;
    const int W = N + 1;                          // row stride (incl. column 0)
    H.assign(static_cast<std::size_t>(m + 1) * W, 0);   // zero row 0 and col 0
    const uint8_t* r = b.ref.data();

    AlignResult best;
    for (int i = 1; i <= m; ++i) {
        const uint8_t qi = q[i - 1];
        int* row      = &H[static_cast<std::size_t>(i) * W];        // H[i][*]
        const int* up = &H[static_cast<std::size_t>(i - 1) * W];    // H[i-1][*]
        for (int j = 1; j <= N; ++j) {
            const uint8_t rj = r[j - 1];
            // Hand the three classic neighbours + the PREVIOUS row (for the N
            // move's H[i-1][k] term) to the SHARED recurrence. All are final.
            const int v = cell_recurrence(qi, rj,
                                          up[j - 1],  // diag  H[i-1][j-1]
                                          up[j],      // up    H[i-1][j]
                                          row[j - 1], // left  H[i][j-1]
                                          up, r, N, j);
            row[j] = v;
            // Track the global best cell (deterministic: first in scan order).
            if (v > best.score) { best.score = v; best.end_i = i; best.end_j = j; }
        }
    }
    return best;
}

void align_batch_cpu(const ReadBatch& b,
                     std::vector<AlignResult>& out,
                     std::vector<int>& H_all) {
    const int M = b.read_len, N = b.n;
    const std::size_t table = static_cast<std::size_t>(M + 1) * (N + 1);
    out.assign(b.num_reads, AlignResult{});
    H_all.assign(static_cast<std::size_t>(b.num_reads) * table, 0);

    // Independent per read -- the serial mirror of the GPU's "one block per read".
    std::vector<int> H;                          // scratch reused across reads
    for (int rIdx = 0; rIdx < b.num_reads; ++rIdx) {
        const uint8_t* q = &b.reads[static_cast<std::size_t>(rIdx) * M];
        const int m = b.read_lens[rIdx];
        out[rIdx] = align_one_cpu(b, q, m, H);
        // Copy this read's full table into the padded H_all slot (rows 0..m;
        // any rows m+1..M stay zero, which is fine -- they are never the best).
        std::copy(H.begin(), H.end(),
                  H_all.begin() + static_cast<std::ptrdiff_t>(rIdx) * table);
    }
}

// ---------------------------------------------------------------------------
// traceback_cigar: walk back from the best cell of one read, reproducing the
//   choice cell_recurrence() made at each step, to emit a CIGAR string with N
//   (intron) operations. We re-derive each move by checking which predecessor
//   reproduces H[i][j] exactly (integers => no float ambiguity). Preference
//   order M > N > I > D makes the path deterministic (matches the recurrence's
//   "first wins" scans). Host-only; traceback is serial and not the GPU point.
//
//   Returns the CIGAR (e.g. "12M48N9M") and, via out-params, the number of
//   introns crossed and the count of M (aligned) columns -- both reported by
//   main.cu so the demo's stdout is an interpretable, deterministic result.
// ---------------------------------------------------------------------------
std::string traceback_cigar(const ReadBatch& b, int read_index,
                            const std::vector<int>& H_all,
                            const AlignResult& res,
                            int& out_introns, int& out_matched) {
    const int M = b.read_len, N = b.n, W = N + 1;
    const std::size_t table = static_cast<std::size_t>(M + 1) * W;
    const int* H = &H_all[static_cast<std::size_t>(read_index) * table];
    const uint8_t* q = &b.reads[static_cast<std::size_t>(read_index) * M];
    const uint8_t* r = b.ref.data();

    out_introns = 0;
    out_matched = 0;

    // We build a list of (op, length) operations from the END backwards, then
    // reverse. ops are characters: 'M' (match/mismatch), 'I' (read insertion),
    // 'D' (one-base ref deletion), 'N' (intron skip).
    std::vector<std::pair<char,int>> ops;        // (operation, run length)
    auto push = [&](char op, int len) {
        if (!ops.empty() && ops.back().first == op) ops.back().second += len;
        else ops.push_back({op, len});
    };

    int i = res.end_i, j = res.end_j;
    while (i > 0 && j > 0 && H[i * W + j] > 0) {
        const int here = H[i * W + j];
        const uint8_t qi = q[i - 1], rj = r[j - 1];
        const int s = sub_score(qi, rj);

        // (M) plain diagonal? H[i][j] == H[i-1][j-1] + s  (no intron here).
        if (here == H[(i - 1) * W + (j - 1)] + s) {
            push('M', 1); ++out_matched; --i; --j; continue;
        }
        // (N) intron-spliced match? q_i still MATCHES r_j (an 'M' column), but
        // an intron r[k+1..j-1] precedes column j, connecting to H[i-1][k]. Scan
        // the same window the recurrence used; take the first donor k that
        // reproduces the cell exactly (deterministic -- matches the fill order).
        bool took_intron = false;
        const int k_hi = j - 1 - MIN_INTRON;
        int k_lo = j - 1 - MAX_INTRON; if (k_lo < 0) k_lo = 0;
        for (int k = k_lo; k <= k_hi; ++k) {
            const int is = intron_score(r, N, k, j);
            if (is <= -1000000) continue;
            if (here == H[(i - 1) * W + k] + s + is) {
                push('M', 1); ++out_matched;          // the match AT column j
                const int intron_len = (j - 1) - (k + 1) + 1;  // bases skipped
                push('N', intron_len);
                ++out_introns;
                --i; j = k;                            // continue at (i-1, k)
                took_intron = true;
                break;
            }
        }
        if (took_intron) continue;
        // (I) up? gap in reference == an inserted read base.
        if (here == H[(i - 1) * W + j] + GAP) { push('I', 1); --i; continue; }
        // (D) left? gap in read == a one-base reference deletion.
        if (here == H[i * W + (j - 1)] + GAP) { push('D', 1); --j; continue; }
        break;                                       // reached the local start
    }

    std::reverse(ops.begin(), ops.end());
    std::ostringstream cig;
    for (const auto& p : ops) cig << p.second << p.first;
    return cig.str();
}
