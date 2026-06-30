// ===========================================================================
// src/reference_cpu.cpp  --  CPU PairHMM forward reference + data loading
// ---------------------------------------------------------------------------
// Project 3.3 : Variant Calling Acceleration
//
// This is the trusted, plain-C++ baseline. It fills the full (read x haplotype)
// DP table with the forward algorithm for every pair, using the EXACT same
// per-cell arithmetic (pairhmm_core.h::pairhmm_step) the GPU kernel uses -- so
// when main.cu compares the two log-likelihood matrices they agree to a few ULP.
//
// Compiled by the host C++ compiler (no CUDA here). See ../THEORY.md for the
// derivation of the recurrence and the log-space scaling we use to avoid
// underflow. Read this AFTER pairhmm_core.h and reference_cpu.h.
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>      // std::log10, std::pow
#include <fstream>    // std::ifstream
#include <limits>     // std::numeric_limits
#include <sstream>    // std::istringstream
#include <stdexcept>  // std::runtime_error
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// load_variant_data: parse the tiny text dataset (format in data/README.md):
//
//   # comment lines start with '#'
//   n_reads n_haps read_len hap_len truth delta epsilon
//   <n_haps lines>  : a haplotype sequence (hap_len bases)
//   <n_reads lines> : a read sequence followed by `read_len` integer qualities
//
//   We encode bases A/C/G/T/N -> 0..4 on load so the kernel handles bytes, and
//   finalize the pair-HMM transition probabilities once (host-side) so the GPU
//   receives identical double bit patterns.
// ---------------------------------------------------------------------------
VariantData load_variant_data(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open input file: " + path);

    VariantData v;
    std::string line;

    // Skip blank / comment lines, return the next data line (false at EOF).
    auto next_data_line = [&](std::string& out) -> bool {
        while (std::getline(in, out)) {
            std::size_t p = out.find_first_not_of(" \t\r\n");
            if (p == std::string::npos) continue;          // blank
            if (out[p] == '#') continue;                   // comment
            return true;
        }
        return false;
    };

    // --- Header line ---------------------------------------------------------
    if (!next_data_line(line)) throw std::runtime_error("empty/short data file: " + path);
    {
        std::istringstream hs(line);
        double delta = 0.0, epsilon = 0.0;
        if (!(hs >> v.n_reads >> v.n_haps >> v.read_len >> v.hap_len >> v.truth >> delta >> epsilon))
            throw std::runtime_error("bad header (need: n_reads n_haps read_len hap_len truth delta epsilon)");
        v.params.delta = delta;
        v.params.epsilon = epsilon;
        pairhmm_finalize_params(v.params);   // fill derived transition probs once
    }
    if (v.n_reads <= 0 || v.n_haps <= 0 || v.read_len <= 0 || v.hap_len <= 0)
        throw std::runtime_error("non-positive dimension in header");

    // --- Haplotype rows ------------------------------------------------------
    v.haps.resize(static_cast<std::size_t>(v.n_haps) * v.hap_len);
    for (int h = 0; h < v.n_haps; ++h) {
        if (!next_data_line(line)) throw std::runtime_error("not enough haplotype rows");
        std::istringstream hs(line);
        std::string seq;
        hs >> seq;
        if (static_cast<int>(seq.size()) != v.hap_len)
            throw std::runtime_error("haplotype length mismatch on row " + std::to_string(h));
        for (int j = 0; j < v.hap_len; ++j)
            v.haps[static_cast<std::size_t>(h) * v.hap_len + j] = encode_base(seq[j]);
    }

    // --- Read rows (sequence + per-base qualities) ---------------------------
    v.reads.resize(static_cast<std::size_t>(v.n_reads) * v.read_len);
    v.quals.resize(static_cast<std::size_t>(v.n_reads) * v.read_len);
    for (int r = 0; r < v.n_reads; ++r) {
        if (!next_data_line(line)) throw std::runtime_error("not enough read rows");
        std::istringstream rs(line);
        std::string seq;
        rs >> seq;
        if (static_cast<int>(seq.size()) != v.read_len)
            throw std::runtime_error("read length mismatch on row " + std::to_string(r));
        for (int i = 0; i < v.read_len; ++i)
            v.reads[static_cast<std::size_t>(r) * v.read_len + i] = encode_base(seq[i]);
        for (int i = 0; i < v.read_len; ++i) {
            int q = 0;
            if (!(rs >> q)) throw std::runtime_error("missing quality value on read row " + std::to_string(r));
            v.quals[static_cast<std::size_t>(r) * v.read_len + i] =
                static_cast<uint8_t>(q < 0 ? 0 : (q > 255 ? 255 : q));
        }
    }
    return v;
}

// ---------------------------------------------------------------------------
// forward_one_pair: log10 P(read r | haplotype h) via the forward algorithm.
//
//   The DP table has (read_len+1) rows and (hap_len+1) columns. Initialisation
//   follows the GATK convention:
//     * Every D cell of read row 0 (i=0) starts at 1 / hap_len. This is the
//       standard "the read may begin anywhere along the haplotype" prior: the
//       alignment can start at any of the hap_len haplotype positions with equal
//       probability, so the initial deletion mass is spread uniformly.
//     * M and I are 0 in row 0 (no read base emitted yet).
//   The likelihood is the sum over the LAST read row of (M + I) across all
//   haplotype columns -- every way the read could finish aligned to the hap.
//
//   We keep only TWO rows (previous + current): the recurrence at row i reads
//   only rows i-1 and i, so memory is O(hap_len) not O(read_len*hap_len). The
//   GPU kernel uses the same two-row trick, which is why one DP table fits in a
//   thread's modest storage. Returns log10 of the summed probability.
// ---------------------------------------------------------------------------
static double forward_one_pair(const VariantData& v, int r, int h) {
    const int R = v.read_len;
    const int H = v.hap_len;
    const int W = H + 1;                 // columns including the j=0 boundary
    const uint8_t* read = &v.reads[static_cast<std::size_t>(r) * R];
    const uint8_t* qual = &v.quals[static_cast<std::size_t>(r) * R];
    const uint8_t* hap  = &v.haps [static_cast<std::size_t>(h) * H];

    // Two rolling rows of cells. prev = row i-1, cur = row i.
    std::vector<PairHmmCell> prev(W), cur(W);

    // Row 0 (no read base emitted yet): M=I=0; D seeded with the uniform start
    // prior 1/H at every haplotype column so the read may begin anywhere.
    const double start = 1.0 / static_cast<double>(H);
    for (int j = 0; j < W; ++j) {
        prev[j].m = 0.0;
        prev[j].i = 0.0;
        prev[j].d = (j == 0) ? 0.0 : start;   // column 0 is the empty-haplotype boundary
    }

    // Fill rows i = 1..R. Column 0 (empty haplotype prefix) stays all-zero: a
    // read base cannot align to "nothing", so M/I/D there are 0 throughout.
    for (int i = 1; i <= R; ++i) {
        cur[0].m = cur[0].i = cur[0].d = 0.0;   // j=0 boundary column
        const uint8_t rb = read[i - 1];
        const int     q  = qual[i - 1];
        for (int j = 1; j <= H; ++j) {
            const uint8_t hb = hap[j - 1];
            // Neighbours: diag=(i-1,j-1) and up=(i-1,j) from the previous row,
            // left=(i,j-1) from the current row (already computed this pass).
            const PairHmmCell diag = prev[j - 1];
            const PairHmmCell up   = prev[j];
            const PairHmmCell left = cur[j - 1];
            cur[j] = pairhmm_step(v.params, rb, hb, q, diag, up, left);
        }
        prev.swap(cur);   // current row becomes previous for the next read base
    }

    // After the last swap, `prev` holds the final read row (i = R). The total
    // likelihood is the sum of M+I over all haplotype columns (any finish point).
    double sum = 0.0;
    for (int j = 1; j <= H; ++j) sum += prev[j].m + prev[j].i;

    // Guard log10(0): an impossible pair returns a large negative log-likelihood.
    if (sum <= 0.0) return -std::numeric_limits<double>::infinity();
    return std::log10(sum);
}

// pairhmm_cpu: fill the whole R x H log-likelihood matrix, one pair at a time.
void pairhmm_cpu(const VariantData& v, std::vector<double>& loglik) {
    loglik.assign(static_cast<std::size_t>(v.n_reads) * v.n_haps, 0.0);
    for (int r = 0; r < v.n_reads; ++r)
        for (int h = 0; h < v.n_haps; ++h)
            loglik[static_cast<std::size_t>(r) * v.n_haps + h] = forward_one_pair(v, r, h);
}

// best_haplotype_per_read: argmax over each read's row of the matrix.
//   Ties resolve to the lowest haplotype index so the output is deterministic.
void best_haplotype_per_read(const VariantData& v, const std::vector<double>& loglik,
                             std::vector<int>& best) {
    best.assign(v.n_reads, -1);
    for (int r = 0; r < v.n_reads; ++r) {
        double bestval = -std::numeric_limits<double>::infinity();
        int besth = 0;
        for (int h = 0; h < v.n_haps; ++h) {
            const double val = loglik[static_cast<std::size_t>(r) * v.n_haps + h];
            if (val > bestval) { bestval = val; besth = h; }
        }
        best[r] = besth;
    }
}
