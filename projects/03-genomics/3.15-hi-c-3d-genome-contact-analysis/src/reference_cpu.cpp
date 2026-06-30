// ===========================================================================
// src/reference_cpu.cpp  --  The plain-C++ Hi-C baseline we trust
// ---------------------------------------------------------------------------
// Project 3.15 : Hi-C / 3D Genome Contact Analysis
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. Written to be
//   OBVIOUSLY correct -- single readable loops, no parallelism -- so that when
//   the GPU and CPU agree (main.cu), we believe the GPU. Compiled by the host
//   C++ compiler only (no CUDA here). The per-element math is shared with the
//   GPU via hic.h, so the two paths cannot silently diverge.
//
//   Pipeline implemented here, mirrored on-device in kernels.cu:
//     load_matrix          -> read the sparse COO sample
//     compute_rowsums_cpu  -> per-bin corrected row sums (the ICE hot loop)
//     ice_update_bias      -> fold row sums into the bias, renormalise
//     ice_balance_cpu      -> iterate the two above `iters` times
//     insulation_score     -> diamond-window mean along the diagonal
//     call_boundaries      -> local minima of the insulation score => TADs
//
// READ THIS AFTER: reference_cpu.h, hic.h. Compare against kernels.cu (GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>       // std::fabs, std::sqrt
#include <cstdio>
#include <fstream>     // std::ifstream
#include <limits>      // std::numeric_limits
#include <stdexcept>   // std::runtime_error

// Sentinel marking an insulation score that could not be computed (a bin too
// close to the matrix edge, where the diamond window underflows). Negative is
// safe because a real insulation score (a mean of non-negative contacts) is >= 0.
static const double INSULATION_NA = -1.0;

// ---------------------------------------------------------------------------
// load_matrix: parse the tiny text sample.
//   Format (data/README.md): first line "n nnz"; then nnz lines "i j count".
//   We validate indices and i<=j so a malformed file fails loudly instead of
//   corrupting the reduction. Complexity O(nnz).
// ---------------------------------------------------------------------------
HicMatrix load_matrix(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open Hi-C matrix file: " + path);

    HicMatrix m;
    long long nnz = 0;
    if (!(in >> m.n >> nnz) || m.n <= 0 || nnz < 0)
        throw std::runtime_error("bad header (expected 'n nnz') in: " + path);

    m.entries.reserve(static_cast<std::size_t>(nnz));
    for (long long k = 0; k < nnz; ++k) {
        CooEntry e;
        if (!(in >> e.i >> e.j >> e.count))
            throw std::runtime_error("truncated entry list in: " + path);
        if (e.i < 0 || e.j < 0 || e.i >= m.n || e.j >= m.n || e.i > e.j)
            throw std::runtime_error("entry out of range or not upper-triangular");
        m.entries.push_back(e);
    }
    return m;
}

// ---------------------------------------------------------------------------
// compute_rowsums_cpu: one ICE iteration's reduction -- the per-bin row sums of
// the matrix corrected by the current bias.
//
//   We quantize each corrected contribution to fixed-point INTEGER quanta with
//   hic_to_fixed() and accumulate those, exactly as the GPU does with atomicAdd.
//   That is the whole trick that makes CPU==GPU bit-identical: both sides sum
//   the SAME integers, and integer addition is associative/commutative. We
//   convert back to double at the end.
//
//   Symmetry handling: an off-diagonal entry (i<j) lives at BOTH (i,j) and (j,i)
//   in the full matrix, so it contributes to row i AND row j. A diagonal entry
//   (i==j) contributes to row i once. kernels.cu follows the identical rule.
//   Complexity: O(nnz).
// ---------------------------------------------------------------------------
void compute_rowsums_cpu(const HicMatrix& m, const std::vector<double>& bias,
                         std::vector<double>& rowsum) {
    // Integer accumulators, one per bin -- the deterministic tally.
    std::vector<unsigned long long> acc(static_cast<std::size_t>(m.n), 0ull);

    for (const CooEntry& e : m.entries) {
        // Balanced contact M'_{ij} = count / (b_i b_j); 0 if either bin masked.
        const double corrected =
            hic_corrected(e.count, bias[e.i], bias[e.j]);
        const unsigned long long q = hic_to_fixed(corrected);

        acc[e.i] += q;                 // contributes to row i
        if (e.i != e.j) acc[e.j] += q; // off-diagonal also contributes to row j
    }

    rowsum.assign(static_cast<std::size_t>(m.n), 0.0);
    for (int k = 0; k < m.n; ++k) rowsum[k] = hic_from_fixed(acc[k]);
}

// ---------------------------------------------------------------------------
// ice_update_bias: fold the fresh row sums into the bias and renormalise.
//   Target: every occupied bin's row sum equals the mean over occupied bins.
//   Update: b_k <- b_k * (rowsum_k / mean). Masked bins (rowsum 0) stay masked.
//   We rescale the mean back into the biases by NOT touching empty bins, which
//   keeps the bias well-conditioned. Returns the row-sum variance (convergence).
//   Complexity: O(n).
// ---------------------------------------------------------------------------
double ice_update_bias(const std::vector<double>& rowsum,
                       std::vector<double>& bias) {
    const int n = static_cast<int>(rowsum.size());

    // Mean of the row sums over OCCUPIED bins only (rowsum > 0). Empty bins must
    // not drag the target down or they would never recover.
    double sum = 0.0;
    int occupied = 0;
    for (int k = 0; k < n; ++k) {
        if (rowsum[k] > 0.0) { sum += rowsum[k]; ++occupied; }
    }
    const double mean = (occupied > 0) ? (sum / occupied) : 0.0;
    if (mean <= 0.0) return 0.0;  // nothing to balance

    // Apply the multiplicative correction and measure how far we still are from
    // a perfectly balanced matrix (variance of row sums about the mean).
    double var = 0.0;
    for (int k = 0; k < n; ++k) {
        if (rowsum[k] > 0.0) {
            const double dev = rowsum[k] - mean;
            var += dev * dev;
            bias[k] *= rowsum[k] / mean;   // the ICE multiplicative update
        }
        // empty bins: bias[k] stays 0 (masked)
    }
    return var / occupied;
}

// ---------------------------------------------------------------------------
// ice_balance_cpu: the serial ICE driver. Start every occupied bin at bias 1,
// every empty bin at bias 0 (masked), then iterate {row sums; update bias}.
//   Returns the final convergence variance. Complexity: O(iters * nnz).
// ---------------------------------------------------------------------------
double ice_balance_cpu(const HicMatrix& m, int iters, std::vector<double>& bias) {
    // Find which bins are occupied (appear in at least one entry). Masked bins
    // get bias 0 so hic_corrected() returns 0 for any pair touching them.
    std::vector<char> occupied(static_cast<std::size_t>(m.n), 0);
    for (const CooEntry& e : m.entries) { occupied[e.i] = 1; occupied[e.j] = 1; }

    bias.assign(static_cast<std::size_t>(m.n), 0.0);
    for (int k = 0; k < m.n; ++k) bias[k] = occupied[k] ? 1.0 : 0.0;

    std::vector<double> rowsum;
    double var = 0.0;
    for (int it = 0; it < iters; ++it) {
        compute_rowsums_cpu(m, bias, rowsum);   // reduction (the GPU's hot loop)
        var = ice_update_bias(rowsum, bias);    // O(n) bias update
    }
    return var;
}

// ---------------------------------------------------------------------------
// insulation_score: the diamond-window mean along the diagonal.
//
//   Build a fast lookup of balanced contacts? No -- the matrix is sparse and we
//   keep it sparse. For each bin k we scan the entries once is too slow per bin,
//   so instead we accumulate every entry's balanced value into the windows it
//   falls inside. An entry (a,b) with a<k<=b and within `window` of k crosses
//   position k. Concretely, entry (a,b) contributes to bin k for every k with
//   a < k <= b, a >= k-window, b <= k+window-1  =>  k in (a, b] intersect
//   [b-window+1, a+window]. We add its balanced value to those bins' sums and
//   bump their counts. The score is sum/count. Complexity O(nnz * window).
//
//   Bins whose full diamond would run off either matrix edge (k < window or
//   k > n-window) are marked INSULATION_NA and skipped by the boundary caller.
// ---------------------------------------------------------------------------
std::vector<double> insulation_score(const HicMatrix& m,
                                     const std::vector<double>& bias,
                                     int window) {
    const int n = m.n;
    std::vector<double> sum(static_cast<std::size_t>(n), 0.0);   // running sum per bin
    std::vector<int>    cnt(static_cast<std::size_t>(n), 0);     // running count per bin

    for (const CooEntry& e : m.entries) {
        const int a = e.i, b = e.j;
        if (a == b) continue;                 // diagonal never crosses a boundary
        const double v = hic_corrected(e.count, bias[a], bias[b]);
        if (v <= 0.0) continue;               // masked pair contributes nothing

        // k must satisfy a < k <= b (the pair straddles k) AND both endpoints lie
        // inside the size-`window` diamond centred on k:
        //   a >= k-window  => k <= a+window ;  b <= k+window-1 => k >= b-window+1
        const int klo = (a + 1   > b - window + 1) ? (a + 1)   : (b - window + 1);
        const int khi = (b       < a + window)     ? (b)       : (a + window);
        for (int k = klo; k <= khi; ++k) {
            if (k < 0 || k >= n) continue;
            sum[k] += v;
            cnt[k] += 1;
        }
    }

    std::vector<double> score(static_cast<std::size_t>(n), INSULATION_NA);
    for (int k = 0; k < n; ++k) {
        // Edge bins cannot host a full diamond -> leave them NA.
        if (k < window || k > n - window) continue;
        score[k] = (cnt[k] > 0) ? (sum[k] / cnt[k]) : 0.0;
    }
    return score;
}

// ---------------------------------------------------------------------------
// call_boundaries: TAD boundaries = strict local minima of the insulation score.
//   Bin k is a boundary if score[k] is valid and strictly below every neighbour
//   within `radius` on both sides (all of which must also be valid). Strict "<"
//   makes the result deterministic (no ties spawn two adjacent boundaries).
//   Complexity O(n * radius). Returns ascending boundary indices.
// ---------------------------------------------------------------------------
std::vector<int> call_boundaries(const std::vector<double>& score, int radius) {
    const int n = static_cast<int>(score.size());
    std::vector<int> boundaries;
    for (int k = radius; k < n - radius; ++k) {
        if (score[k] == INSULATION_NA) continue;
        bool is_min = true;
        for (int d = 1; d <= radius && is_min; ++d) {
            const double l = score[k - d], r = score[k + d];
            // A neighbour that is NA or not strictly larger disqualifies k.
            if (l == INSULATION_NA || r == INSULATION_NA) { is_min = false; break; }
            if (!(score[k] < l) || !(score[k] < r)) is_min = false;
        }
        if (is_min) boundaries.push_back(k);
    }
    return boundaries;
}
