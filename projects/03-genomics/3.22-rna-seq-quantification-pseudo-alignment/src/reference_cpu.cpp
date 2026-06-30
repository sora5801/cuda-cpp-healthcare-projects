// ===========================================================================
// src/reference_cpu.cpp  --  Loader, shared helpers, serial EM reference
// ---------------------------------------------------------------------------
// Project 3.22 : RNA-seq Quantification / Pseudo-alignment
// Compiled by the host compiler only. The per-ec E-step math lives in
// pseudoalign.h and is shared verbatim with the GPU kernel.
// ===========================================================================
#include "reference_cpu.h"

#include <algorithm>   // std::fill
#include <cmath>       // std::fabs
#include <fstream>     // std::ifstream
#include <sstream>     // std::istringstream
#include <stdexcept>   // std::runtime_error
#include <string>

// ---------------------------------------------------------------------------
// load_dataset: parse the equivalence-class text format (see data/README.md).
//
//   Lines beginning with '#' are comments and skipped. The token stream is:
//       T  M
//       eff_len[0] eff_len[1] ... eff_len[T-1]
//   then M equivalence-class lines, each:
//       count  k  m_0 m_1 ... m_{k-1}
//   and an OPTIONAL trailing truth block introduced by the literal token "TRUTH"
//   followed by T fractions (ground-truth rho), used only for reporting recovery.
//
//   We strip comments line-by-line, then read token-by-token with operator>> so
//   whitespace and line breaks are interchangeable.
// ---------------------------------------------------------------------------
EcDataset load_dataset(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open dataset file: " + path);

    // Concatenate all non-comment lines into one token stream. This lets the
    // sample be human-friendly (comments + free layout) without complicating
    // the parse below.
    std::ostringstream body;
    std::string line;
    while (std::getline(in, line)) {
        const std::size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);  // drop comment
        body << ' ' << line;
    }
    std::istringstream ts(body.str());

    EcDataset d;
    if (!(ts >> d.T >> d.M) || d.T <= 0 || d.M <= 0)
        throw std::runtime_error("bad header (expected 'T M') in " + path);

    // Effective lengths.
    d.eff_len.resize(d.T);
    for (int t = 0; t < d.T; ++t)
        if (!(ts >> d.eff_len[t]) || d.eff_len[t] <= 0.0)
            throw std::runtime_error("bad/positive eff_len expected in " + path);

    // Equivalence classes in CSR form.
    d.ec_count.resize(d.M);
    d.ec_offset.assign(d.M + 1, 0);
    d.total_reads = 0.0;
    for (int e = 0; e < d.M; ++e) {
        double count; int k;
        if (!(ts >> count >> k) || count < 0.0 || k <= 0 || k > PSA_MAX_EC_SIZE)
            throw std::runtime_error("bad ec line (count k ...), or ec too large, in " + path);
        d.ec_count[e]      = count;
        d.total_reads     += count;
        d.ec_offset[e + 1] = d.ec_offset[e] + k;          // running CSR offset
        for (int j = 0; j < k; ++j) {
            std::int32_t m;
            if (!(ts >> m) || m < 0 || m >= d.T)
                throw std::runtime_error("ec member id out of range in " + path);
            d.ec_members.push_back(m);
        }
    }

    // Optional ground-truth block. We peek for the literal "TRUTH"; if present we
    // read T fractions, otherwise we leave truth_rho empty.
    std::string tag;
    if (ts >> tag && tag == "TRUTH") {
        d.truth_rho.resize(d.T);
        double s = 0.0;
        for (int t = 0; t < d.T; ++t) {
            if (!(ts >> d.truth_rho[t]) || d.truth_rho[t] < 0.0)
                throw std::runtime_error("bad TRUTH block in " + path);
            s += d.truth_rho[t];
        }
        // Normalise the truth so it is a proper distribution (sums to 1).
        if (s > 0.0) for (int t = 0; t < d.T; ++t) d.truth_rho[t] /= s;
    }
    return d;
}

// ---------------------------------------------------------------------------
// init_rho_uniform: every transcript equally likely at the start of EM. Starting
// uniform (never all-zero) keeps the first E-step well defined and is what
// kallisto/RSEM do. Deterministic, so CPU and GPU begin from the same point.
// ---------------------------------------------------------------------------
void init_rho_uniform(const EcDataset& d, std::vector<double>& rho) {
    rho.assign(d.T, 1.0 / static_cast<double>(d.T));
}

// ---------------------------------------------------------------------------
// counts_to_rho: the renormalise that finishes each M-step. Convert the
// fixed-point per-transcript read sums back to floating counts, then divide by
// their total so rho is a probability distribution (sums to 1). Because the
// fixed-point sums are integers built by commuting adds, BOTH the CPU and the GPU
// feed identical inputs here -> identical rho out.
// ---------------------------------------------------------------------------
void counts_to_rho(const EcDataset& d, const std::vector<unsigned long long>& fixed_counts,
                   std::vector<double>& rho) {
    rho.assign(d.T, 0.0);
    double total = 0.0;
    for (int t = 0; t < d.T; ++t) {
        const double c = psa_from_fixed(fixed_counts[t]);   // back to read counts
        rho[t] = c;
        total += c;
    }
    if (total > 0.0) for (int t = 0; t < d.T; ++t) rho[t] /= total;
}

// ---------------------------------------------------------------------------
// tpm_from_rho: TPM = transcripts per million. rho is the read FRACTION; to get
// a molar abundance comparable across genes/samples we divide by effective
// length and rescale so the values sum to 1e6. This is exactly kallisto's TPM.
// ---------------------------------------------------------------------------
void tpm_from_rho(const EcDataset& d, const std::vector<double>& rho,
                  std::vector<double>& tpm) {
    tpm.assign(d.T, 0.0);
    double denom = 0.0;
    for (int t = 0; t < d.T; ++t) denom += rho[t] / d.eff_len[t];
    if (denom <= 0.0) return;
    for (int t = 0; t < d.T; ++t)
        tpm[t] = (rho[t] / d.eff_len[t]) / denom * 1.0e6;
}

// ---------------------------------------------------------------------------
// em_cpu: the trusted serial EM. One iteration is E-step (per ec, split reads)
// + M-step (accumulate to per-transcript fixed-point counts) + renormalise. We
// run a FIXED number of iterations so the result is deterministic and matches
// the GPU step for step (no data-dependent early stop, which could diverge).
// ---------------------------------------------------------------------------
double em_cpu(const EcDataset& d, int iters,
              std::vector<double>& rho, std::vector<double>& est_counts) {
    init_rho_uniform(d, rho);

    std::vector<unsigned long long> fixed_counts(d.T, 0ull);
    std::vector<double> prev_rho(d.T, 0.0);
    double contrib[PSA_MAX_EC_SIZE];               // per-ec E-step scratch
    double last_delta = 0.0;

    for (int it = 0; it < iters; ++it) {
        prev_rho = rho;

        // M-step accumulator starts empty each iteration.
        std::fill(fixed_counts.begin(), fixed_counts.end(), 0ull);

        // For every ec: E-step (split reads among members), then scatter the
        // resulting expected counts into the per-transcript fixed-point sums.
        for (int e = 0; e < d.M; ++e) {
            const std::int32_t base = d.ec_offset[e];
            const int k = d.ec_offset[e + 1] - base;
            psa_ec_contributions(d.ec_count[e], &d.ec_members[base], k,
                                  rho.data(), d.eff_len.data(), contrib);
            for (int j = 0; j < k; ++j) {
                const std::int32_t t = d.ec_members[base + j];
                fixed_counts[t] += psa_to_fixed(contrib[j]);   // commuting integer add
            }
        }

        // M-step finish: counts -> next rho.
        counts_to_rho(d, fixed_counts, rho);

        // Convergence witness: L1 change in rho this iteration (reported, not
        // used to stop, so CPU and GPU run the exact same number of iterations).
        last_delta = 0.0;
        for (int t = 0; t < d.T; ++t) last_delta += std::fabs(rho[t] - prev_rho[t]);
    }

    // Report the final expected read counts per transcript from the last
    // fixed-point sums (exact, integer-derived).
    est_counts.assign(d.T, 0.0);
    for (int t = 0; t < d.T; ++t) est_counts[t] = psa_from_fixed(fixed_counts[t]);
    return last_delta;
}
