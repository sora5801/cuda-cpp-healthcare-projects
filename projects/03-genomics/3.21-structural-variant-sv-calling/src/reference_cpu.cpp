// ===========================================================================
// src/reference_cpu.cpp  --  Loader, shared post-processing, serial reference
// ---------------------------------------------------------------------------
// Project 3.21 : Structural Variant (SV) Calling
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. It is written to be
//   OBVIOUSLY correct -- one readable loop per read, no parallelism -- so that
//   when the GPU and CPU agree we believe the GPU. The per-read math is the SAME
//   sv.h code the kernel runs, so agreement is exact (integer arithmetic).
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: sv.h, reference_cpu.h. Compare against kernels.cu (GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <algorithm>   // std::sort, std::max
#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error
#include <string>

// ---------------------------------------------------------------------------
// sv_encode_base / sv_decode_base: 2-bit-ish base alphabet.
//   We store sequence as small integers so the kernel does branchless integer
//   compares (A==A) instead of char arithmetic, and so an ambiguous base (N) has
//   a single sentinel code (4) that never "matches" in sv_match_score.
// ---------------------------------------------------------------------------
signed char sv_encode_base(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default:            return 4;   // N or anything unexpected -> unknown
    }
}
char sv_decode_base(signed char code) {
    static const char* L = "ACGTN";
    return (code >= 0 && code <= 4) ? L[code] : 'N';
}

// ---------------------------------------------------------------------------
// load_dataset: parse the tiny text sample (format in data/README.md).
//   File layout (whitespace/newline separated):
//     line 1 : REF <ref_sequence>           (ACGT string, the reference window)
//     line 2 : TRUTH <true_bp> <true_len>    (planted ground truth, -1 if none)
//     line 3 : N <num_reads>
//     then N lines: <raw_guess> <del_len> <flank_sequence>
//       flank_sequence = SV_FLANK bases the read carries left of its breakpoint.
//   We keep the format human-readable so a learner can open the sample and SEE
//   the planted deletion.
// ---------------------------------------------------------------------------
SvDataset load_dataset(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open dataset file: " + path);

    SvDataset d;
    std::string tag;

    // --- reference sequence ---
    std::string ref_seq;
    if (!(in >> tag >> ref_seq) || tag != "REF")
        throw std::runtime_error("expected 'REF <sequence>' in " + path);
    d.ref_len = static_cast<int>(ref_seq.size());
    d.ref.resize(d.ref_len);
    for (int i = 0; i < d.ref_len; ++i) d.ref[i] = sv_encode_base(ref_seq[i]);

    // --- ground truth (synthetic only) ---
    if (!(in >> tag >> d.truth_bp >> d.truth_len) || tag != "TRUTH")
        throw std::runtime_error("expected 'TRUTH <bp> <len>' in " + path);

    // --- reads ---
    int nreads = 0;
    if (!(in >> tag >> nreads) || tag != "N" || nreads < 0)
        throw std::runtime_error("expected 'N <count>' in " + path);
    d.reads.resize(static_cast<std::size_t>(nreads));
    for (int r = 0; r < nreads; ++r) {
        SvRead& rd = d.reads[r];
        std::string flank;
        if (!(in >> rd.raw_guess >> rd.del_len >> flank))
            throw std::runtime_error("read record truncated in " + path);
        if (static_cast<int>(flank.size()) != SV_FLANK)
            throw std::runtime_error("read flank must be exactly SV_FLANK bases in " + path);
        for (int j = 0; j < SV_FLANK; ++j) rd.left[j] = sv_encode_base(flank[j]);
    }
    return d;
}

// ---------------------------------------------------------------------------
// histogram_to_calls: merge per-bin votes into SV calls (shared CPU+GPU host
// step). Identical input (the exact same integer histogram) -> identical calls,
// whether the histogram came from the CPU loop or the GPU atomics.
//
// ALGORITHM (greedy peak-merging, deterministic):
//   Scan bins left to right. Whenever a bin's support clears `min_support` and is
//   a LOCAL maximum within +/- SV_MERGE (so two adjacent fuzzy breakpoints from
//   the same SV collapse to one call), emit a call. Consensus breakpoint = the
//   peak bin; support = summed votes in the merge window; consensus length =
//   integer mean of the length-vote sum over that window.
// ---------------------------------------------------------------------------
std::vector<SvCall> histogram_to_calls(const std::vector<unsigned int>& hist,
                                       const std::vector<unsigned long long>& len_sum,
                                       int ref_len, unsigned int total_reads,
                                       unsigned int min_support) {
    std::vector<SvCall> calls;
    for (int b = 0; b < ref_len; ++b) {
        if (hist[b] < min_support) continue;   // below the noise floor: skip

        // Is bin b the strict peak of its merge window? Ties broken toward the
        // LOWEST coordinate (strict '>' against earlier bins, '>=' against later)
        // so exactly one bin in a flat plateau is chosen -> deterministic.
        bool is_peak = true;
        for (int o = -SV_MERGE; o <= SV_MERGE && is_peak; ++o) {
            int nb = b + o;
            if (nb < 0 || nb >= ref_len || nb == b) continue;
            if (nb < b) { if (hist[nb] >= hist[b]) is_peak = false; }
            else        { if (hist[nb] >  hist[b]) is_peak = false; }
        }
        if (!is_peak) continue;

        // Accumulate support + length votes across the merge window.
        unsigned int       support = 0;
        unsigned long long lsum    = 0;
        unsigned int       lcount  = 0;
        for (int o = -SV_MERGE; o <= SV_MERGE; ++o) {
            int nb = b + o;
            if (nb < 0 || nb >= ref_len) continue;
            support += hist[nb];
            lsum    += len_sum[nb];
            lcount  += hist[nb];
        }
        SvCall c;
        c.breakpoint = b;
        c.support    = support;
        c.del_len    = (lcount > 0) ? static_cast<int>(lsum / lcount) : 0;  // integer mean
        c.genotype   = sv_geno_from_vaf(support, total_reads);
        calls.push_back(c);
    }
    // Already in ascending breakpoint order by construction, but sort to be safe
    // and explicit (determinism is sacred for the diffed stdout).
    std::sort(calls.begin(), calls.end(),
              [](const SvCall& a, const SvCall& b) { return a.breakpoint < b.breakpoint; });
    return calls;
}

// ---------------------------------------------------------------------------
// sv_call_cpu: the trusted serial pipeline.
//   Stage 1 (per read, serial): refine the breakpoint by banded SW and vote into
//            the histogram + length-sum arrays. This is the loop the GPU
//            parallelizes (one thread per read) in kernels.cu.
//   Stage 2 (shared host): merge the histogram into calls.
//   Returns the calls; also hands back `hist` + `len_sum` so main.cu can compare
//   the GPU's histogram against this one bin-for-bin (exact integer equality).
// ---------------------------------------------------------------------------
std::vector<SvCall> sv_call_cpu(const SvDataset& d, unsigned int min_support,
                                std::vector<unsigned int>& hist,
                                std::vector<unsigned long long>& len_sum) {
    hist.assign(static_cast<std::size_t>(d.ref_len), 0u);
    len_sum.assign(static_cast<std::size_t>(d.ref_len), 0ull);

    const int N = d.N();
    for (int r = 0; r < N; ++r) {
        const SvRead& rd = d.reads[r];
        // Refine this read's breakpoint independently (banded SW over a small
        // search window). int score is discarded here but exposed for QC.
        int score = 0;
        int bp = sv_refine_breakpoint(rd.left, SV_FLANK, d.ref.data(), d.ref_len,
                                      rd.raw_guess, &score);
        int bin = sv_bin(bp);
        if (bin < 0 || bin >= d.ref_len) continue;   // refined off the reference: drop
        // Cast the vote: +1 support and += del_len into this bin (integer adds).
        hist[bin]    += 1u;
        len_sum[bin] += static_cast<unsigned long long>(rd.del_len);
    }
    return histogram_to_calls(hist, len_sum, d.ref_len,
                              static_cast<unsigned int>(N), min_support);
}
