// ===========================================================================
// src/reference_cpu.cpp  --  Loader + trusted serial MEME EM baseline
// ---------------------------------------------------------------------------
// Project 3.29 : Motif Finding in Genomic Sequences
//
// ROLE IN THE PROJECT
//   (1) load_sequences(): parse the FASTA-like sample (data/README.md format),
//       encode bases to 0..4, and precompute the flat WINDOW index that turns
//       "N ragged sequences" into one 1-D list of scoring jobs.
//   (2) build_logodds(): turn a PWM + background into the log-odds table the
//       shared window_score() reads.
//   (3) score_windows_cpu(): the E-step (score every window) -- the trusted twin
//       of the GPU kernel, using the SAME motif_core.h formula so results match
//       bit-for-bit.
//   (4) run_meme_em_cpu(): the full OOPS Expectation-Maximisation loop that
//       drives the demo and recovers the planted motif.
//
//   Compiled by the host C++ compiler only (no CUDA). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, motif_core.h. Compare score_windows_cpu()
// against score_windows_gpu() in kernels.cu (the GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <algorithm>   // std::max_element, std::max
#include <cmath>       // std::log2, std::exp, std::log
#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error
#include <string>

// ---------------------------------------------------------------------------
// encode_base / decode_base : the 2-bit DNA alphabet (+ an "unknown" sentinel).
//   We map A/C/G/T to 0/1/2/3 so a base is a direct PWM row index, and anything
//   else (N, gaps, lowercase masking artefacts) to MOTIF_BASE_N=4. A window that
//   contains a 4 cannot be scored under the PWM and is excluded by the loader's
//   window index (we only enumerate windows that are all-ACGT... see below).
// ---------------------------------------------------------------------------
unsigned char encode_base(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default:            return MOTIF_BASE_N;   // N / gap / unknown
    }
}

char decode_base(int b) {
    static const char L[4] = {'A', 'C', 'G', 'T'};
    return (b >= 0 && b < 4) ? L[b] : 'N';
}

// ---------------------------------------------------------------------------
// load_sequences : read the FASTA-like text file and build the SequenceSet.
//
//   File format (data/README.md): standard FASTA --
//       >header_for_seq_0
//       ACGT...ACGT          (may wrap across multiple lines)
//       >header_for_seq_1
//       ...
//
//   We concatenate every sequence's encoded bases into one flat `data` buffer
//   and record CSR offsets, then enumerate every valid length-w window into a
//   flat list (seq_of_win / start_of_win). "Valid" = entirely within one
//   sequence AND containing no unknown (4) base, so window_score() never reads a
//   non-ACGT index. This precomputation is what makes the GPU launch trivial:
//   total_windows() independent jobs, each described by two ints.
// ---------------------------------------------------------------------------
SequenceSet load_sequences(const std::string& path, int w) {
    if (w <= 0) throw std::runtime_error("motif width w must be positive");

    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open sequence file: " + path);

    SequenceSet set;
    set.w = w;
    set.offset.push_back(0);   // CSR: sequence 0 starts at data index 0

    std::string line;
    bool in_record = false;    // have we seen at least one '>' header yet?
    int cur_len = 0;           // bases accumulated for the current sequence

    // Helper: close out the sequence currently being read (push its length).
    auto finish_seq = [&]() {
        if (in_record) {
            set.length.push_back(cur_len);
            set.offset.push_back(static_cast<int>(set.data.size()));
            cur_len = 0;
        }
    };

    while (std::getline(in, line)) {
        // Trim a trailing '\r' so Windows-CRLF files parse the same as LF.
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (line[0] == '>') {            // a new FASTA header
            finish_seq();                // close the previous sequence (if any)
            in_record = true;
            continue;
        }
        // A sequence line: append its encoded bases to the flat buffer.
        for (char c : line) {
            if (c == ' ' || c == '\t') continue;  // tolerate stray whitespace
            set.data.push_back(encode_base(c));
            ++cur_len;
        }
    }
    finish_seq();                        // close the final sequence at EOF

    set.n = static_cast<int>(set.length.size());
    if (set.n == 0) throw std::runtime_error("no sequences found in " + path);

    // ---- Build the flat window index ------------------------------------
    // For each sequence s, a window can start at j = 0 .. length[s]-w. We only
    // KEEP windows whose w bases are all ACGT (no 4), so the scoring loop is
    // branch-free. win_off is CSR over sequences so the GPU can also report
    // per-sequence argmax sites by scanning win_off[s]..win_off[s+1].
    set.win_off.assign(set.n + 1, 0);
    for (int s = 0; s < set.n; ++s) {
        const int base = set.offset[s];
        const int len  = set.length[s];
        if (len < w)
            throw std::runtime_error("sequence " + std::to_string(s) +
                " is shorter (" + std::to_string(len) + ") than motif width " +
                std::to_string(w));
        for (int j = 0; j + w <= len; ++j) {
            bool ok = true;
            for (int p = 0; p < w; ++p)
                if (set.data[base + j + p] >= MOTIF_BASE_N) { ok = false; break; }
            if (ok) {
                set.seq_of_win.push_back(s);
                set.start_of_win.push_back(base + j);  // ABSOLUTE index into data
            }
        }
        set.win_off[s + 1] = static_cast<int>(set.seq_of_win.size());
    }
    if (set.total_windows() == 0)
        throw std::runtime_error("no valid (all-ACGT) windows of width " +
                                 std::to_string(w) + " in " + path);
    return set;
}

// ---------------------------------------------------------------------------
// build_logodds : logodds[p*4+b] = log2( pwm[p*4+b] / bg[b] ).
//   The PWM rows are kept strictly positive by the M-step pseudocount, so the
//   log is always finite. This table is what makes window_score() a plain sum:
//   we pay the log() here ONCE per (column,base) per iteration, not once per
//   window. (THEORY sec "The algorithm": precompute the log-odds.)
// ---------------------------------------------------------------------------
void build_logodds(MotifModel& model) {
    const int w = model.w;
    model.logodds.assign(static_cast<std::size_t>(w) * MOTIF_ALPHABET, 0.0f);
    for (int p = 0; p < w; ++p)
        for (int b = 0; b < MOTIF_ALPHABET; ++b) {
            const float pr = model.pwm[p * MOTIF_ALPHABET + b];
            const float bg = model.bg[b];
            model.logodds[p * MOTIF_ALPHABET + b] =
                static_cast<float>(std::log2(static_cast<double>(pr) /
                                             static_cast<double>(bg)));
        }
}

// ---------------------------------------------------------------------------
// score_windows_cpu : the E-step's expensive part, done serially.
//   For each window in the flat list, call the shared window_score() -- the
//   exact function the GPU kernel calls. scores[win] is the log-odds of that
//   window under the current model. O(total_windows * w).
// ---------------------------------------------------------------------------
void score_windows_cpu(const SequenceSet& set, const MotifModel& model,
                       std::vector<float>& scores) {
    const int nw = set.total_windows();
    scores.assign(static_cast<std::size_t>(nw), 0.0f);
    for (int win = 0; win < nw; ++win) {
        const int start = set.start_of_win[win];   // absolute index into data
        // window_score reads set.data[start .. start+w-1]; identical to the GPU.
        scores[win] = window_score(set.data.data(), start, set.w,
                                    model.logodds.data());
    }
}

// ---------------------------------------------------------------------------
// run_meme_em_cpu : the OOPS Expectation-Maximisation loop.
//
//   STATE: a PWM (model.pwm). Each iteration:
//     E-STEP: score every window (score_windows_cpu), then for each sequence
//        turn its windows' log-odds into RESPONSIBILITIES via a numerically
//        stable softmax (subtract the per-sequence max before exp). Under OOPS,
//        the responsibilities of a sequence's windows sum to 1 -- they are the
//        posterior probability that the motif sits at each offset.
//     M-STEP: accumulate responsibility-weighted base counts at each motif
//        column across ALL windows, add a pseudocount, and renormalise to get
//        the new PWM. Background bg stays fixed (estimated once from the data).
//   CONVERGENCE: track the total data log-likelihood (the sum over sequences of
//     log sum_j exp(score_j), i.e. the log of each sequence's unnormalised
//     evidence); stop when it changes by less than `tol`, or at max_iters.
//
//   Everything here is DETERMINISTIC: fixed loop orders, no atomics, no
//   parallel float reductions -- so the recovered motif is byte-stable and the
//   demo's stdout never changes (PATTERNS.md sec 3).
// ---------------------------------------------------------------------------
EMResult run_meme_em_cpu(const SequenceSet& set, MotifModel& model,
                         int max_iters, double tol) {
    const int w  = set.w;
    const int nw = set.total_windows();

    std::vector<float> scores;            // [nw] reused each E-step
    std::vector<double> resp(static_cast<std::size_t>(nw), 0.0); // responsibilities
    const double PSEUDO = 0.25;           // Laplace-style pseudocount per base

    double prev_ll = -1e300;              // previous total log-likelihood
    EMResult res;
    res.iters = 0;

    for (int it = 0; it < max_iters; ++it) {
        // ---- E-step: log-odds of every window under the current model ----
        build_logodds(model);
        score_windows_cpu(set, model, scores);

        // Per-sequence softmax over its windows -> responsibilities, and
        // accumulate the total data log-likelihood for the convergence test.
        double total_ll = 0.0;
        for (int s = 0; s < set.n; ++s) {
            const int lo = set.win_off[s], hi = set.win_off[s + 1];
            // (a) max log-odds in this sequence (for numerical stability).
            double m = -1e300;
            for (int win = lo; win < hi; ++win)
                m = std::max(m, static_cast<double>(scores[win]));
            // (b) sum of exp(score - m); log-sum-exp gives this sequence's
            //     evidence and normalises the responsibilities.
            double sum = 0.0;
            for (int win = lo; win < hi; ++win)
                sum += std::exp(static_cast<double>(scores[win]) - m);
            const double inv = (sum > 0.0) ? 1.0 / sum : 0.0;
            for (int win = lo; win < hi; ++win)
                resp[win] = std::exp(static_cast<double>(scores[win]) - m) * inv;
            // log evidence of sequence s = m + log(sum)  (log-sum-exp).
            total_ll += m + std::log(sum);
        }

        // ---- M-step: responsibility-weighted base counts -> new PWM -------
        // counts[p*4+b] = sum over windows of resp(window) * [base at col p == b]
        std::vector<double> counts(static_cast<std::size_t>(w) * MOTIF_ALPHABET,
                                   PSEUDO);  // seed with the pseudocount
        for (int win = 0; win < nw; ++win) {
            const double r = resp[win];
            if (r <= 0.0) continue;
            const int start = set.start_of_win[win];
            for (int p = 0; p < w; ++p) {
                const int b = set.data[start + p];   // 0..3 (window is all-ACGT)
                counts[p * MOTIF_ALPHABET + b] += r;
            }
        }
        // Renormalise each column to a probability distribution.
        for (int p = 0; p < w; ++p) {
            double z = 0.0;
            for (int b = 0; b < MOTIF_ALPHABET; ++b) z += counts[p * MOTIF_ALPHABET + b];
            for (int b = 0; b < MOTIF_ALPHABET; ++b)
                model.pwm[p * MOTIF_ALPHABET + b] =
                    static_cast<float>(counts[p * MOTIF_ALPHABET + b] / z);
        }

        res.iters = it + 1;
        // Converged once the log-likelihood barely moves.
        if (it > 0 && std::abs(total_ll - prev_ll) < tol) { prev_ll = total_ll; break; }
        prev_ll = total_ll;
    }

    // ---- Final model: rebuild log-odds + score windows one last time -----
    // These final scores are what we VERIFY GPU-vs-CPU on (the parallel step's
    // exact output), and what we derive the consensus + sites from.
    build_logodds(model);
    score_windows_cpu(set, model, res.final_scores);

    // Consensus = argmax base per column (ties -> lowest base index, so it is
    // deterministic). Also accumulate information content in bits.
    res.consensus.clear();
    res.info_content = 0.0;
    for (int p = 0; p < w; ++p) {
        int best_b = 0;
        float best_p = model.pwm[p * MOTIF_ALPHABET + 0];
        double col_ic = 2.0;   // max info per DNA column is log2(4) = 2 bits
        for (int b = 0; b < MOTIF_ALPHABET; ++b) {
            const float pr = model.pwm[p * MOTIF_ALPHABET + b];
            if (pr > best_p) { best_p = pr; best_b = b; }
            if (pr > 0.0f) col_ic += static_cast<double>(pr) * std::log2(pr);
        }
        res.consensus.push_back(decode_base(best_b));
        res.info_content += col_ic;
    }

    // Per-sequence predicted site = argmax window start under the final model
    // (ties -> earliest start, deterministic).
    res.best_site.assign(set.n, 0);
    for (int s = 0; s < set.n; ++s) {
        const int lo = set.win_off[s], hi = set.win_off[s + 1];
        int best_win = lo;
        for (int win = lo + 1; win < hi; ++win)
            if (res.final_scores[win] > res.final_scores[best_win]) best_win = win;
        // Convert the absolute data index back to an offset within sequence s.
        res.best_site[s] = set.start_of_win[best_win] - set.offset[s];
    }
    return res;
}
