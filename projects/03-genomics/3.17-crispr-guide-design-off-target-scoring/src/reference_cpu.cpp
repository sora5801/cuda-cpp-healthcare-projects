// ===========================================================================
// src/reference_cpu.cpp  --  The trusted serial CRISPR scan + data loader
// ---------------------------------------------------------------------------
// Project 3.17 : CRISPR Guide Design & Off-Target Scoring
//
// ROLE IN THE PROJECT
//   (1) encode_base/decode_base : the ASCII <-> 2-bit nucleotide mapping used by
//       BOTH the loader and the reporter, so every encoding is consistent.
//   (2) load_problem()          : parse the tiny text dataset (data/README.md).
//   (3) scan_cpu()              : the obviously-correct serial off-target scan
//       the GPU kernel is verified against. It does the SAME per-window work as
//       the kernel by calling the SAME score_window() from cfd_score.h -- so a
//       disagreement can only mean a GPU bug, never a different formula.
//   (4) specificity_score()     : the guide-level summary metric.
//
//   Compiled by the host C++ compiler only (no CUDA). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, cfd_score.h. Compare scan_cpu() against
// scan_kernel() in kernels.cu -- they are deliberately the same loop body.
// ===========================================================================
#include "reference_cpu.h"

#include <cctype>      // std::toupper
#include <fstream>     // std::ifstream
#include <sstream>     // std::istringstream
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// encode_base: ASCII nucleotide -> 2-bit code. A switch keeps the mapping
// explicit and branch-predictable; anything not A/C/G/T (including 'N' ambiguity
// codes) becomes BASE_INVALID so it can never spuriously match a guide base.
// ---------------------------------------------------------------------------
uint8_t encode_base(char c) {
    switch (c) {
        case 'A': return BASE_A;   // adenine
        case 'C': return BASE_C;   // cytosine
        case 'G': return BASE_G;   // guanine
        case 'T': return BASE_T;   // thymine
        default:  return BASE_INVALID;  // N, gaps, anything else
    }
}

// decode_base: 2-bit code -> ASCII, for printing the matched protospacer.
char decode_base(uint8_t code) {
    switch (code) {
        case BASE_A: return 'A';
        case BASE_C: return 'C';
        case BASE_G: return 'G';
        case BASE_T: return 'T';
        default:     return 'N';   // invalid / ambiguous
    }
}

// ---------------------------------------------------------------------------
// load_problem: read the data/sample format. We accept three kinds of line:
//   * "# ..."                    -> comment, skipped
//   * "guide <NAME> <SEQ>"       -> the 20-base guide spacer (SEQ is ACGT)
//   * "genome <SEQ>"             -> append SEQ to the genome (may repeat)
// Whitespace is ignored. All bases are uppercased then encoded. We validate the
// guide length and that the genome holds at least one 23-base window, so the
// demo fails loudly on a malformed file rather than silently scanning nothing.
// ---------------------------------------------------------------------------
CrisprProblem load_problem(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open CRISPR data file: " + path);

    CrisprProblem prob;
    std::string raw_guide;     // accumulates the guide letters
    std::string raw_genome;    // accumulates the genome letters

    std::string line;
    while (std::getline(in, line)) {
        // Trim a trailing '\r' so Windows-CRLF files parse identically on Linux.
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;     // blank / comment

        std::istringstream ls(line);
        std::string tag;
        ls >> tag;
        if (tag == "guide") {
            // "guide <name> <sequence>" -- name is for the report only.
            ls >> prob.guide_name >> raw_guide;
        } else if (tag == "genome") {
            // "genome <sequence>" -- one or more such lines concatenate.
            std::string chunk;
            ls >> chunk;
            raw_genome += chunk;
        } else {
            // Unknown leading token: treat the WHOLE line as genome bases so a
            // bare wrapped FASTA-like sequence still loads (lenient on purpose).
            raw_genome += line;
        }
    }

    if (static_cast<int>(raw_guide.size()) != GUIDE_LEN)
        throw std::runtime_error("guide must be exactly " + std::to_string(GUIDE_LEN) +
                                 " bases; got " + std::to_string(raw_guide.size()) +
                                 " in " + path);

    // Encode the guide; reject any non-ACGT base (a guide cannot contain N).
    prob.guide.resize(GUIDE_LEN);
    for (int i = 0; i < GUIDE_LEN; ++i) {
        uint8_t code = encode_base(static_cast<char>(std::toupper(raw_guide[i])));
        if (code == BASE_INVALID)
            throw std::runtime_error("guide has a non-ACGT base at position " +
                                     std::to_string(i) + " in " + path);
        prob.guide[i] = code;
    }

    // Encode the genome. Non-ACGT characters are kept as BASE_INVALID (they can
    // never match a guide base nor form a G of the PAM), which is the correct
    // behavior for masked/ambiguous reference regions.
    prob.genome.reserve(raw_genome.size());
    for (char c : raw_genome)
        prob.genome.push_back(encode_base(static_cast<char>(std::toupper(c))));

    prob.genome_len = static_cast<int>(prob.genome.size());
    prob.n_windows  = prob.genome_len - WINDOW_LEN + 1;   // # sliding windows
    if (prob.n_windows <= 0)
        throw std::runtime_error("genome too short (" + std::to_string(prob.genome_len) +
                                 " bases) to hold a single " + std::to_string(WINDOW_LEN) +
                                 "-base window in " + path);
    return prob;
}

// ---------------------------------------------------------------------------
// scan_cpu: slide the 23-base window across the genome. For each start position
// i, the protospacer is genome[i .. i+19] and the PAM is genome[i+20 .. i+22];
// we hand both to the shared score_window() and store the result. This is an
// O(genome_len * GUIDE_LEN) serial loop -- the exact work the GPU spreads across
// threads (one thread per i) in kernels.cu.
// ---------------------------------------------------------------------------
void scan_cpu(const CrisprProblem& prob, ScanResult& out) {
    out.mismatches.assign(static_cast<std::size_t>(prob.n_windows), -1);
    out.cfd.assign(static_cast<std::size_t>(prob.n_windows), 0.0);

    const uint8_t* g = prob.guide.data();      // [GUIDE_LEN] spacer
    for (int i = 0; i < prob.n_windows; ++i) {
        const uint8_t* proto = &prob.genome[static_cast<std::size_t>(i)];               // 20 bases
        const uint8_t* pam   = &prob.genome[static_cast<std::size_t>(i) + GUIDE_LEN];   // 3 bases
        WindowScore ws = score_window(g, proto, pam);   // the ONE TRUE scorer
        out.mismatches[static_cast<std::size_t>(i)] = ws.mismatches;
        out.cfd[static_cast<std::size_t>(i)]        = ws.cfd;
    }
}

// ---------------------------------------------------------------------------
// specificity_score: CRISPOR/MIT aggregate. See reference_cpu.h for the formula
// and meaning. Kept here (not inlined at the call site) so the CPU and GPU code
// paths both call the identical implementation and report the same number.
// ---------------------------------------------------------------------------
double specificity_score(double sum_offtarget_cfd) {
    return 100.0 / (100.0 + 100.0 * sum_offtarget_cfd);
}
