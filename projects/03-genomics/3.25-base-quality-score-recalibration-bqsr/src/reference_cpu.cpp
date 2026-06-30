// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial BQSR reference (the trusted baseline)
// ---------------------------------------------------------------------------
// Project 3.25 : Base Quality Score Recalibration (BQSR)
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU is checked against. Written to be OBVIOUSLY
//   correct -- single readable loops, no parallelism -- so that when the GPU and
//   CPU agree we believe the GPU. The per-base covariate decision and the
//   empirical-quality math live in bqsr.h (shared host+device), so this file just
//   loops classify_base() over every base and tallies integer counts.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//   Compare each function against its GPU twin in kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <cctype>      // std::toupper
#include <fstream>     // std::ifstream
#include <sstream>     // std::istringstream
#include <stdexcept>   // std::runtime_error
#include <string>

// ---------------------------------------------------------------------------
// load_dataset: parse the data/sample text format into a Dataset.
//
// FILE FORMAT (documented in data/README.md):
//     REF <reference-string>            # 1 line: the reference bases (ACGTN)
//     KNOWN <p1> <p2> ...               # 1 line: known-variant reference positions
//     READS <R> <L>                     # R reads, each of length L
//     <pos> <bases(L chars)> <q0> <q1> ... <q(L-1)>     # one line per read
//     ... (R such lines)
//
// We read it line by line. Anything malformed throws so demos fail loudly rather
// than silently clustering garbage.
// ---------------------------------------------------------------------------
Dataset load_dataset(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open dataset file: " + path);

    Dataset d;
    std::string line, tag;

    // ---- REF line ----------------------------------------------------------
    if (!std::getline(in, line)) throw std::runtime_error("empty dataset: " + path);
    {
        std::istringstream ss(line);
        ss >> tag >> d.reference;
        if (tag != "REF" || d.reference.empty())
            throw std::runtime_error("expected 'REF <string>' first in " + path);
        // Normalize to uppercase so 'a' and 'A' bin identically.
        for (char& c : d.reference) c = static_cast<char>(std::toupper((unsigned char)c));
    }
    const int ref_len = static_cast<int>(d.reference.size());

    // Start with no known sites; the KNOWN line flips bits on.
    d.known_site.assign(static_cast<std::size_t>(ref_len), 0u);

    // ---- KNOWN line --------------------------------------------------------
    if (!std::getline(in, line)) throw std::runtime_error("missing KNOWN line in " + path);
    {
        std::istringstream ss(line);
        ss >> tag;
        if (tag != "KNOWN") throw std::runtime_error("expected 'KNOWN ...' line in " + path);
        int p;
        while (ss >> p)
            if (p >= 0 && p < ref_len) d.known_site[static_cast<std::size_t>(p)] = 1u;
    }

    // ---- READS header ------------------------------------------------------
    if (!std::getline(in, line)) throw std::runtime_error("missing READS header in " + path);
    {
        std::istringstream ss(line);
        ss >> tag >> d.num_reads >> d.read_len;
        if (tag != "READS" || d.num_reads <= 0 || d.read_len <= 0 || d.read_len > MAX_CYCLE)
            throw std::runtime_error("bad 'READS <R> <L>' (L must be <= MAX_CYCLE) in " + path);
    }

    // Allocate the flat read arrays now that we know R and L.
    const std::size_t total = static_cast<std::size_t>(d.num_reads) * d.read_len;
    d.read_bases.resize(total);
    d.read_quals.resize(total);
    d.read_pos.resize(static_cast<std::size_t>(d.num_reads));

    // ---- one line per read -------------------------------------------------
    for (int i = 0; i < d.num_reads; ++i) {
        if (!std::getline(in, line))
            throw std::runtime_error("dataset truncated: fewer reads than declared in " + path);
        std::istringstream ss(line);
        int pos;
        std::string bases;
        if (!(ss >> pos >> bases) || static_cast<int>(bases.size()) != d.read_len)
            throw std::runtime_error("bad read row (need pos + L-char base string) in " + path);
        d.read_pos[static_cast<std::size_t>(i)] = pos;
        for (int c = 0; c < d.read_len; ++c) {
            const std::size_t g = static_cast<std::size_t>(i) * d.read_len + c;
            d.read_bases[g] = static_cast<char>(std::toupper((unsigned char)bases[c]));
            int q;
            if (!(ss >> q))
                throw std::runtime_error("bad read row (need L quality scores) in " + path);
            d.read_quals[g] = q;
        }
    }
    return d;
}

// ---------------------------------------------------------------------------
// build_table_cpu: serial covariate-table accumulation (the BQSR baseline).
//   For every base, classify_base() (bqsr.h) tells us whether to tally it and, if
//   so, the bin and whether it was an error. We add 1 to that bin's obs counter
//   (always) and to its err counter (if it was a mismatch). These are INTEGER
//   counts, so the order of accumulation does not matter -- which is exactly why
//   the GPU's atomic version (kernels.cu) reproduces this table bit-for-bit.
//   Complexity: O(R*L) time, O(NUM_BINS) space.
// ---------------------------------------------------------------------------
void build_table_cpu(const Dataset& d,
                     std::vector<unsigned int>& obs,
                     std::vector<unsigned int>& err) {
    obs.assign(static_cast<std::size_t>(NUM_BINS), 0u);
    err.assign(static_cast<std::size_t>(NUM_BINS), 0u);

    const int ref_len = static_cast<int>(d.reference.size());
    const char*          ref      = d.reference.data();
    const char*          bases    = d.read_bases.data();
    const int*           quals    = d.read_quals.data();
    const int*           read_pos = d.read_pos.data();
    const unsigned char* known    = d.known_site.data();

    for (int g = 0; g < d.total_bases(); ++g) {
        int bin = 0, is_err = 0;
        if (!classify_base(g, d.read_len, ref, ref_len, bases, quals,
                           read_pos, known, &bin, &is_err))
            continue;                                     // masked / skipped base
        obs[static_cast<std::size_t>(bin)] += 1u;          // one more observation
        err[static_cast<std::size_t>(bin)] += static_cast<unsigned int>(is_err);
    }
}

// ---------------------------------------------------------------------------
// recalibrate_cpu: apply a finished (obs,err) table to every base.
//   The recalibrated quality is the empirical quality of the base's covariate
//   bin (empirical_q in bqsr.h). A base whose bin has no evidence (Q_emp == -1)
//   -- including masked/skipped bases that never contributed -- keeps its
//   original reported quality. We recompute the bin with classify_base so a
//   skipped base is detected and left unchanged. Output matches the GPU twin.
// ---------------------------------------------------------------------------
void recalibrate_cpu(const Dataset& d,
                     const std::vector<unsigned int>& obs,
                     const std::vector<unsigned int>& err,
                     std::vector<int>& new_quals) {
    new_quals.assign(static_cast<std::size_t>(d.total_bases()), 0);

    const int ref_len = static_cast<int>(d.reference.size());
    const char*          ref      = d.reference.data();
    const char*          bases    = d.read_bases.data();
    const int*           quals    = d.read_quals.data();
    const int*           read_pos = d.read_pos.data();
    const unsigned char* known    = d.known_site.data();

    for (int g = 0; g < d.total_bases(); ++g) {
        const int orig = quals[g];                        // reported quality
        int bin = 0, is_err = 0;
        if (!classify_base(g, d.read_len, ref, ref_len, bases, quals,
                           read_pos, known, &bin, &is_err)) {
            new_quals[static_cast<std::size_t>(g)] = orig; // skipped: keep reported
            continue;
        }
        const int qe = empirical_q(obs[static_cast<std::size_t>(bin)],
                                   err[static_cast<std::size_t>(bin)]);
        new_quals[static_cast<std::size_t>(g)] = (qe < 0) ? orig : qe;
    }
}
