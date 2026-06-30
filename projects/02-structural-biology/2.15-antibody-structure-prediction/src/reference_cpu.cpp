// ===========================================================================
// src/reference_cpu.cpp  --  Trusted serial CDR-screening baseline + loader
// ---------------------------------------------------------------------------
// Project 2.15 : Antibody Structure Prediction  (reduced-scope: CDR screening)
//
// ROLE
//   (1) load_library(): parse the tiny text dataset (format in data/README.md),
//       encoding each CDR string to integers and padding to AB_CDR_LEN.
//   (2) score_cpu(): the obviously-correct serial computation the GPU kernel is
//       verified against. No cleverness on purpose -- a single loop over the
//       library calling the shared scoring core ab_cdr_score(). If CPU and GPU
//       agree, we trust the GPU.
//
//   Compiled by the host C++ compiler only (no CUDA). The scoring math itself
//   lives in antibody.h as __host__ __device__ functions, so this file and the
//   GPU kernel run byte-for-byte identical arithmetic (PATTERNS.md §2).
//
// READ THIS AFTER: reference_cpu.h, antibody.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// encode_cdr_into: encode one ASCII CDR token into a destination CDR field of
//   AB_CDR_LEN encoded residues, right-padded with the gap symbol.
//     token : the amino-acid string for this CDR (e.g. "ARDYYGSGS")
//     dst   : pointer to AB_CDR_LEN bytes to fill (one CDR field of a record)
//     returns: true if the token had to be TRUNCATED (longer than AB_CDR_LEN).
//   We pad with AB_GAP so a short CDR aligned against a short CDR scores on its
//   real residues plus a uniform run of gap-vs-gap (a small constant) -- see
//   antibody.h ab_score_one_cdr for why that is harmless.
// ---------------------------------------------------------------------------
static bool encode_cdr_into(const std::string& token, uint8_t* dst) {
    const int L = static_cast<int>(token.size());
    const int keep = (L < AB_CDR_LEN) ? L : AB_CDR_LEN;     // clamp to field width
    for (int p = 0; p < keep; ++p)
        dst[p] = static_cast<uint8_t>(ab_encode_residue(token[p]));
    for (int p = keep; p < AB_CDR_LEN; ++p)
        dst[p] = static_cast<uint8_t>(AB_GAP);              // right-pad with gaps
    return L > AB_CDR_LEN;                                  // did we drop residues?
}

// ---------------------------------------------------------------------------
// parse_record: read "<name> <H1> <H2> <H3> <L1> <L2> <L3>" from a line stream
//   into a destination record (AB_RECORD_LEN encoded residues) + its name.
//   Returns the number of CDR tokens that were truncated. Throws if a line does
//   not have all six CDR tokens (malformed dataset -> fail loudly).
// ---------------------------------------------------------------------------
static int parse_record(std::istringstream& ls, uint8_t* record, std::string& name) {
    if (!(ls >> name))
        throw std::runtime_error("antibody record is missing its name field");
    int truncated = 0;
    for (int c = 0; c < AB_NUM_CDRS; ++c) {                 // H1,H2,H3,L1,L2,L3
        std::string tok;
        if (!(ls >> tok))
            throw std::runtime_error("antibody '" + name + "' is missing CDR token " +
                                     std::to_string(c + 1) + " of 6");
        if (encode_cdr_into(tok, record + c * AB_CDR_LEN))  // fill CDR field c
            ++truncated;
    }
    return truncated;
}

AntibodyLibrary load_library(const std::string& path, int* truncated) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open antibody dataset: " + path);

    AntibodyLibrary ab;
    int trunc_total = 0;
    bool have_query = false;

    // We read the file line by line so we can skip '#' comments and blank lines.
    // The query line is tagged "QUERY"; every other data line is a library entry.
    std::string line;
    std::vector<uint8_t> rec(AB_RECORD_LEN);                // scratch for one record
    while (std::getline(in, line)) {
        // Strip a trailing CR (Windows files) and skip comment/blank lines.
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::size_t first = line.find_first_not_of(" \t");
        if (first == std::string::npos || line[first] == '#') continue;

        std::istringstream ls(line);
        std::string tag;
        ls >> tag;                                          // peek first token
        if (tag == "QUERY") {
            // The rest of the line is "<name> <6 CDRs>".
            std::string rest;
            std::getline(ls, rest);
            std::istringstream qs(rest);
            trunc_total += parse_record(qs, rec.data(), ab.query_name);
            ab.query.assign(rec.begin(), rec.end());
            have_query = true;
        } else {
            // A library record. The first token (tag) is its name, so rebuild a
            // stream positioned at the name and parse the whole record.
            std::istringstream rs(line);
            std::string name;
            trunc_total += parse_record(rs, rec.data(), name);
            ab.names.push_back(name);
            ab.lib.insert(ab.lib.end(), rec.begin(), rec.end());
            ++ab.n;
        }
    }

    if (!have_query) throw std::runtime_error("dataset has no QUERY line: " + path);
    if (ab.n <= 0)   throw std::runtime_error("dataset has no library antibodies: " + path);
    if (truncated) *truncated = trunc_total;
    return ab;
}

// ---------------------------------------------------------------------------
// score_cpu: the serial reference. For each library antibody i, compute the
//   CDR-weighted similarity to the query via the shared core ab_cdr_score().
//   Complexity: O(n * AB_RECORD_LEN) integer additions; O(1) extra space. This
//   is the baseline whose wall time (timed in main.cu) we compare with the GPU.
//   Because ab_cdr_score is exact integer arithmetic, this matches the GPU
//   result bit-for-bit (THEORY "How we verify correctness").
// ---------------------------------------------------------------------------
void score_cpu(const AntibodyLibrary& ab, std::vector<int32_t>& out) {
    out.assign(static_cast<std::size_t>(ab.n), 0);
    for (int i = 0; i < ab.n; ++i) {
        // Row i of the library is AB_RECORD_LEN encoded residues.
        const uint8_t* lib_i = &ab.lib[static_cast<std::size_t>(i) * AB_RECORD_LEN];
        out[static_cast<std::size_t>(i)] = ab_cdr_score(ab.query.data(), lib_i);
    }
}
