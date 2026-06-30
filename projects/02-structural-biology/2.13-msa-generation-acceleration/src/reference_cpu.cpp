// ===========================================================================
// src/reference_cpu.cpp  --  Data loader + the plain-C++ Viterbi baseline
// ---------------------------------------------------------------------------
// Project 2.13 : MSA Generation Acceleration
//
// ROLE IN THE PROJECT
//   (1) load_problem(): parse the tiny text dataset (format in data/README.md)
//       into a SearchProblem (one query profile HMM + a packed sequence DB).
//   (2) viterbi_search_cpu(): the "ground truth" the GPU result is checked
//       against -- a single readable loop, no parallelism, calling the SAME
//       shared recurrence (viterbi_step in hmm_core.h) the kernel uses. When the
//       GPU and CPU agree (they should, exactly: integer math), we believe the GPU.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, hmm_core.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// aa_to_index : amino-acid LETTER -> 0..20 index (20 = unknown / non-standard).
//   The 20 standard amino acids in the conventional order used by most scoring
//   matrices (BLOSUM/PAM). Anything else (X, B, Z, '-', lowercase, junk) maps to
//   the catch-all index 20, so a messy database line can never read out of the
//   emission table. A plain switch keeps the mapping explicit and dependency-free.
// ---------------------------------------------------------------------------
int aa_to_index(char c) {
    switch (c) {
        case 'A': return 0;  case 'R': return 1;  case 'N': return 2;
        case 'D': return 3;  case 'C': return 4;  case 'Q': return 5;
        case 'E': return 6;  case 'G': return 7;  case 'H': return 8;
        case 'I': return 9;  case 'L': return 10; case 'K': return 11;
        case 'M': return 12; case 'F': return 13; case 'P': return 14;
        case 'S': return 15; case 'T': return 16; case 'W': return 17;
        case 'Y': return 18; case 'V': return 19;
        default:  return 20;   // X / non-standard / gap -> catch-all slot
    }
}

// ---------------------------------------------------------------------------
// load_problem : parse the dataset text format (see data/README.md).
//
//   The format (whitespace-separated tokens, '#'-prefixed comment lines skipped):
//     L  N                         <- profile length, number of DB sequences
//     t_mm t_mi t_im t_ii t_md t_dm t_dd   <- 7 transition log-odds (scaled int)
//     <L lines>: each is ALPHABET_SIZE(=21) emission log-odds (scaled int) for
//                one match column, residues in the aa_to_index order, slot 20 last
//     <N lines>: each is one database sequence as amino-acid LETTERS
//
//   We read all NUMERIC tokens for the header + transitions + emission table with
//   a stream, then read the N sequence lines as text. Numbers are pre-scaled
//   integers (log-odds * SCORE_SCALE) so no floating point enters the DP.
// ---------------------------------------------------------------------------
SearchProblem load_problem(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open dataset file: " + path);

    // Helper: read the next non-comment, non-blank line into `line`.
    auto next_line = [&](std::string& line) -> bool {
        while (std::getline(in, line)) {
            // Strip a trailing '\r' if the file has Windows line endings.
            if (!line.empty() && line.back() == '\r') line.pop_back();
            // Skip blank lines and full-line comments beginning with '#'.
            std::string trimmed = line;
            std::size_t first = trimmed.find_first_not_of(" \t");
            if (first == std::string::npos) continue;          // all whitespace
            if (trimmed[first] == '#') continue;               // comment line
            return true;
        }
        return false;
    };

    SearchProblem prob;
    std::string line;

    // ---- header: L N ----
    if (!next_line(line)) throw std::runtime_error("missing header line in " + path);
    {
        std::istringstream ss(line);
        if (!(ss >> prob.hmm.L >> prob.db.N))
            throw std::runtime_error("bad header (expected 'L N') in " + path);
        if (prob.hmm.L <= 0 || prob.db.N <= 0)
            throw std::runtime_error("non-positive L or N in " + path);
    }

    // ---- transitions: 7 scaled-integer log-odds ----
    if (!next_line(line)) throw std::runtime_error("missing transition line in " + path);
    {
        std::istringstream ss(line);
        ProfileHMM& h = prob.hmm;
        if (!(ss >> h.t_mm >> h.t_mi >> h.t_im >> h.t_ii >> h.t_md >> h.t_dm >> h.t_dd))
            throw std::runtime_error("bad transition line (need 7 ints) in " + path);
    }

    // ---- emission table: L rows of ALPHABET_SIZE scaled-integer log-odds ----
    prob.hmm.emit.resize(static_cast<std::size_t>(prob.hmm.L) * ALPHABET_SIZE);
    for (int k = 0; k < prob.hmm.L; ++k) {
        if (!next_line(line))
            throw std::runtime_error("missing emission row " + std::to_string(k) + " in " + path);
        std::istringstream ss(line);
        for (int a = 0; a < ALPHABET_SIZE; ++a) {
            int v;
            if (!(ss >> v))
                throw std::runtime_error("emission row " + std::to_string(k) +
                                         " needs " + std::to_string(ALPHABET_SIZE) +
                                         " ints in " + path);
            prob.hmm.emit[static_cast<std::size_t>(k) * ALPHABET_SIZE + a] = v;
        }
    }

    // ---- database: N sequence lines (amino-acid letters) ----
    //   Pack into CSR: concatenate residues into `res`, record start offsets.
    prob.db.offset.assign(static_cast<std::size_t>(prob.db.N) + 1, 0);
    prob.db.length.assign(static_cast<std::size_t>(prob.db.N), 0);
    for (int i = 0; i < prob.db.N; ++i) {
        if (!next_line(line))
            throw std::runtime_error("missing database sequence " + std::to_string(i) + " in " + path);
        // Encode each letter; ignore in-line whitespace so the file can be tidy.
        int len = 0;
        for (char c : line) {
            if (c == ' ' || c == '\t') continue;
            prob.db.res.push_back(static_cast<uint8_t>(aa_to_index(c)));
            ++len;
        }
        if (len == 0)
            throw std::runtime_error("empty database sequence " + std::to_string(i) + " in " + path);
        prob.db.length[static_cast<std::size_t>(i)] = len;
        prob.db.offset[static_cast<std::size_t>(i) + 1] =
            prob.db.offset[static_cast<std::size_t>(i)] + len;
    }
    return prob;
}

// ---------------------------------------------------------------------------
// viterbi_search_cpu : score the profile against every database sequence.
//
//   For each sequence we run the three-state Viterbi sweep ROW BY ROW, keeping
//   two rows (previous / current) of length L+1 -- O(L) memory. After each row we
//   record the best match score (best_in_row); the sequence's HIT SCORE is the
//   maximum over all rows (a local-style score, ../THEORY.md "The algorithm").
//
//   Complexity: O(T_i * L) per sequence i, O((sum T_i) * L) total. This serial
//   loop is the baseline whose wall time (timed in main.cu) we compare against the
//   GPU kernel -- and the trusted answer the kernel is verified against.
//
//   The DP calls viterbi_step() and best_in_row() from hmm_core.h: the EXACT
//   functions the GPU kernel calls, so CPU and GPU agree bit-for-bit (integers).
// ---------------------------------------------------------------------------
void viterbi_search_cpu(const SearchProblem& prob, std::vector<int>& out) {
    const ProfileHMM& h = prob.hmm;
    const SeqDB& db = prob.db;
    const int L = h.L;
    out.assign(static_cast<std::size_t>(db.N), NEG_INF);

    // Two ping-pong rows of the DP (length L+1: column 0 is the begin state).
    std::vector<int> prevM(L + 1), prevI(L + 1), prevD(L + 1);
    std::vector<int> curM(L + 1),  curI(L + 1),  curD(L + 1);

    for (int i = 0; i < db.N; ++i) {
        const int T = db.length[static_cast<std::size_t>(i)];          // sequence length
        const uint8_t* seq = &db.res[static_cast<std::size_t>(db.offset[static_cast<std::size_t>(i)])];

        // Row 0 (before consuming any residue): only the begin state is reachable.
        // M[0]=0 means "we may start the alignment here for free"; everything else
        // is impossible. (We initialise prev* as the "row -1" so the first
        // viterbi_step sees a valid begin state at column 0 of the previous row.)
        for (int k = 0; k <= L; ++k) { prevM[k] = NEG_INF; prevI[k] = NEG_INF; prevD[k] = NEG_INF; }
        prevM[0] = 0;   // free entry at the profile start

        int best = NEG_INF;   // best match score seen across all rows of this seq

        for (int r = 0; r < T; ++r) {
            const int a = static_cast<int>(seq[r]);   // this residue's aa index
            // One row of the shared recurrence -> fills cur* from prev*.
            viterbi_step(L, h.emit.data(), a,
                         h.t_mm, h.t_mi, h.t_im, h.t_ii, h.t_md, h.t_dm, h.t_dd,
                         prevM.data(), prevI.data(), prevD.data(),
                         curM.data(), curI.data(), curD.data());
            // Could the best alignment END at this residue? Track the running max.
            const int rb = best_in_row(L, curM.data());
            if (rb > best) best = rb;
            // Ping-pong: current row becomes the previous row for the next residue.
            prevM.swap(curM); prevI.swap(curI); prevD.swap(curD);
        }
        out[static_cast<std::size_t>(i)] = best;
    }
}
