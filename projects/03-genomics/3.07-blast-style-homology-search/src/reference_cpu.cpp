// ===========================================================================
// src/reference_cpu.cpp  --  BLOSUM62, FASTA loader, query index, CPU reference
// ---------------------------------------------------------------------------
// Project 3.7 : BLAST-Style Homology Search
//
// ROLE
//   (1) blosum62()         -- the canonical 24x24 substitution-score table.
//   (2) load_fasta()       -- parse query + DB sequences, encode residues.
//   (3) build_query_index()-- the query k-mer hash that both CPU and GPU seed on.
//   (4) blast_cpu()        -- the obviously-correct serial search the GPU is
//                             verified against. No cleverness on purpose.
//
//   Compiled by the host C++ compiler only (no CUDA). The actual per-residue
//   scoring (gapless X-drop) lives in blast_core.h, shared verbatim with the
//   GPU kernel, so CPU and GPU produce BIT-IDENTICAL scores (all integers).
//
// READ THIS AFTER: reference_cpu.h, blast_core.h. Compare with kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

// ===========================================================================
// (1) BLOSUM62
// ---------------------------------------------------------------------------
// The standard protein substitution matrix (Henikoff & Henikoff 1992), in the
// classic NCBI row/column order  A R N D C Q E G H I L K M F P S T W Y V B Z X *
// -- which is EXACTLY our ALPHA ordering in blast_core.h. Each entry score(a,b)
// is the log-odds (base 2, scaled) that residues a and b are aligned in truly
// homologous proteins versus by chance: the diagonal (identities) is strongly
// positive, conservative swaps (e.g. I<->L, K<->R) are mildly positive, and
// dissimilar pairs are negative. Storing it as int8 (range fits in [-4, 11])
// keeps it tiny (576 bytes) -- small enough to live in GPU constant memory.
//
// We define it once as a static table and hand out a pointer; the GPU copies
// these same bytes to a __constant__ symbol (see kernels.cu). Because the
// values are integers, every score computed from them is exact on both sides.
// ===========================================================================
static const int8_t BLOSUM62[N_ALPHA * N_ALPHA] = {
    //   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  B  Z  X  *
    /*A*/ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0,-2,-1, 0,-4,
    /*R*/-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3,-1, 0,-1,-4,
    /*N*/-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3, 3, 0,-1,-4,
    /*D*/-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3, 4, 1,-1,-4,
    /*C*/ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,-3,-3,-2,-4,
    /*Q*/-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2, 0, 3,-1,-4,
    /*E*/-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2, 1, 4,-1,-4,
    /*G*/ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3,-1,-2,-1,-4,
    /*H*/-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3, 0, 0,-1,-4,
    /*I*/-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3,-3,-3,-1,-4,
    /*L*/-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1,-4,-3,-1,-4,
    /*K*/-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2, 0, 1,-1,-4,
    /*M*/-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1,-3,-1,-1,-4,
    /*F*/-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1,-3,-3,-1,-4,
    /*P*/-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2,-2,-1,-2,-4,
    /*S*/ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2, 0, 0, 0,-4,
    /*T*/ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0,-1,-1, 0,-4,
    /*W*/-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3,-4,-3,-2,-4,
    /*Y*/-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1,-3,-2,-1,-4,
    /*V*/ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4,-3,-2,-1,-4,
    /*B*/-2,-1, 3, 4,-3, 0, 1,-1, 0,-3,-4, 0,-3,-3,-2, 0,-1,-4,-3,-3, 4, 1,-1,-4,
    /*Z*/-1, 0, 0, 1,-3, 3, 4,-2, 0,-3,-3, 1,-1,-3,-1, 0,-1,-3,-2,-2, 1, 4,-1,-4,
    /*X*/ 0,-1,-1,-1,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2, 0, 0,-2,-1,-1,-1,-1,-1,-4,
    /***/-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4, 1,
};

const int8_t* blosum62() { return BLOSUM62; }

// ===========================================================================
// (2) FASTA loader
// ---------------------------------------------------------------------------
// A FASTA file is a sequence of records, each:
//     >header text
//     RESIDUELETTERS (possibly wrapped over many lines)
// We treat the FIRST record as the QUERY and every subsequent record as a DB
// sequence. Residues are encoded to 0..23 (encode_residue, blast_core.h) on the
// fly. Unknown bytes map to 'X'; blank lines and whitespace are ignored.
// ===========================================================================
SequenceDB load_fasta(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open FASTA file: " + path);

    // First pass: collect (header, encoded-residues) records in order.
    std::vector<std::string>          headers;
    std::vector<std::vector<int8_t>>  seqs;
    std::string line;
    while (std::getline(in, line)) {
        // Strip a trailing '\r' so Windows-CRLF files parse identically.
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (line[0] == '>') {
            // New record: header is everything after '>' up to first space.
            std::string h = line.substr(1);
            std::size_t sp = h.find_first_of(" \t");
            if (sp != std::string::npos) h = h.substr(0, sp);
            headers.push_back(h);
            seqs.emplace_back();
        } else {
            // Residue line: append encoded residues to the current record.
            if (seqs.empty())
                throw std::runtime_error("FASTA: residues before any '>' header in " + path);
            for (char c : line) {
                if (std::isspace(static_cast<unsigned char>(c))) continue;
                seqs.back().push_back(static_cast<int8_t>(encode_residue(c)));
            }
        }
    }
    if (seqs.size() < 2)
        throw std::runtime_error("FASTA must contain a query + >=1 DB sequence in " + path);

    // Second pass: pack into the SequenceDB layout (query + concatenated DB).
    SequenceDB db;
    db.query_name = headers[0];
    db.query      = seqs[0];

    db.n = static_cast<int>(seqs.size()) - 1;
    db.db_off.reserve(db.n);
    db.db_len.reserve(db.n);
    db.names.reserve(db.n);
    int off = 0;
    for (int i = 1; i < static_cast<int>(seqs.size()); ++i) {
        const auto& s = seqs[i];
        db.names.push_back(headers[i]);
        db.db_off.push_back(off);
        db.db_len.push_back(static_cast<int>(s.size()));
        db.db_res.insert(db.db_res.end(), s.begin(), s.end());  // concatenate
        off += static_cast<int>(s.size());
    }
    return db;
}

// ===========================================================================
// (3) Query k-mer index
// ---------------------------------------------------------------------------
// Slide a length-k window across the query; for each valid window record its
// packed code -> position. This is the prefilter index: a DB k-mer is a SEED
// iff its code appears here. Built once and reused by BOTH the CPU reference
// and (after flattening, in kernels.cu) the GPU kernel, so they seed identically.
// ===========================================================================
QueryIndex build_query_index(const std::vector<int8_t>& query, int k) {
    QueryIndex qi;
    const int len = static_cast<int>(query.size());
    for (int p = 0; p + k <= len; ++p) {
        int code = pack_kmer(query.data(), len, p, k);
        if (code < 0) continue;            // window contained an 'X' -> skip
        qi.table[code].push_back(p);       // record this query position
    }
    return qi;
}

// ===========================================================================
// (4) CPU reference search
// ---------------------------------------------------------------------------
// For each DB sequence, slide its k-mer window; on a hit in the query index,
// run gapless X-drop extension from every (qpos,dpos) seed and keep the best
// HSP score. The result best_score[i] is the homology score of DB sequence i.
//
// This is the SAME logic the GPU thread runs (kernels.cu) -- one DB sequence at
// a time -- but here serially over all i. It is the ground truth for verify.
// ===========================================================================
void blast_cpu(const SequenceDB& db, const QueryIndex& query_idx,
               std::vector<int>& best_score) {
    best_score.assign(static_cast<std::size_t>(db.n), 0);
    const int8_t* mat = blosum62();
    const SeqView q   = db.query_view();

    for (int i = 0; i < db.n; ++i) {
        const SeqView d = db.db_view(i);
        int best = 0;                       // best HSP score for this DB sequence

        // Slide the length-k window across DB sequence i.
        for (int dpos = 0; dpos + SEED_K <= d.len; ++dpos) {
            int code = pack_kmer(d.data, d.len, dpos, SEED_K);
            if (code < 0) continue;         // ambiguous window -> no seed
            auto it = query_idx.table.find(code);
            if (it == query_idx.table.end()) continue;  // not in query -> no seed

            // Every query position with this k-mer is a seed on its own diagonal.
            for (int qpos : it->second) {
                int hsp = gapless_xdrop(q, d, qpos, dpos, SEED_K, mat, X_DROP);
                if (hsp > best) best = hsp;  // keep the maximal HSP
            }
        }
        best_score[static_cast<std::size_t>(i)] = best;
    }
}
