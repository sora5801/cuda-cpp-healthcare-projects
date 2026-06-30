// ===========================================================================
// src/reference_cpu.cpp  --  Loader, DB builder, and the CPU reference
// ---------------------------------------------------------------------------
// Project 3.14 : Metagenomic Taxonomic Classification
//
// ROLE IN THE PROJECT
//   Three host-only jobs, all CUDA-free (compiled by cl.exe / g++):
//     (1) load_problem()    -- parse the self-contained text dataset.
//     (2) build_database()  -- insert reference k-mers into the hash table. The
//         INSERTION here and the LOOK-UP in kmer_core.h must use the same hash
//         and the same linear walk, or look-ups would miss inserted keys; they
//         do, by construction.
//     (3) classify_cpu()    -- the trusted baseline. It calls the SHARED
//         classify_read() from kmer_core.h -- the exact function the GPU kernel
//         calls -- so the two produce integer-identical taxon ids (tolerance 0).
//
//   Written to be obviously correct: a single readable loop per job, no
//   parallelism, no cleverness. When the GPU agrees with this, we trust the GPU.
//
// READ THIS AFTER: reference_cpu.h, kmer_core.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// next_pow2_at_least: smallest power of two >= v (and >= 1). The hash table
// capacity must be a power of two so the probe can wrap with (& (capacity-1))
// instead of a modulo. We size the table to keep the load factor below ~0.5,
// which keeps linear-probe chains short.
// ---------------------------------------------------------------------------
static uint64_t next_pow2_at_least(uint64_t v) {
    uint64_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

// ---------------------------------------------------------------------------
// insert_kmer: place (key -> taxon) into the open-addressing table via linear
// probing, mirroring the look-up walk in kmer_core.h::table_lookup.
//   * Walk from the home slot forward until we find either the SAME key already
//     present (do nothing -- first taxon to claim a k-mer wins; see build note)
//     or an EMPTY slot (claim it). This "first writer wins" rule is the teaching
//     stand-in for Kraken2's lowest-common-ancestor assignment (THEORY sec
//     "real world"): a k-mer shared by two taxa keeps the taxon that inserted
//     first instead of being mapped to their common ancestor.
//   Returns true if a NEW distinct k-mer was inserted (so the caller can count).
// ---------------------------------------------------------------------------
static bool insert_kmer(RefDatabase& db, uint64_t key, uint32_t taxon) {
    uint64_t mask = db.capacity - 1ULL;
    uint64_t slot = hash_kmer(key) & mask;            // same home slot as look-up
    for (uint64_t step = 0; step < db.capacity; ++step) {
        if (db.taxa[slot] == TAXON_EMPTY) {           // free slot -> claim it
            db.keys[slot] = key;
            db.taxa[slot] = taxon;
            return true;                              // a new distinct k-mer
        }
        if (db.keys[slot] == key) return false;       // already present -> keep it
        slot = (slot + 1ULL) & mask;                  // linear step, wrap around
    }
    return false;  // table full (never happens: we size capacity generously)
}

// ---------------------------------------------------------------------------
// build_database: slide a k-mer window over each reference sequence and insert
// its canonical k-mers under that sequence's taxon id. Taxon id (t+1) belongs to
// seqs[t] / names[t] (id 0 is reserved for "unclassified").
// ---------------------------------------------------------------------------
void build_database(const std::vector<std::string>& seqs,
                    const std::vector<std::string>& names,
                    RefDatabase& db) {
    // First, count the total k-mer windows so we can size the table once. The
    // count is an UPPER bound (k-mers may repeat); sizing for it keeps the load
    // factor comfortably low even before dedup.
    uint64_t total_windows = 0;
    for (const std::string& s : seqs) {
        if (static_cast<int>(s.size()) >= KMER_K)
            total_windows += s.size() - KMER_K + 1;
    }
    // capacity >= 2 * total so load factor < 0.5; floor at 16 for tiny inputs.
    db.capacity = next_pow2_at_least(total_windows * 2 + 16);
    db.keys.assign(db.capacity, 0ULL);
    db.taxa.assign(db.capacity, TAXON_EMPTY);
    db.num_kmers = 0;

    // names[0] = "unclassified" so names[id] indexes directly by taxon id.
    db.names.clear();
    db.names.push_back("unclassified");
    for (const std::string& nm : names) db.names.push_back(nm);

    // Insert every canonical k-mer of every reference under its taxon id.
    for (std::size_t t = 0; t < seqs.size(); ++t) {
        const std::string& s = seqs[t];
        uint32_t taxon = static_cast<uint32_t>(t + 1);   // ids start at 1
        uint64_t fwd = 0;
        int valid = 0;
        for (std::size_t p = 0; p < s.size(); ++p) {
            uint8_t code = base_to_2bit(s[p]);
            if (code == 0xFF) { valid = 0; fwd = 0; continue; }   // N breaks window
            fwd = ((fwd << 2) | code) & KMER_MASK;
            if (valid < KMER_K) ++valid;
            if (valid < KMER_K) continue;                         // window not full
            uint64_t key = canonical_kmer(fwd);                   // strand-agnostic
            if (insert_kmer(db, key, taxon)) ++db.num_kmers;
        }
    }
}

// ---------------------------------------------------------------------------
// load_problem: parse the dataset. Format (see data/README.md), '#'-comments and
// blank lines ignored:
//   T <num_taxa>
//   <num_taxa> lines:  REF <taxon_name> <sequence>
//   R <num_reads>
//   <num_reads> lines: READ <true_taxon_id> <sequence>
// The reference lines build the table; the read lines become the ReadSet.
// ---------------------------------------------------------------------------
void load_problem(const std::string& path, RefDatabase& db, ReadSet& reads) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open dataset file: " + path);

    std::vector<std::string> ref_names, ref_seqs;
    int expect_taxa = -1, expect_reads = -1;
    std::string line;

    // read_nonblank: pull the next meaningful line (skip blanks and '#' comments).
    auto read_nonblank = [&](std::string& out) -> bool {
        while (std::getline(in, out)) {
            // trim leading whitespace to test for comment/blank robustly
            std::size_t a = out.find_first_not_of(" \t\r\n");
            if (a == std::string::npos) continue;        // blank line
            if (out[a] == '#') continue;                 // comment line
            return true;
        }
        return false;
    };

    // --- Reference block: "T <n>" then n "REF name seq" lines ----------------
    if (!read_nonblank(line)) throw std::runtime_error("empty dataset: " + path);
    {
        std::istringstream ss(line);
        std::string tag; ss >> tag >> expect_taxa;
        if (tag != "T" || expect_taxa <= 0)
            throw std::runtime_error("expected 'T <num_taxa>' header in " + path);
    }
    for (int i = 0; i < expect_taxa; ++i) {
        if (!read_nonblank(line)) throw std::runtime_error("missing REF line in " + path);
        std::istringstream ss(line);
        std::string tag, name, seq; ss >> tag >> name >> seq;
        if (tag != "REF" || name.empty() || seq.empty())
            throw std::runtime_error("malformed REF line in " + path + ": " + line);
        ref_names.push_back(name);
        ref_seqs.push_back(seq);
    }

    // --- Read block: "R <n>" then n "READ truth seq" lines -------------------
    if (!read_nonblank(line)) throw std::runtime_error("missing 'R <num_reads>' in " + path);
    {
        std::istringstream ss(line);
        std::string tag; ss >> tag >> expect_reads;
        if (tag != "R" || expect_reads < 0)
            throw std::runtime_error("expected 'R <num_reads>' header in " + path);
    }
    reads.bases.clear(); reads.offset.clear(); reads.length.clear(); reads.truth.clear();
    for (int i = 0; i < expect_reads; ++i) {
        if (!read_nonblank(line)) throw std::runtime_error("missing READ line in " + path);
        std::istringstream ss(line);
        std::string tag, seq; uint32_t truth = 0; ss >> tag >> truth >> seq;
        if (tag != "READ" || seq.empty())
            throw std::runtime_error("malformed READ line in " + path + ": " + line);
        // Append this read to the flat buffer and record its slice.
        reads.offset.push_back(static_cast<int>(reads.bases.size()));
        reads.length.push_back(static_cast<int>(seq.size()));
        reads.truth.push_back(truth);
        reads.bases.insert(reads.bases.end(), seq.begin(), seq.end());
    }
    reads.n_reads = static_cast<int>(reads.offset.size());

    // Build the hash table from the reference sequences.
    build_database(ref_seqs, ref_names, db);
}

// ---------------------------------------------------------------------------
// classify_cpu: the trusted serial baseline. One read at a time, call the SHARED
// classify_read() core (kmer_core.h) -- exactly what each GPU thread does -- so
// the outputs are integer-identical to the GPU's (verification tolerance = 0).
// ---------------------------------------------------------------------------
void classify_cpu(const ReadSet& reads, const RefDatabase& db,
                  std::vector<uint32_t>& out) {
    out.assign(static_cast<std::size_t>(reads.n_reads), TAXON_UNCLASSIFIED);
    int votes[MAX_TAXA];   // per-read vote histogram (reused; classify_read zeros it)
    for (int i = 0; i < reads.n_reads; ++i) {
        const char* read = reads.bases.data() + reads.offset[i];
        out[static_cast<std::size_t>(i)] =
            classify_read(read, reads.length[i],
                          db.keys.data(), db.taxa.data(), db.capacity, votes);
    }
}
