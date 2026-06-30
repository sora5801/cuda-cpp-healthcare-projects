// ===========================================================================
// src/reference_cpu.h  --  Data model, DB builder, and CPU reference
// ---------------------------------------------------------------------------
// Project 3.14 : Metagenomic Taxonomic Classification
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the host C++ compiler and must not see any
//   CUDA syntax, so the shared DATA MODEL (the reads, the reference hash table,
//   the file loader, the table builder) and the CPU reference prototype live in
//   this CUDA-free header. kernels.cuh also includes it to reuse these types --
//   nothing CUDA-specific leaks in either direction. The per-read MATH is in
//   kmer_core.h (shared __host__ __device__); this header is the host-only
//   scaffolding around it.
//
// THE DATA MODEL
//   * RefDatabase : an open-addressing hash table mapping canonical k-mer ->
//     taxon id, PLUS the list of taxon names (for the human-readable report).
//   * ReadSet     : the metagenomic sample = a flat char buffer of concatenated
//     reads (so it copies to the GPU as one contiguous array) + per-read offsets
//     and lengths + the TRUE taxon each read was simulated from (synthetic data
//     ships the ground truth so we can measure accuracy).
//
// READ THIS AFTER: kmer_core.h.  READ THIS BEFORE: reference_cpu.cpp, kernels.cuh.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "kmer_core.h"   // KMER_K, hash_kmer, classify_read, MAX_TAXA (pure C++ under host compiler)

// ---------------------------------------------------------------------------
// RefDatabase: the reference k-mer -> taxon hash table.
//   Stored as two parallel arrays (Structure-of-Arrays) so it copies to the GPU
//   as two flat buffers and so the device probe touches only the array it needs:
//     keys[slot] : canonical k-mer at that slot (valid iff taxa[slot] != 0)
//     taxa[slot] : taxon id, or 0 (TAXON_EMPTY) for an empty slot
//   capacity is a power of two so the probe can use (& (capacity-1)) for modulo.
//   names[id] is the human-readable name of taxon `id` (names[0] = "unclassified").
// ---------------------------------------------------------------------------
struct RefDatabase {
    std::vector<uint64_t>    keys;     // [capacity] canonical k-mers
    std::vector<uint32_t>    taxa;     // [capacity] taxon ids (0 = empty)
    uint64_t                 capacity = 0;   // power of two
    std::vector<std::string> names;    // [num_taxa] taxon id -> name
    uint64_t                 num_kmers = 0;  // distinct k-mers actually inserted
};

// ---------------------------------------------------------------------------
// ReadSet: the metagenomic sample to classify.
//   bases   : all reads concatenated into ONE char buffer (no separators). A
//             single contiguous array is the GPU-friendly layout: one H2D copy,
//             coalesced access, no array-of-pointers chasing.
//   offset  : [n_reads] start index of read i within `bases`.
//   length  : [n_reads] length of read i in bases.
//   truth   : [n_reads] the taxon id read i was SIMULATED from (synthetic ground
//             truth). Used only to report accuracy, never by the classifier.
// ---------------------------------------------------------------------------
struct ReadSet {
    std::vector<char>     bases;    // concatenated read characters
    std::vector<int>      offset;   // [n_reads] start of each read in `bases`
    std::vector<int>      length;   // [n_reads] length of each read
    std::vector<uint32_t> truth;    // [n_reads] ground-truth taxon id (synthetic)
    int n_reads = 0;
};

// ---------------------------------------------------------------------------
// load_problem: parse the tiny text dataset (format documented in data/README.md
// and produced by scripts/make_synthetic.py). It contains BOTH the reference
// genomes (to build the table) and the reads (to classify), so a single file
// makes the demo fully self-contained and offline.
//   Returns the built RefDatabase and the ReadSet by reference. Throws
//   std::runtime_error on a missing/malformed file.
// ---------------------------------------------------------------------------
void load_problem(const std::string& path, RefDatabase& db, ReadSet& reads);

// ---------------------------------------------------------------------------
// build_database: turn a set of named reference sequences into the k-mer hash
// table. For each reference sequence (one per taxon), slide a k-mer window,
// canonicalize, and INSERT (k-mer -> taxon) via linear probing. If two taxa
// share a k-mer, the first insertion wins and later ones are dropped at that
// slot -- a simplification of Kraken2's lowest-common-ancestor rule, explained
// in THEORY. Called by load_problem; exposed for testing.
//   seqs[t]  : the reference sequence for taxon id (t+1)  (ASCII A/C/G/T)
//   names[t] : the name of taxon id (t+1)
//   Builds db.keys/taxa/capacity/names sized to keep the load factor < ~0.5.
// ---------------------------------------------------------------------------
void build_database(const std::vector<std::string>& seqs,
                    const std::vector<std::string>& names,
                    RefDatabase& db);

// ---------------------------------------------------------------------------
// classify_cpu: the trusted serial baseline. For each read, call the SHARED
// classify_read() from kmer_core.h (the exact function the GPU kernel calls),
// and write the winning taxon id into out[i]. Because both sides call the same
// __host__ __device__ core, the CPU and GPU outputs are integer-identical -- the
// verification tolerance is therefore ZERO. out is resized to reads.n_reads.
// ---------------------------------------------------------------------------
void classify_cpu(const ReadSet& reads, const RefDatabase& db,
                  std::vector<uint32_t>& out);
