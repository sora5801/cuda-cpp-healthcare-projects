// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference API (the teaching baseline)
// ---------------------------------------------------------------------------
// Project 3.6 : k-mer Counting & Minimiser Sketching
//
// This header declares:
//   * ReadSet      -- how a set of DNA reads is stored in memory (GPU-friendly
//                     flat layout: one big char buffer + per-read offsets).
//   * KmerCount    -- one (canonical k-mer, count) pair, the unit of a histogram.
//   * Sketch       -- a bottom-s MinHash sketch (the s smallest distinct hashes).
//   * the CPU reference functions that main.cu runs alongside the GPU and then
//     compares against. Each has a GPU twin in kernels.cu computing the same math
//     via the shared inline functions in kmer.h.
//
// The CPU reference exists for two reasons (CLAUDE.md section 5): it is the
// legible baseline that makes the GPU speed-up meaningful, and it is the oracle
// the GPU result is verified against (exactly, here -- both sides count the same
// canonical k-mers and sort the same way).
//
// READ THIS AFTER: kmer.h. READ BEFORE: reference_cpu.cpp, main.cu, kernels.cuh.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "kmer.h"   // shared encode/canonical/hash/minimiser primitives

// ---------------------------------------------------------------------------
// ReadSet: a set of DNA reads in a flat, GPU-friendly layout.
//   Instead of a vector<string> (pointer-chasing, not copyable to the device),
//   we concatenate every read into one contiguous `bases` buffer and remember
//   where each read starts in `offsets`. Read r occupies
//   bases[offsets[r] .. offsets[r+1]) -- the classic CSR-style "ragged array".
//   This single buffer is what we cudaMemcpy to the GPU; a thread locates its
//   read by binary-searching offsets (we build an explicit per-position map in
//   main.cu so each thread does O(1) lookup instead).
// ---------------------------------------------------------------------------
struct ReadSet {
    int                       k = 0;        // k-mer length for this dataset
    int                       w = 0;        // minimiser window (in k-mers)
    int                       num_reads = 0;
    std::vector<char>         bases;        // concatenated read characters (no separators)
    std::vector<std::size_t>  offsets;      // size num_reads+1; offsets[r]..offsets[r+1]

    // Length (in characters) of read r.
    std::size_t read_len(int r) const { return offsets[r + 1] - offsets[r]; }
};

// One entry of a k-mer histogram: a canonical k-mer and how many times it occurs.
struct KmerCount {
    uint64_t     key;     // packed canonical k-mer (2 bits/base)
    unsigned int count;   // occurrences across all reads
};

// A bottom-s MinHash sketch: the `s` SMALLEST DISTINCT minimiser hashes seen in a
// read set, kept sorted ascending. Two sketches' Jaccard estimate is the fraction
// of the merged bottom-s hashes that appear in both (see jaccard_estimate).
struct Sketch {
    std::vector<uint64_t> hashes;   // sorted ascending, distinct, size <= s
};

// === Loading ===============================================================
// Parse the tiny sample format (see data/README.md):
//   line 1:  "k w s"   (k-mer length, minimiser window in k-mers, sketch size)
//   then exactly two labelled sections, ">A" then ">B", each followed by one or
//   more read lines (ACGT/N). Two labelled sets let us compute a Jaccard distance
//   between them. The function returns set A, fills `setB`, and reports `sketch_s`.
//   Throws std::runtime_error on malformed input.
ReadSet load_reads(const std::string& path, ReadSet& setB, int& sketch_s);

// === k-mer counting (CPU reference) ========================================
// Count every canonical k-mer across all reads of `rs`, returning the histogram
// SORTED ASCENDING BY KEY (so the output is deterministic regardless of order).
std::vector<KmerCount> count_kmers_cpu(const ReadSet& rs);

// === Minimiser sketching (CPU reference) ===================================
// Build a bottom-s MinHash sketch from the minimisers of every read in `rs`.
// `s` is the sketch size. Returns hashes sorted ascending, distinct.
Sketch sketch_cpu(const ReadSet& rs, int s);

// === MinHash Jaccard ========================================================
// Estimate the Jaccard similarity J(A,B) = |A intersect B| / |A union B| from two
// bottom-s sketches, the standard MinHash estimator: merge the two sorted hash
// lists, take the s smallest distinct values, and report (# in BOTH) / s'.
// Returns the estimate in [0,1]. `s` is the nominal sketch size.
double jaccard_estimate(const Sketch& a, const Sketch& b, int s);

// === Small shared helpers (used by main + reference) =======================
// Decode a packed canonical k-mer back to an ACGT string (for human-readable
// output). `k` is the k-mer length.
std::string kmer_to_string(uint64_t code, int k);
