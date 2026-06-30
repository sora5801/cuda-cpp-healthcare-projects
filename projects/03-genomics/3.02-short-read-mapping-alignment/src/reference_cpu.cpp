// ===========================================================================
// src/reference_cpu.cpp  --  Loader, index builder, and the trusted CPU mapper
// ---------------------------------------------------------------------------
// Project 3.2 : Short-Read Mapping / Alignment
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. It is written to be
//   OBVIOUSLY correct -- plain serial loops, no parallelism, no cleverness -- so
//   that when the GPU and CPU agree on every read's (pos, score), we believe the
//   GPU. It contains three pieces:
//
//     (1) load_problem()  : parse the tiny text sample (data/README.md format).
//     (2) build_index()   : sort the reference's k-mers so a seed is found by
//                           binary search (the same index the GPU uses).
//     (3) map_reads_cpu() : the trusted seed-and-extend loop over all reads.
//
//   Compiled by the host C++ compiler only (no CUDA here). The per-window math
//   it calls (kmer_code, kmer_equal_range, score_window) lives in the shared
//   `__host__ __device__` header reference_cpu.h, so the kernel runs IDENTICAL
//   arithmetic -> exact verification.
//
// READ THIS AFTER: reference_cpu.h. Compare against kernels.cu (the GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <algorithm>   // std::stable_sort
#include <fstream>     // std::ifstream
#include <numeric>     // std::iota
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// encode: map one nucleotide character to its 0..3 base code, or throw. We
// accept upper- and lower-case; anything else (N, gaps, junk) is rejected so the
// demo fails loudly rather than silently mis-encoding ambiguous bases.
// ---------------------------------------------------------------------------
static uint8_t encode(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default:
            throw std::runtime_error(
                std::string("non-ACGT character in sequence: '") + c + "'");
    }
}

// Encode one text line into base codes, tolerating stray carriage returns /
// spaces / tabs (so Windows CRLF files and lightly-formatted samples both load).
static std::vector<uint8_t> encode_line(const std::string& s) {
    std::vector<uint8_t> v;
    v.reserve(s.size());
    for (char c : s) {
        if (c == '\r' || c == ' ' || c == '\t') continue;
        v.push_back(encode(c));
    }
    return v;
}

// ---------------------------------------------------------------------------
// load_problem: the sample format is dead simple (see data/README.md):
//   line 1            : the reference sequence
//   each later line   : one read (all reads the same length)
// Blank lines are skipped. We validate that the reference is long enough to hold
// at least one k-mer and that every read has the same length.
// ---------------------------------------------------------------------------
MappingProblem load_problem(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open problem file: " + path);

    MappingProblem prob;
    std::string line;
    bool have_ref = false;

    while (std::getline(in, line)) {
        std::vector<uint8_t> enc = encode_line(line);
        if (enc.empty()) continue;            // skip blank / whitespace-only lines

        if (!have_ref) {
            // First non-empty line is the reference.
            prob.ref = std::move(enc);
            prob.ref_len = static_cast<int>(prob.ref.size());
            have_ref = true;
        } else {
            // Every subsequent line is one read. Enforce uniform length.
            if (prob.read_len == 0) {
                prob.read_len = static_cast<int>(enc.size());
            } else if (static_cast<int>(enc.size()) != prob.read_len) {
                throw std::runtime_error(
                    "all reads must have the same length (this teaching version "
                    "uses a uniform read_len); see data/README.md");
            }
            prob.reads.insert(prob.reads.end(), enc.begin(), enc.end());
            ++prob.n_reads;
        }
    }

    if (!have_ref)                     throw std::runtime_error("empty reference in " + path);
    if (prob.ref_len < SEED_K)         throw std::runtime_error("reference shorter than SEED_K");
    if (prob.n_reads == 0)             throw std::runtime_error("no reads in " + path);
    if (prob.read_len < SEED_K)        throw std::runtime_error("reads shorter than SEED_K");
    return prob;
}

// ---------------------------------------------------------------------------
// build_index: enumerate every length-SEED_K window of the reference, record
// (k-mer code, offset), then sort by code (offset breaks ties for determinism).
// The result is the KmerIndex that BOTH the CPU reference and the GPU kernel use
// to seed -- identical input -> identical candidate positions on both sides.
//   Complexity: O(n_kmers log n_kmers) once, then reused for all reads.
// ---------------------------------------------------------------------------
KmerIndex build_index(const MappingProblem& prob) {
    KmerIndex idx;
    idx.n_kmers = prob.ref_len - SEED_K + 1;   // number of length-K windows

    // A permutation we will sort; perm[i] is a reference offset. Sorting the
    // permutation (instead of pairs) keeps the code/offset arrays in lockstep.
    std::vector<int> perm(static_cast<std::size_t>(idx.n_kmers));
    std::iota(perm.begin(), perm.end(), 0);    // 0,1,...,n_kmers-1

    // Precompute each window's k-mer code once (so the sort comparator is cheap).
    std::vector<uint64_t> code(static_cast<std::size_t>(idx.n_kmers));
    for (int w = 0; w < idx.n_kmers; ++w) {
        code[static_cast<std::size_t>(w)] = kmer_code(prob.ref.data(), w);
    }

    // stable_sort keyed by (code, then offset): equal k-mers end up contiguous
    // AND in ascending-offset order -> the per-read tie-break (lowest offset
    // wins) is deterministic and matches the GPU's identical ordering.
    std::stable_sort(perm.begin(), perm.end(), [&](int a, int b) {
        if (code[static_cast<std::size_t>(a)] != code[static_cast<std::size_t>(b)])
            return code[static_cast<std::size_t>(a)] < code[static_cast<std::size_t>(b)];
        return a < b;   // equal codes: lower reference offset first
    });

    // Materialize the sorted arrays the GPU will consume.
    idx.sorted_codes.resize(static_cast<std::size_t>(idx.n_kmers));
    idx.sorted_offsets.resize(static_cast<std::size_t>(idx.n_kmers));
    for (int i = 0; i < idx.n_kmers; ++i) {
        const int off = perm[static_cast<std::size_t>(i)];
        idx.sorted_codes[static_cast<std::size_t>(i)]   = code[static_cast<std::size_t>(off)];
        idx.sorted_offsets[static_cast<std::size_t>(i)] = off;
    }
    return idx;
}

// ---------------------------------------------------------------------------
// map_one: the per-read seed-and-extend logic, factored out so the CPU loop here
// and (conceptually) the GPU thread in kernels.cu do the SAME steps in the SAME
// order. Returns the best MapResult for one read.
//
// Tie-breaking (must match the GPU EXACTLY for == verification):
//   pick the HIGHEST score; on a tie, the LOWEST reference offset. We scan
//   candidate offsets in ascending order (the index is sorted by offset within
//   an equal-code run) and only REPLACE the best on a strictly-higher score, so
//   the first (lowest) offset achieving the max wins.
// ---------------------------------------------------------------------------
static MapResult map_one(const MappingProblem& prob, const KmerIndex& index,
                         const uint8_t* read) {
    MapResult best;          // pos = NO_HIT, score = 0 initially
    best.score = -2000000;   // below any real or off-end score, so first real
                             // candidate always wins; reset to a clean miss below.

    // SEED: this read's leading k-mer (bases [0, SEED_K)). seed_pos_in_read = 0,
    // so a reference window starting at offset `o` means the read's base 0 lines
    // up with reference position `o` directly.
    const uint64_t qcode = kmer_code(read, 0);
    int lo = 0, hi = 0;
    kmer_equal_range(index.sorted_codes.data(), index.n_kmers, qcode, &lo, &hi);

    // EXTEND: score the whole read at each candidate offset; keep the best.
    for (int i = lo; i < hi; ++i) {
        const int pos = index.sorted_offsets[static_cast<std::size_t>(i)];
        int mism = 0;
        const int sc = score_window(prob.ref.data(), prob.ref_len,
                                    read, prob.read_len, pos, &mism);
        if (sc > best.score) {       // strict > keeps the lowest-offset tie winner
            best.score = sc;
            best.pos   = pos;
            best.mism  = mism;
        }
    }

    // If the seed hit nothing (lo == hi), best.pos is still NO_HIT; normalize the
    // score to 0 so a miss reports cleanly rather than the -2000000 sentinel.
    if (best.pos == NO_HIT) { best.score = 0; best.mism = 0; }
    return best;
}

// ---------------------------------------------------------------------------
// map_reads_cpu: run map_one() for every read, serially. O(R * C * L) where C is
// the average number of candidate positions per seed and L is the read length --
// the GPU does the exact same total work but spreads the R reads across threads.
// ---------------------------------------------------------------------------
void map_reads_cpu(const MappingProblem& prob, const KmerIndex& index,
                   std::vector<MapResult>& results) {
    results.assign(static_cast<std::size_t>(prob.n_reads), MapResult{});
    for (int r = 0; r < prob.n_reads; ++r) {
        // Pointer to read r's row in the flat [n_reads * read_len] buffer.
        const uint8_t* read = prob.reads.data()
                            + static_cast<std::size_t>(r) * prob.read_len;
        results[static_cast<std::size_t>(r)] = map_one(prob, index, read);
    }
}
