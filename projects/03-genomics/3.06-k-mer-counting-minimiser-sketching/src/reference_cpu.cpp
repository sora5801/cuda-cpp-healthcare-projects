// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial CPU reference (the verification oracle)
// ---------------------------------------------------------------------------
// Project 3.6 : k-mer Counting & Minimiser Sketching
//
// Compiled by the host compiler ONLY (no CUDA here). Every per-k-mer computation
// (encode, canonicalise, hash, minimiser) is delegated to the shared inline
// functions in kmer.h, so this reference and the GPU kernels do byte-identical
// math. The output of count_kmers_cpu / sketch_cpu is the GROUND TRUTH that
// main.cu checks the GPU result against.
//
// READ THIS AFTER: kmer.h, reference_cpu.h.
// ===========================================================================
#include "reference_cpu.h"

#include <algorithm>     // std::sort, std::unique, std::binary_search
#include <fstream>       // std::ifstream
#include <map>           // std::map (ordered -> sorted histogram for free)
#include <sstream>       // std::istringstream
#include <stdexcept>     // std::runtime_error
#include <string>

// ---------------------------------------------------------------------------
// append_read: push one read's characters onto a ReadSet's flat buffer and
//   record its end offset. Centralised so load_reads stays readable.
// ---------------------------------------------------------------------------
static void append_read(ReadSet& rs, const std::string& seq) {
    rs.bases.insert(rs.bases.end(), seq.begin(), seq.end());
    rs.offsets.push_back(rs.bases.size());   // new end == next read's start
    rs.num_reads += 1;
}

// ---------------------------------------------------------------------------
// load_reads: parse the tiny two-set sample format (see data/README.md).
//   Header line: "k w s". Then ">A" + reads, then ">B" + reads. We keep the
//   parser deliberately small and strict so malformed input fails loudly.
// ---------------------------------------------------------------------------
ReadSet load_reads(const std::string& path, ReadSet& setB, int& sketch_s) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open dataset file: " + path);

    ReadSet setA;
    int k = 0, w = 0, s = 0;

    // -- header: k w s -----------------------------------------------------
    {
        std::string line;
        if (!std::getline(in, line)) throw std::runtime_error("empty dataset: " + path);
        std::istringstream hs(line);
        if (!(hs >> k >> w >> s)) throw std::runtime_error("bad header (expected 'k w s'): " + path);
        if (k < 1 || k > KMER_MAX_K) throw std::runtime_error("k out of range [1,31]");
        if (w < 1)                   throw std::runtime_error("w must be >= 1");
        if (s < 1)                   throw std::runtime_error("s must be >= 1");
    }

    // Both sets share the same k and w; offsets start with a single 0 sentinel.
    setA.k = setB.k = k;
    setA.w = setB.w = w;
    setA.offsets.push_back(0);
    setB.offsets.push_back(0);
    sketch_s = s;

    // -- body: section headers ">A"/">B" then read lines -------------------
    ReadSet* cur = nullptr;       // which set the following reads belong to
    std::string line;
    while (std::getline(in, line)) {
        // Trim a trailing '\r' so Windows-CRLF files parse the same as LF.
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;                       // skip blank lines
        if (line[0] == '>') {                             // section header
            char tag = (line.size() > 1) ? line[1] : '?';
            cur = (tag == 'A') ? &setA : (tag == 'B') ? &setB : nullptr;
            if (!cur) throw std::runtime_error("unknown section header: " + line);
            continue;
        }
        if (!cur) throw std::runtime_error("read line before any >A/>B section");
        append_read(*cur, line);
    }
    if (setA.num_reads == 0 || setB.num_reads == 0)
        throw std::runtime_error("both sets A and B must contain at least one read");
    return setA;
}

// ---------------------------------------------------------------------------
// count_kmers_cpu: tally canonical k-mers across all reads.
//   We slide a length-k window over every read; for each valid window we compute
//   the canonical k-mer (kmer.h) and bump its count. A std::map<key,count> keeps
//   the histogram ORDERED BY KEY, so copying it out yields a deterministic,
//   ascending-by-key vector -- the same order the GPU path sorts into. Windows
//   containing an invalid base (e.g. 'N') are skipped, exactly like real tools.
//
//   Complexity: O(total_bases * k) here (O(k) re-encode per window). A production
//   counter rolls the 2-bit window in O(1)/step; we keep the simple form for
//   clarity and note the rolling optimisation in THEORY.md.
// ---------------------------------------------------------------------------
std::vector<KmerCount> count_kmers_cpu(const ReadSet& rs) {
    std::map<uint64_t, unsigned int> hist;   // ordered map -> sorted output
    const int k = rs.k;
    for (int r = 0; r < rs.num_reads; ++r) {
        const std::size_t start = rs.offsets[r];
        const std::size_t len   = rs.read_len(r);
        if (len < (std::size_t)k) continue;              // read too short for any k-mer
        const char* seq = rs.bases.data() + start;       // this read's first base
        const std::size_t n_windows = len - k + 1;       // number of k-mer start positions
        for (std::size_t p = 0; p < n_windows; ++p) {
            uint64_t canon, hash;
            if (canonical_hash_at(seq, p, k, &canon, &hash))
                hist[canon] += 1;                        // count this canonical k-mer
        }
    }
    std::vector<KmerCount> out;
    out.reserve(hist.size());
    for (const auto& kv : hist) out.push_back({kv.first, kv.second});
    return out;                                          // already sorted by key
}

// ---------------------------------------------------------------------------
// sketch_cpu: bottom-s MinHash sketch from per-read minimisers.
//   STEP 1 (minimisers): for each read, slide a window of w consecutive k-mers
//   and select the k-mer whose HASH is smallest in that window; collect those
//   minimiser hashes. (Minimisers compress ~w-fold yet two overlapping reads pick
//   the SAME minimiser in shared regions -- the property that makes them useful.)
//   STEP 2 (bottom-s): from all minimiser hashes keep the s SMALLEST DISTINCT
//   values. That bottom-s set is a MinHash sketch: its overlap with another set's
//   sketch is an unbiased estimator of the Jaccard similarity.
//
//   We collect minimiser hashes, then sort/dedup/truncate to s. This matches the
//   GPU path, which produces the same global distinct-hash set and truncates
//   identically.
// ---------------------------------------------------------------------------
Sketch sketch_cpu(const ReadSet& rs, int s) {
    const int k = rs.k;
    const int w = rs.w;
    std::vector<uint64_t> mins;   // all minimiser hashes (with duplicates), pre-dedup

    for (int r = 0; r < rs.num_reads; ++r) {
        const std::size_t start = rs.offsets[r];
        const std::size_t len   = rs.read_len(r);
        if (len < (std::size_t)k) continue;
        const char* seq = rs.bases.data() + start;
        const std::size_t n_windows = len - k + 1;       // # of k-mers in this read

        // Precompute the hash (or "invalid") of every k-mer position in the read.
        // KMER_EMPTY (all-ones) is "no valid k-mer here" so it never wins a min.
        std::vector<uint64_t> kh(n_windows, KMER_EMPTY);
        for (std::size_t p = 0; p < n_windows; ++p) {
            uint64_t canon, hash;
            if (canonical_hash_at(seq, p, k, &canon, &hash)) kh[p] = hash;
        }

        // Slide a window of w consecutive k-mers; emit the minimum hash in each.
        if (n_windows < (std::size_t)w) continue;        // read too short for a full window
        const std::size_t n_min_windows = n_windows - w + 1;
        for (std::size_t i = 0; i < n_min_windows; ++i) {
            uint64_t best = KMER_EMPTY;
            for (int j = 0; j < w; ++j) best = (kh[i + j] < best) ? kh[i + j] : best;
            if (best != KMER_EMPTY) mins.push_back(best); // valid minimiser found
        }
    }

    // Bottom-s: sort, dedup, truncate to s smallest distinct hashes.
    std::sort(mins.begin(), mins.end());
    mins.erase(std::unique(mins.begin(), mins.end()), mins.end());
    if ((int)mins.size() > s) mins.resize(s);
    Sketch out;
    out.hashes = std::move(mins);
    return out;
}

// ---------------------------------------------------------------------------
// jaccard_estimate: MinHash estimator of Jaccard similarity from two sketches.
//   Merge the two sorted bottom-s sketches, take the s' = min(s, |merged distinct|)
//   smallest DISTINCT hashes of the union, and count how many of those appear in
//   BOTH sketches. estimate = shared / s'. This is the bottom-s MinHash estimator
//   used by Mash: it converges to the true Jaccard as s grows. Returns 0 if there
//   is nothing to compare.
// ---------------------------------------------------------------------------
double jaccard_estimate(const Sketch& a, const Sketch& b, int s) {
    // Merge-distinct of the two ascending lists, capped at s entries.
    std::vector<uint64_t> merged;
    merged.reserve(a.hashes.size() + b.hashes.size());
    std::size_t i = 0, j = 0;
    while (merged.size() < (std::size_t)s &&
           (i < a.hashes.size() || j < b.hashes.size())) {
        uint64_t next;
        if (j >= b.hashes.size())                next = a.hashes[i++];
        else if (i >= a.hashes.size())           next = b.hashes[j++];
        else if (a.hashes[i] < b.hashes[j])      next = a.hashes[i++];
        else if (b.hashes[j] < a.hashes[i])      next = b.hashes[j++];
        else { next = a.hashes[i]; ++i; ++j; }   // equal -> take once
        if (merged.empty() || merged.back() != next) merged.push_back(next);
    }
    if (merged.empty()) return 0.0;

    // Count how many of these "union bottom-s'" hashes are present in BOTH.
    // Membership test via binary search on the (sorted) sketches.
    int shared = 0;
    for (uint64_t h : merged) {
        bool inA = std::binary_search(a.hashes.begin(), a.hashes.end(), h);
        bool inB = std::binary_search(b.hashes.begin(), b.hashes.end(), h);
        if (inA && inB) ++shared;
    }
    return static_cast<double>(shared) / static_cast<double>(merged.size());
}

// ---------------------------------------------------------------------------
// kmer_to_string: decode a packed canonical k-mer to an ACGT string.
//   Bases were packed MSB-first (kmer.h), so we peel from the top down.
// ---------------------------------------------------------------------------
std::string kmer_to_string(uint64_t code, int k) {
    static const char LUT[4] = {'A', 'C', 'G', 'T'};
    std::string s(k, 'A');
    for (int i = 0; i < k; ++i) {
        int shift = 2 * (k - 1 - i);                     // bits for position i (MSB-first)
        s[i] = LUT[(code >> shift) & 3ull];
    }
    return s;
}
