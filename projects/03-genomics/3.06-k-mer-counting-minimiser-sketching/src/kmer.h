// ===========================================================================
// src/kmer.h  --  Shared (host + device) k-mer primitives: encode, canonical,
//                 hash, minimiser. The ONE place the "per-element physics" lives.
// ---------------------------------------------------------------------------
// Project 3.6 : k-mer Counting & Minimiser Sketching   (see ../THEORY.md)
//
// WHAT THIS PROJECT COMPUTES
//   Given a set of DNA reads (strings over {A,C,G,T}), we extract every length-k
//   substring (a "k-mer"), fold each to its CANONICAL form (the lexicographically
//   smaller of the k-mer and its reverse complement, so a fragment counts the
//   same regardless of which strand it was read from), and then do two things:
//
//     (1) k-MER COUNTING  -- tally how many times each distinct canonical k-mer
//         occurs. The histogram of counts drives genome-size estimation, error
//         detection, assembly, and metagenomics.
//
//     (2) MINIMISER SKETCHING -- from each read pick, per sliding window of w
//         consecutive k-mers, the one k-mer whose hash is smallest (the
//         "minimiser"). The set of minimisers is a small, strand-symmetric
//         SUBSET of the k-mers that two related sequences tend to SHARE. From the
//         globally smallest `s` distinct minimiser hashes (a bottom-s MinHash
//         sketch) we estimate the JACCARD similarity between two read sets -- the
//         core of Mash-style "are these the same species?" distance.
//
// WHY THESE FUNCTIONS LIVE IN ONE HEADER  (PATTERNS.md section 2)
//   Both the CPU reference (reference_cpu.cpp, compiled by the host compiler) and
//   the GPU kernels (kernels.cu, compiled by nvcc) call EXACTLY these inline
//   functions. Sharing the math is what makes the GPU result match the CPU result
//   BIT-FOR-BIT -- our verification is then exact (max key/count difference == 0),
//   not "close enough". Keep CUDA-only constructs (no __global__, no kernel
//   launches) out of this header so the host compiler can include it too.
//
//   The KMER_HD macro expands to `__host__ __device__` under nvcc and to nothing
//   under the host compiler (which has never heard of those decorators).
//
// 2-BIT ENCODING
//   A nucleotide needs only 2 bits: A=00, C=01, G=10, T=11. A k-mer (k <= 31)
//   therefore packs into a single 64-bit unsigned integer -- the natural "word"
//   the GPU sorts, hashes, and atomically counts. The reverse complement is then
//   a few bit tricks rather than a string reversal (see kmer_revcomp).
//
// READ THIS BEFORE: kernels.cuh, kernels.cu, reference_cpu.cpp.
// ===========================================================================
#pragma once

#include <cstdint>   // uint8_t, uint32_t, uint64_t
#include <cstddef>   // std::size_t

// --- host/device portability macro (PATTERNS.md section 2) -----------------
#ifdef __CUDACC__
#define KMER_HD __host__ __device__
#else
#define KMER_HD
#endif

// Maximum k we support: a k-mer is packed 2 bits/base into a 64-bit word, so
// 2*k <= 64 => k <= 31. We also need a sentinel "empty" key for the open-
// addressing hash table that no real k-mer can collide with; we reserve the
// all-ones value 0xFFFF...F for that (it would only ever be a real k-mer of all
// 'T's at k==32, which we forbid). See KMER_EMPTY below.
static const int      KMER_MAX_K  = 31;
static const uint64_t KMER_EMPTY  = ~0ull;   // 0xFFFFFFFFFFFFFFFF = "slot is free"

// ---------------------------------------------------------------------------
// base_code: map an ASCII nucleotide to its 2-bit code, or 0xFF if invalid.
//   A/a -> 0, C/c -> 1, G/g -> 2, T/t -> 3. Anything else (N, newline, junk)
//   returns 0xFF so the caller can SKIP k-mers spanning an ambiguous base --
//   exactly what real counters do (an 'N' is "unknown", not a real base).
// ---------------------------------------------------------------------------
KMER_HD inline uint8_t base_code(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default:            return 0xFF;   // invalid / ambiguous base
    }
}

// ---------------------------------------------------------------------------
// kmer_revcomp: reverse complement of a packed k-mer.
//   The complement of a 2-bit base b is (3 - b) == (~b & 3): A<->T, C<->G. The
//   reverse complement also REVERSES base order, so we peel bases off the low
//   end of `x` and stack them onto the low end of `rc` (which reverses them).
//   This is the bit-twiddling that replaces an O(k) string reversal with O(k)
//   register ops and no memory traffic.
//
//   `code` is the packed forward k-mer; `k` is its length. Returns the packed
//   reverse-complement k-mer.
// ---------------------------------------------------------------------------
KMER_HD inline uint64_t kmer_revcomp(uint64_t code, int k) {
    uint64_t rc = 0;
    for (int i = 0; i < k; ++i) {
        uint64_t base = code & 3ull;          // lowest base of the remaining word
        uint64_t comp = 3ull - base;          // complement: A<->T (0<->3), C<->G (1<->2)
        rc = (rc << 2) | comp;                // append complemented base (reverses order)
        code >>= 2;                           // drop the base we just consumed
    }
    return rc;
}

// ---------------------------------------------------------------------------
// kmer_canonical: the smaller of a k-mer and its reverse complement.
//   DNA is double-stranded; a fragment read off the '-' strand is the reverse
//   complement of the same fragment on the '+' strand. To count both as the SAME
//   k-mer we pick a canonical representative: min(forward, reverse-complement).
//   This halves the table and makes the sketch strand-symmetric.
// ---------------------------------------------------------------------------
KMER_HD inline uint64_t kmer_canonical(uint64_t code, int k) {
    uint64_t rc = kmer_revcomp(code, k);
    return (code < rc) ? code : rc;
}

// ---------------------------------------------------------------------------
// kmer_hash: a strong, fast integer mix (the finalizer of the well-known
//   SplitMix64 / Murmur3 family). We DON'T use the raw k-mer as its own hash
//   because consecutive k-mers differ in only a few bits, which would clump in a
//   modulo table and pick poor (correlated) minimisers. This avalanche mix
//   spreads every input bit across all output bits, giving (a) near-uniform hash-
//   table buckets and (b) a minimiser selection that behaves like a random hash
//   -- the property MinHash's unbiasedness relies on.
//
//   It is a pure function of the 64-bit key => identical on CPU and GPU.
// ---------------------------------------------------------------------------
KMER_HD inline uint64_t kmer_hash(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;                 // odd constant (golden-ratio fractional bits)
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull; // mix high bits down
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull; // mix again with a different multiplier
    x =  x ^ (x >> 31);                          // final fold
    return x;
}

// ---------------------------------------------------------------------------
// encode_kmer: pack the k bases starting at seq[pos] into a 64-bit word.
//   Returns true on success (and writes the packed code to *out); returns false
//   if ANY of the k bases is invalid (e.g. an 'N'), in which case no k-mer is
//   emitted at this position. Bases are packed most-significant-first so that the
//   numeric order of codes matches lexicographic order of the strings (A<C<G<T),
//   which is what canonicalisation and sorted output assume.
//
//   `seq` points at the read's characters; `pos` is the window start; `k` is the
//   k-mer length. Complexity O(k) per call (the host/GPU both just loop k times;
//   a production counter would roll the window in O(1) -- noted in THEORY).
// ---------------------------------------------------------------------------
KMER_HD inline bool encode_kmer(const char* seq, std::size_t pos, int k, uint64_t* out) {
    uint64_t code = 0;
    for (int i = 0; i < k; ++i) {
        uint8_t b = base_code(seq[pos + i]);
        if (b == 0xFF) return false;            // ambiguous base -> skip this window
        code = (code << 2) | b;                 // shift in 2 bits, MSB-first
    }
    *out = code;
    return true;
}

// ---------------------------------------------------------------------------
// canonical_hash_at: the full per-position pipeline used EVERYWHERE.
//   encode -> canonicalise -> hash. Returns true and writes both the canonical
//   k-mer (*canon) and its hash (*hash) on success; false if the window has an
//   invalid base. Centralising this guarantees the CPU loop and the GPU thread
//   run identical steps in identical order. This is the single function that
//   feeds BOTH counting (keyed by *canon) and sketching (keyed by *hash).
// ---------------------------------------------------------------------------
KMER_HD inline bool canonical_hash_at(const char* seq, std::size_t pos, int k,
                                      uint64_t* canon, uint64_t* hash) {
    uint64_t code;
    if (!encode_kmer(seq, pos, k, &code)) return false;
    uint64_t c = kmer_canonical(code, k);
    *canon = c;
    *hash  = kmer_hash(c);
    return true;
}
