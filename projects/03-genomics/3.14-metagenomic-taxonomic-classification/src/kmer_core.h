// ===========================================================================
// src/kmer_core.h  --  The shared __host__ __device__ classification core
// ---------------------------------------------------------------------------
// Project 3.14 : Metagenomic Taxonomic Classification
//
// WHY THIS HEADER EXISTS  (the single most important idea in the project)
//   The CPU reference (reference_cpu.cpp) and the GPU kernel (kernels.cu) must
//   produce IDENTICAL taxon assignments, or "verify GPU==CPU" is meaningless.
//   The way we guarantee that is the HD-macro idiom (docs/PATTERNS.md sec 2):
//   put every piece of per-read PHYSICS -- the k-mer encoding, the hash, the
//   open-addressing probe, the majority vote -- in ONE header marked
//   __host__ __device__, so the host compiler and nvcc compile the *same source*
//   into the *same arithmetic*. The output here is an INTEGER taxon id, so the
//   two sides agree EXACTLY (tolerance 0) -- the strongest verification possible.
//
//   Keep this header free of CUDA-only constructs (no __global__, no <<<>>>),
//   so the plain host compiler can include it from reference_cpu.cpp. The only
//   CUDA-aware thing is the KC_HD decorator macro, which vanishes on the host.
//
// THE PROBLEM (see ../THEORY.md for the full derivation)
//   A metagenomic sample is a soup of DNA reads from many organisms. To build an
//   abundance profile we must assign each READ to a TAXON (species/genus/...).
//   Kraken2-style classifiers do this WITHOUT alignment: they slide a window of
//   length k over the read, look each k-mer up in a precomputed hash map
//   k-mer -> taxon (built from reference genomes), and let the read's k-mers
//   VOTE for a taxon. The look-up is the bottleneck at clinical throughput
//   (millions of reads/min), and it is embarrassingly parallel -- one read per
//   GPU thread -- which is exactly why it belongs on a GPU.
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cu, kernels.cuh.
// ===========================================================================
#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// KC_HD: the host/device decorator. Under nvcc (__CUDACC__ is defined) every
// function below is compiled BOTH for the host and the device. Under the plain
// host compiler the decorators do not exist, so KC_HD expands to nothing and the
// same functions compile as ordinary C++. This is the mechanism that makes the
// CPU and GPU run byte-for-byte identical math (docs/PATTERNS.md sec 2).
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define KC_HD __host__ __device__
#else
#define KC_HD
#endif

// ---------------------------------------------------------------------------
// Compile-time parameters of the classifier. They are constants (not runtime
// args) so the inner loops unroll and the encodings are fixed across CPU/GPU.
// ---------------------------------------------------------------------------

// k-mer length. 2-bit encoding packs each base into 2 bits, so a k-mer fits in
// 2*k bits; we store it in a uint64_t, which caps k at 31. k=15 is a small,
// teaching-friendly value (Kraken2 defaults to 35-mers with minimizers; the
// trade-off -- specificity vs. table size -- is discussed in THEORY).
constexpr int      KMER_K    = 15;
// Bit mask that keeps only the low 2*KMER_K bits when we shift a new base in
// (a rolling-window trick: shift left 2, OR the new base, AND off the overflow).
constexpr uint64_t KMER_MASK = (KMER_K >= 32) ? ~0ULL : ((1ULL << (2 * KMER_K)) - 1ULL);

// A taxon id of 0 means "the hash slot is empty" (sentinel) OR, for a read,
// "unclassified". Real taxon ids therefore start at 1. This lets us use a single
// uint32_t array as the hash table with 0 = empty, no separate occupancy bitmap.
constexpr uint32_t TAXON_EMPTY        = 0u;
constexpr uint32_t TAXON_UNCLASSIFIED = 0u;

// How many distinct taxa we tally per read. The synthetic demo uses a handful of
// "species"; a cap keeps the per-thread vote histogram in registers/local memory
// instead of needing a device-side dynamic map. Reads whose k-mers hit taxa with
// id > MAX_TAXA would be ignored by the vote (documented limitation in THEORY).
constexpr int MAX_TAXA = 16;   // taxon ids 1..MAX_TAXA-1 are votable (index = id)

// ---------------------------------------------------------------------------
// base_to_2bit: map an ASCII nucleotide to its 2-bit code, or 0xFF for "not a
// base" (N, ambiguity codes, newlines). A=00, C=01, G=10, T=11 -- chosen so that
// the COMPLEMENT of a base is simply (3 - code): A<->T (0<->3), C<->G (1<->2),
// which makes reverse-complement canonicalization a couple of cheap ops.
//   Returns 0..3 for a valid base, 0xFF otherwise (caller resets the window).
// ---------------------------------------------------------------------------
KC_HD inline uint8_t base_to_2bit(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default:            return 0xFF;   // N / ambiguity / non-base -> invalid
    }
}

// ---------------------------------------------------------------------------
// reverse_complement: given a forward k-mer packed in the low 2*KMER_K bits,
// return the 2-bit encoding of its reverse complement.
//   DNA is double-stranded; a read can come from either strand, so a k-mer and
//   its reverse complement are the SAME biological feature. To make the table
//   strand-independent we always index by the CANONICAL k-mer = min(fwd, rc).
//   Algorithm: complement every 2-bit base (XOR with 0b11 == 3) and reverse the
//   order of the k bases. We reverse base-by-base; for k=15 this is 15 cheap
//   iterations -- clear over clever, and identical on host and device.
// ---------------------------------------------------------------------------
KC_HD inline uint64_t reverse_complement(uint64_t fwd) {
    uint64_t rc = 0;
    for (int i = 0; i < KMER_K; ++i) {
        // Peel the lowest base off `fwd`, complement it (3 - base == base ^ 3),
        // and append it to `rc`. Because we consume `fwd` low-to-high and build
        // `rc` high-to-low, the bases come out in reversed order automatically.
        uint64_t base = fwd & 0x3ULL;          // lowest 2-bit base of the k-mer
        rc = (rc << 2) | (base ^ 0x3ULL);      // complement and shift into place
        fwd >>= 2;                             // advance to the next base
    }
    return rc & KMER_MASK;
}

// canonical: the strand-independent key = the smaller of the forward k-mer and
// its reverse complement. Both strands of the same locus therefore hash to one
// slot, halving the table and making classification strand-agnostic.
KC_HD inline uint64_t canonical_kmer(uint64_t fwd) {
    uint64_t rc = reverse_complement(fwd);
    return (fwd < rc) ? fwd : rc;
}

// ---------------------------------------------------------------------------
// hash_kmer: map a 64-bit canonical k-mer to a table slot in [0, capacity).
//   We use a fixed-constant variant of the well-known SplitMix64/Murmur finalizer
//   (xor-shift + odd-constant multiply). It scrambles the bits so structurally
//   similar k-mers (which differ in only a few low bits as the window slides)
//   land in well-separated slots, keeping the open-addressing probe short.
//   `capacity` is a power of two so `& (capacity-1)` replaces a modulo.
//   This is the EXACT same hash on CPU and GPU -> identical probe sequences.
// ---------------------------------------------------------------------------
KC_HD inline uint64_t hash_kmer(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;   // odd multiplier (invertible mix)
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

// ---------------------------------------------------------------------------
// table_lookup: probe an open-addressing (linear-probing) hash table for the
// taxon assigned to `key`. The table is two parallel arrays:
//   keys[slot]  : the canonical k-mer stored there (only valid if taxa[slot]!=0)
//   taxa[slot]  : the taxon id at that slot, or TAXON_EMPTY (0) if the slot is
//                 empty -- which also terminates the probe (the key is absent).
//   capacity    : number of slots, a power of two (so & mask == % capacity).
// Returns the taxon id (>=1) if found, or TAXON_UNCLASSIFIED (0) if not.
//
//   Linear probing: start at hash(key) & mask, walk forward slot by slot until
//   we either find the key (hit) or hit an empty slot (miss). Because BUILD used
//   the same hash and the same forward-walk insertion, this read-side walk is
//   guaranteed to find any inserted key -- and to be identical CPU vs GPU.
//   The `& mask` wraps the walk around the end of the table.
// ---------------------------------------------------------------------------
KC_HD inline uint32_t table_lookup(const uint64_t* keys, const uint32_t* taxa,
                                   uint64_t capacity, uint64_t key) {
    uint64_t mask = capacity - 1ULL;             // capacity is a power of two
    uint64_t slot = hash_kmer(key) & mask;       // home slot for this key
    // Probe at most `capacity` slots; in practice the table is < 70% full so the
    // average probe length is ~1-2. The bound also prevents an infinite loop if
    // the table were ever completely full (it never is -- we size it generously).
    for (uint64_t step = 0; step < capacity; ++step) {
        uint32_t t = taxa[slot];                 // taxon id at this slot
        if (t == TAXON_EMPTY) return TAXON_UNCLASSIFIED;  // empty -> key absent
        if (keys[slot] == key) return t;         // key matches -> this taxon
        slot = (slot + 1ULL) & mask;             // linear step, wrap at the end
    }
    return TAXON_UNCLASSIFIED;                    // table full & key absent (never)
}

// ---------------------------------------------------------------------------
// classify_read: the WHOLE per-read computation, shared by CPU and GPU.
//   Slides a k-mer window across the read (a rolling 2-bit encoding), looks each
//   canonical k-mer up in the table, and tallies a vote per taxon. The read is
//   assigned to the taxon with the most k-mer hits (a simplified Kraken2 vote;
//   the real LCA-on-a-taxonomy-tree version is described in THEORY). Ties are
//   broken by the LOWEST taxon id, which makes the result DETERMINISTIC and
//   independent of thread scheduling.
//
//   Parameters
//     read     : pointer to the read's bases (ASCII), NOT necessarily null-terminated
//     len      : number of bases in the read
//     keys/taxa/capacity : the reference hash table (see table_lookup)
//     votes    : scratch array of MAX_TAXA ints, provided by the caller (lives in
//                registers/local memory on the GPU, on the stack on the CPU). We
//                take it as a parameter rather than allocating, so this function
//                does zero dynamic allocation and is callable from a kernel.
//   Returns the winning taxon id (>=1), or TAXON_UNCLASSIFIED (0) if no k-mer
//   matched (e.g. a read with no reference k-mers, or all-N).
//
//   Complexity: O(len) hash probes per read (each probe ~O(1) amortized), so the
//   whole sample is O(total_bases) -- linear, alignment-free. That linearity is
//   why k-mer classification scales to clinical throughput.
// ---------------------------------------------------------------------------
KC_HD inline uint32_t classify_read(const char* read, int len,
                                    const uint64_t* keys, const uint32_t* taxa,
                                    uint64_t capacity, int* votes) {
    // Zero the per-taxon vote histogram. Index i holds the count of k-mers that
    // matched taxon id i (so index 0, the "unclassified/empty" sentinel, is unused).
    for (int t = 0; t < MAX_TAXA; ++t) votes[t] = 0;

    uint64_t fwd   = 0;   // rolling forward k-mer (low 2*KMER_K bits valid)
    int      valid = 0;   // count of consecutive valid bases currently in window
    int      hits  = 0;   // total k-mers that matched ANY taxon (for the miss case)

    // Slide across the read one base at a time, updating the rolling encoding.
    for (int p = 0; p < len; ++p) {
        uint8_t code = base_to_2bit(read[p]);
        if (code == 0xFF) {
            // An ambiguous base (N) breaks the window: we cannot form a valid
            // k-mer spanning it, so reset and start accumulating bases again.
            valid = 0;
            fwd   = 0;
            continue;
        }
        // Shift the new base into the low end and drop any bits above 2*KMER_K.
        fwd = ((fwd << 2) | code) & KMER_MASK;
        if (valid < KMER_K) ++valid;             // still filling the first window
        if (valid < KMER_K) continue;            // not enough bases yet -> no k-mer

        // We have a full k-mer ending at position p. Canonicalize (strand-agnostic)
        // and look it up; a hit votes for that taxon.
        uint64_t key   = canonical_kmer(fwd);
        uint32_t taxon = table_lookup(keys, taxa, capacity, key);
        if (taxon != TAXON_UNCLASSIFIED && taxon < (uint32_t)MAX_TAXA) {
            votes[taxon] += 1;
            hits += 1;
        }
    }

    if (hits == 0) return TAXON_UNCLASSIFIED;    // no reference k-mer matched

    // Majority vote with a deterministic tie-break (lowest id wins, because we
    // scan ids ascending and only replace on a STRICTLY greater count).
    int best_id = 0, best_votes = 0;
    for (int t = 1; t < MAX_TAXA; ++t) {
        if (votes[t] > best_votes) { best_votes = votes[t]; best_id = t; }
    }
    return (uint32_t)best_id;
}
