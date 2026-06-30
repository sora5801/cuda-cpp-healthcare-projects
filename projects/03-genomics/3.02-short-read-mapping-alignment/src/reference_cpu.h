// ===========================================================================
// src/reference_cpu.h  --  Data model, the shared scoring core, CPU reference
// ---------------------------------------------------------------------------
// Project 3.2 : Short-Read Mapping / Alignment
//
// WHAT THIS PROJECT COMPUTES
//   Given a (small) REFERENCE genome and a batch of short READS, find for each
//   read the position in the reference where it best aligns. This is the core
//   of every short-read mapper (BWA-MEM, Bowtie2, NVIDIA Parabricks): line each
//   sequenced fragment up against the genome so downstream tools can call
//   variants. We use the classic two-phase recipe:
//
//     (1) SEED   -- take the read's leading k-mer (k consecutive bases), look it
//                   up in an index built from the reference's k-mers. The index
//                   returns every reference offset where that exact k-mer occurs.
//                   Those are the only positions worth scoring -- a huge pruning
//                   win versus scoring the read at all L_ref positions.
//     (2) EXTEND -- at each candidate offset p (corrected so the read's first
//                   base lines up with the start of the matched k-mer) lay the
//                   whole read against the reference window and score it. We use
//                   UNGAPPED extension: score = sum over read bases of
//                   (+MATCH if the base equals the reference base, else MISMATCH).
//                   The best-scoring offset is the read's mapping.
//
//   This is the "Beginner / Established" teaching slice the catalog asks for:
//   a real seed-and-extend pipeline, but with a sorted-k-mer index (instead of
//   an FM-index/BWT) and UNGAPPED extension (instead of banded gapped Smith-
//   Waterman). THEORY.md explains exactly what the full version adds, and where
//   project 3.01 (the gapped-SW wavefront) fits in.
//
// WHY A GPU
//   Reads are mutually INDEPENDENT: read 7's best position does not depend on
//   read 3's. So we give each read its own GPU thread, and all reads are mapped
//   in one parallel launch -- the "independent jobs" pattern (cf. 1.12 Tanimoto).
//   At WGS scale (~900 M reads) this embarrassingly-parallel structure is exactly
//   why GPUs map a 30x human genome in minutes instead of >30 CPU-hours.
//
// THE SHARED CORE (CPU/GPU PARITY)
//   The per-window scoring math lives in ONE `__host__ __device__` function,
//   score_window(), defined in this header. The CPU reference loops it; the GPU
//   kernel calls it from one thread. Same integer arithmetic on both sides ->
//   results are BYTE-IDENTICAL, so verification is exact (==), not approximate.
//   (PATTERNS.md section 2: the HD-macro idiom.)
//
//   This header is pure C++/CUDA-attribute-only: it is included by
//   reference_cpu.cpp (host compiler), by main.cu, and by kernels.cu (nvcc).
//   No `__global__`, no kernel-launch syntax here, so the host compiler is happy.
//
// READ THIS BEFORE: kernels.cuh, reference_cpu.cpp, main.cu.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// HD: expands to `__host__ __device__` when compiled by nvcc (so the function
// can run on BOTH the CPU and inside a kernel), and to nothing when compiled by
// the plain host compiler (which has never heard of those attributes). This one
// macro is what lets reference_cpu.cpp and kernels.cu share the exact same math.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ---------------------------------------------------------------------------
// Problem-wide constants. These are compile-time so both the CPU reference and
// the GPU kernel see identical values (and the kernel can unroll the k-mer loop).
// ---------------------------------------------------------------------------
constexpr int SEED_K   = 12;   // seed length in bases. 12 -> 4^12 = ~16.7M
                               // possible k-mers, plenty to make a 12-mer nearly
                               // unique in our small synthetic reference. Real
                               // mappers use ~19-21 bp minimizer seeds.
constexpr int MATCH    =  1;   // score added per read base that equals the ref
constexpr int MISMATCH = -1;   // score added per read base that differs from ref
constexpr int NO_HIT   = -1;   // sentinel: "this read mapped nowhere"

constexpr char ALPHABET[] = "ACGT";  // code 0..3 <-> nucleotide character

// ---------------------------------------------------------------------------
// A loaded mapping problem: one reference sequence + many equal-length reads,
// all encoded as 0..3 base codes (A=0,C=1,G=2,T=3). We require equal-length
// reads purely to keep the teaching code simple (a single read_len, regular
// 2-D indexing); real data has variable lengths handled with an offset array.
// ---------------------------------------------------------------------------
struct MappingProblem {
    int                   ref_len  = 0;   // L_ref: bases in the reference
    int                   read_len = 0;   // L: bases in every read (uniform here)
    int                   n_reads  = 0;   // R: number of reads to map
    std::vector<uint8_t>  ref;            // [ref_len] encoded reference
    std::vector<uint8_t>  reads;          // [n_reads * read_len] row-major reads
};

// ---------------------------------------------------------------------------
// The k-mer index over the reference, in the flat layout the GPU wants. For each
// of the n_kmers = ref_len - SEED_K + 1 reference windows we record the integer
// code of its k-mer and the offset it lives at. We SORT those pairs by code on
// the host so that a query k-mer is found with a branchless binary search over a
// contiguous array -- deterministic and coalescing-friendly. (THEORY section 4
// explains why a sorted array beats a chained hash table on a GPU.)
//
//   sorted_codes[i]   : the i-th smallest reference k-mer code (ascending)
//   sorted_offsets[i] : the reference offset that k-mer came from
// Equal codes are contiguous, so all offsets for one query k-mer form a single
// [lo, hi) run found by two binary searches.
// ---------------------------------------------------------------------------
struct KmerIndex {
    int                   n_kmers = 0;     // number of indexed reference windows
    std::vector<uint64_t> sorted_codes;    // [n_kmers] k-mer codes, ascending
    std::vector<int>      sorted_offsets;  // [n_kmers] matching ref offsets
};

// The result for one read: where it mapped and how well.
struct MapResult {
    int pos   = NO_HIT;   // best reference offset (read base 0 aligns here), or NO_HIT
    int score = 0;        // best ungapped alignment score at that offset
    int mism  = 0;        // number of mismatching bases in that best alignment
};

// ===========================================================================
//                       THE SHARED `__host__ __device__` CORE
// ===========================================================================

// ---------------------------------------------------------------------------
// kmer_code: pack SEED_K base codes starting at seq[start] into one 64-bit
// integer (2 bits per base). Because k=12 needs only 24 bits this never
// overflows. Identical packing on CPU and GPU means a read's seed code matches
// the reference's seed code exactly. Caller guarantees start+SEED_K <= length.
//   seq   : pointer to encoded bases (host or device memory)
//   start : index of the first base of the k-mer
//   returns: the 2-bit-packed code, MSB = first base
// ---------------------------------------------------------------------------
HD inline uint64_t kmer_code(const uint8_t* seq, int start) {
    uint64_t code = 0;
    for (int b = 0; b < SEED_K; ++b) {
        // Shift left 2 and OR in the next base's 2-bit code. The loop bound is a
        // compile-time constant (SEED_K), so nvcc fully unrolls it.
        code = (code << 2) | static_cast<uint64_t>(seq[start + b]);
    }
    return code;
}

// ---------------------------------------------------------------------------
// score_window: THE per-alignment physics, shared verbatim by CPU and GPU.
// Lay a read of length read_len against the reference starting at offset `pos`
// and score it ungapped: +MATCH where the bases agree, +MISMATCH where they
// differ. Also count the mismatches (handy for a CIGAR/edit summary).
//
//   ref       : encoded reference, length ref_len
//   ref_len   : bases in the reference
//   read      : encoded read, length read_len (points at this read's row)
//   read_len  : bases in the read
//   pos       : reference offset where read base 0 is placed (may be anything)
//   out_mism  : (out) number of mismatching bases in this alignment
//   returns   : the integer alignment score, or a large negative if the window
//               falls off either end of the reference (an invalid placement).
//
// Integer-only on purpose: integer adds are associative and exact, so the CPU
// loop and the single-thread GPU call produce the SAME number -> exact (==)
// verification (PATTERNS.md section 4). No early-exit on mismatch, so the work
// per window is fixed and the two implementations stay lockstep.
// ---------------------------------------------------------------------------
HD inline int score_window(const uint8_t* ref, int ref_len,
                           const uint8_t* read, int read_len,
                           int pos, int* out_mism) {
    // Reject placements that hang off either end of the reference. We return a
    // very negative score so such a window can never win the max-reduction.
    if (pos < 0 || pos + read_len > ref_len) {
        if (out_mism) *out_mism = read_len;   // "all mismatch" for bookkeeping
        return -1000000;
    }
    int score = 0;
    int mism  = 0;
    for (int b = 0; b < read_len; ++b) {
        // Compare the read base to the reference base it lands on.
        if (read[b] == ref[pos + b]) {
            score += MATCH;
        } else {
            score += MISMATCH;
            ++mism;
        }
    }
    if (out_mism) *out_mism = mism;
    return score;
}

// ---------------------------------------------------------------------------
// kmer_equal_range: find the half-open run [lo, hi) of indices in the ascending
// array `sorted` whose value equals `code`, via two binary searches (lower- and
// upper-bound). If the code is absent, lo == hi (an empty run = "seed not in
// reference"). Shared by the CPU reference and mirrored by the device kernel so
// both seed identically. Pure integer comparisons -> deterministic on both sides.
//   sorted : ascending array of k-mer codes
//   n      : its length
//   code   : the query k-mer code
//   lo,hi  : (out) the matching run; empty run means the seed was not found
// ---------------------------------------------------------------------------
HD inline void kmer_equal_range(const uint64_t* sorted, int n, uint64_t code,
                                int* lo, int* hi) {
    // lower_bound: first index with sorted[i] >= code.
    int l = 0, r = n;
    while (l < r) {
        int mid = l + ((r - l) >> 1);
        if (sorted[mid] < code) l = mid + 1; else r = mid;
    }
    const int first = l;
    // upper_bound: first index with sorted[i] > code (continue from `first`).
    l = first; r = n;
    while (l < r) {
        int mid = l + ((r - l) >> 1);
        if (sorted[mid] <= code) l = mid + 1; else r = mid;
    }
    *lo = first;
    *hi = l;
}

// ===========================================================================
//                       Host-only API (declared here, defined in .cpp)
// ===========================================================================

// Load a mapping problem from the simple text format documented in
// data/README.md (line 1: reference; following lines: one read each). Encodes
// ACGT -> 0..3. Throws std::runtime_error on a bad file / non-ACGT character /
// reads of unequal length. Pure host code (uses std::ifstream).
MappingProblem load_problem(const std::string& path);

// Build the sorted k-mer index over the reference (host only). This is the
// one-time "index the genome" step; in production it is done once and reused for
// billions of reads. Sorting by k-mer code lets both CPU and GPU look a seed up
// with a binary search. Deterministic: a stable sort keyed by (code, offset).
KmerIndex build_index(const MappingProblem& prob);

// CPU REFERENCE: map every read serially using the index + score_window(). This
// is the trusted baseline the GPU is checked against (every read's chosen pos
// and score must match exactly). Fills results[n_reads].
//   prob    : the loaded problem
//   index   : the prebuilt k-mer index (so CPU and GPU seed identically)
//   results : resized to n_reads; per-read best pos/score/mismatches
void map_reads_cpu(const MappingProblem& prob, const KmerIndex& index,
                   std::vector<MapResult>& results);
