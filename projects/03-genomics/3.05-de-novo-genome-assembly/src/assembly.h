// ===========================================================================
// src/assembly.h  --  Shared data model + the ONE TRUE per-element math
// ---------------------------------------------------------------------------
// Project 3.5 : De Novo Genome Assembly  (reduced-scope teaching version:
//               the minimizer-based ALL-VS-ALL READ OVERLAP stage)
//
// WHY THIS HEADER EXISTS  (PATTERNS.md sec.2: the __host__ __device__ core)
//   De-novo assembly reconstructs a genome from raw sequencing reads with NO
//   reference. Its first and most GPU-amenable bottleneck is all-vs-all read
//   *overlap detection*: which reads share enough sequence to be neighbours in
//   the assembly graph? Comparing read i against read j is independent of every
//   other pair, so it is the classic "embarrassingly parallel" workload (the
//   same shape as project 1.12, but over PAIRS instead of one-query-vs-N).
//
//   The per-pair score (how many minimizers two reads share) must be computed
//   IDENTICALLY by the CPU reference (reference_cpu.cpp, compiled by cl.exe) and
//   by the GPU kernel (kernels.cu, compiled by nvcc). To guarantee that, the
//   shared integer arithmetic lives here as `__host__ __device__` inline
//   functions, included by BOTH sides. Integer counts are associative and have
//   no rounding, so CPU and GPU agree BIT-FOR-BIT (tolerance == 0; PATTERNS.md
//   sec.4 "Exact").
//
//   Keep this header free of CUDA-only constructs (no __global__, no <<<>>>)
//   so the host compiler can include it. The HD macro below makes the
//   __host__ __device__ decorators vanish when compiled by a plain C++ compiler.
//
// READ THIS BEFORE: reference_cpu.h, kernels.cuh. The science/derivation is in
// ../THEORY.md; the catalog deep-dive is project 3.5.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// HD: the host/device portability shim (PATTERNS.md sec.2).
//   * Under nvcc, __CUDACC__ is defined -> functions are tagged so they compile
//     for BOTH the CPU (host) and the GPU (device): the exact same code runs
//     in the reference and in the kernel.
//   * Under cl.exe/g++ (reference_cpu.cpp), the decorators do not exist, so HD
//     expands to nothing and the function is an ordinary inline.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ---------------------------------------------------------------------------
// Sketching parameters (minimap2's vocabulary; see THEORY "The algorithm").
//   K  : k-mer length. A k-mer is a length-K substring of a read; we slide it
//        one base at a time. K=15 is a typical long-read default (large enough
//        to be mostly unique in a small genome, small enough to survive errors).
//   W  : window length, in k-mers. A *minimizer* is the smallest-hash k-mer in
//        each window of W consecutive k-mers. Picking ~1/W of the k-mers shrinks
//        the sketch ~W-fold while two overlapping reads still tend to pick the
//        SAME minimizers (that is the whole point of the minimizer trick).
//   These are compile-time constants so loops unroll and the 2-bit packing of a
//   K=15 k-mer fits in a single 32-bit integer (2*15 = 30 bits).
// ---------------------------------------------------------------------------
constexpr int K = 15;   // k-mer length in bases
constexpr int W = 5;    // window length in k-mers

// A minimizer is stored as a 32-bit hash key. (For K<=16 the 2-bit-packed
// canonical k-mer itself fits in 32 bits, and we hash it for good spread.)
using minimizer_t = std::uint32_t;

// ---------------------------------------------------------------------------
// hash32: a fast integer mixer (Thomas Wang's 32-bit hash).
//   Minimizers are chosen by SMALLEST HASH, not smallest k-mer value, so that
//   the chosen set is a pseudo-random but DETERMINISTIC sample of k-mers (a raw
//   lexicographic minimum would over-pick poly-A runs). The function is a pure
//   bijection of bit-mixing steps -> identical on host and device, fully
//   deterministic. Marked HD so both sides hash a k-mer the same way.
//   x : the 2-bit-packed canonical k-mer.  returns: its 32-bit hash.
// ---------------------------------------------------------------------------
HD inline std::uint32_t hash32(std::uint32_t x) {
    x = (x ^ 61u) ^ (x >> 16);
    x = x + (x << 3);
    x = x ^ (x >> 4);
    x = x * 0x27d4eb2du;     // a large odd multiplier -> good avalanche
    x = x ^ (x >> 15);
    return x;
}

// ---------------------------------------------------------------------------
// count_shared_sorted: the ONE TRUE per-pair score, shared by CPU and GPU.
//   Given two SORTED, DEDUPLICATED minimizer lists (read a and read b), count
//   how many minimizer keys they have in common. This is a classic merge-style
//   set intersection: walk both lists with two cursors, advancing the smaller.
//   Complexity O(na + nb) per pair -- linear, branch-light, no allocation.
//
//   Returning an INTEGER is deliberate: integer addition is exact and order-
//   independent, so the host loop and the single GPU thread produce identical
//   results with zero floating-point ambiguity (THEORY "Verification").
//
//   a, na : pointer + length of read a's sorted unique minimizers
//   b, nb : pointer + length of read b's sorted unique minimizers
//   returns: number of shared minimizers (a similarity score; higher = more
//            likely the two reads overlap and should be graph neighbours).
// ---------------------------------------------------------------------------
HD inline int count_shared_sorted(const minimizer_t* a, int na,
                                  const minimizer_t* b, int nb) {
    int i = 0, j = 0, shared = 0;
    while (i < na && j < nb) {
        const minimizer_t av = a[i];
        const minimizer_t bv = b[j];
        // Branchless-ish merge: advance whichever cursor points at the smaller
        // key; when they are equal, count one match and advance both.
        if (av < bv)      { ++i; }
        else if (bv < av) { ++j; }
        else              { ++shared; ++i; ++j; }   // equal keys -> a shared minimizer
    }
    return shared;
}

// ---------------------------------------------------------------------------
// ReadSet: a batch of reads stored in the FLATTENED, GPU-friendly layout the
// kernel consumes. We pre-compute every read's minimizer sketch on the host
// (sketching is cheap and inherently serial per read), then ship the flat
// arrays to the device where the O(n^2) pairwise comparison happens.
//
//   n          : number of reads.
//   mins       : ALL reads' minimizers concatenated, each read's slice already
//                SORTED and DEDUPLICATED (so count_shared_sorted works).
//   offset     : [n+1] prefix-sum (CSR-style) into `mins`. Read r's minimizers
//                are mins[offset[r] .. offset[r+1]-1]; offset[n] == mins.size().
//                This is the standard "ragged array" layout: one flat buffer +
//                an offsets array, so a thread can find any read's slice in O(1)
//                without a 2-D jagged structure (which GPUs handle poorly).
//   read_len   : [n] original read length in bases (for reporting only).
// ---------------------------------------------------------------------------
struct ReadSet {
    int n = 0;
    std::vector<minimizer_t> mins;   // concatenated sorted-unique sketches
    std::vector<int>         offset; // [n+1] CSR offsets into `mins`
    std::vector<int>         read_len;
};

// An overlap edge in the assembly graph: reads `i` and `j` (i<j) share
// `shared` minimizers. The list of these edges IS the overlap graph that a full
// assembler would then thread into contigs (THEORY "Where this sits"). We emit
// edges with shared >= a threshold, deterministically ordered.
struct Overlap {
    int i;       // first  read index (always < j)
    int j;       // second read index
    int shared;  // shared-minimizer count (the edge weight)
};

// ---------------------------------------------------------------------------
// pair_to_ij / num_pairs: map a flat pair index to the (i,j) upper-triangle
// coordinate, and count how many unordered pairs exist for n reads.
//   The all-vs-all comparison has P = n*(n-1)/2 unordered pairs (i<j). We give
//   each pair its own GPU thread, indexed by a single flat id `p in [0,P)`, then
//   decode (i,j) from p so threads need no 2-D grid. These helpers are HD so the
//   CPU reference can decode pairs the identical way (keeping the two in lockstep
//   even in how they enumerate pairs). Derivation in THEORY "GPU mapping".
// ---------------------------------------------------------------------------
HD inline long long num_pairs(int n) {
    return (long long)n * (n - 1) / 2;   // n choose 2
}

// Decode flat pair index p (0-based) into upper-triangle (i,j) with i<j.
//   We walk the triangle row by row: row i (i=0..n-2) holds (n-1-i) pairs. This
//   closed-free decode keeps the host and device enumeration identical and
//   avoids storing an (i,j) table. For the tiny teaching sizes here the linear
//   walk is plenty; THEORY notes the O(1) sqrt decode used at scale.
HD inline void pair_to_ij(long long p, int n, int* i_out, int* j_out) {
    int i = 0;
    long long row = (long long)(n - 1);   // pairs in row 0
    while (p >= row) { p -= row; ++i; --row; }
    *i_out = i;
    *j_out = i + 1 + (int)p;
}

// Build a ReadSet (sketch every read into sorted-unique minimizers). Declared
// here, defined in reference_cpu.cpp because it is pure host C++ (string I/O +
// std::sort) shared by the whole program. See reference_cpu.h for the loader.
ReadSet sketch_reads(const std::vector<std::string>& reads);
