// ===========================================================================
// src/reference_cpu.h  --  Data model + CPU reference for HiFi overlap chaining
// ---------------------------------------------------------------------------
// Project 3.20 : Long-Read HiFi Assembly Overlap & Polishing
//
// WHY A PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the host C++ compiler and must not see any
//   CUDA syntax, so the shared DATA MODEL (the minimiser-sketch container, the
//   file loader, the pair-result struct) and the CPU reference prototype live
//   here. kernels.cuh also includes this header so the GPU side reuses exactly
//   the same ReadSet / OverlapResult types -- nothing CUDA-specific leaks across.
//
// THE PROBLEM  (see ../THEORY.md for the full derivation)
//   We are given N long reads. We want the all-vs-all OVERLAP graph: for every
//   ordered pair (i, j) with i < j, how strongly do reads i and j overlap?
//   Comparing whole reads base-by-base is O(L^2) per pair and O(N^2 L^2) overall
//   -- hopeless at N = millions. The standard fix (minimap2, Darwin, hifiasm):
//
//     1) SKETCH each read down to its MINIMISERS -- a sparse, strand-symmetric
//        subset of its k-mers (one per window). This is done once per read.
//     2) For a pair, the shared minimisers are SEED ANCHORS: positions (qpos,
//        tpos) carrying the same minimiser hash. A true overlap shows up as a
//        diagonal band of anchors.
//     3) CHAIN the anchors: find the best collinear run (both coordinates
//        increasing) by dynamic programming. The chain's integer score is the
//        overlap strength. (overlap_core.h holds the per-link scoring.)
//
//   Step 3 is the per-pair work this project parallelises: each ordered pair is
//   INDEPENDENT, so each GPU thread owns one pair (kernels.cu). The number of
//   pairs is N*(N-1)/2 -- the O(N^2) blow-up the GPU attacks.
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cuh. The math is overlap_core.h.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "overlap_core.h"   // OVL_K, OVL_MAX_ANCHORS, ovl_chain_link_score, ...

// ---------------------------------------------------------------------------
// A single minimiser seed within one read:
//   pos  : the base offset of the minimiser's k-mer in the read (0-based).
//   hash : the canonical, mixed minimiser hash (see ovl_canonical_kmer_hash).
// Reads are stored as their list of minimisers, SORTED BY pos ascending -- the
// loader guarantees that order so the pairwise anchor merge below is linear.
// ---------------------------------------------------------------------------
struct Minimizer {
    int32_t  pos;    // k-mer start position in the read (bases)
    uint32_t hash;   // canonical minimiser hash
};

// ---------------------------------------------------------------------------
// ReadSet: the whole sketched dataset in a flat, GPU-friendly layout.
//   We store ALL reads' minimisers back-to-back in one `mins` array and index
//   into it with per-read [offset, count). A flat array (rather than a vector of
//   vectors) is what we can hand straight to cudaMemcpy -- the GPU cannot chase
//   host pointers, so contiguous storage + offsets is the canonical pattern.
//
//   n_reads          : number of reads N.
//   read_len[r]      : length of read r in bases (for end-overlap bookkeeping).
//   off[r], cnt[r]   : read r's minimisers are mins[off[r] .. off[r]+cnt[r]-1].
//   mins             : all minimisers concatenated, each read's slice sorted by pos.
// ---------------------------------------------------------------------------
struct ReadSet {
    int n_reads = 0;
    std::vector<int32_t>   read_len;   // [n_reads]
    std::vector<int32_t>   off;        // [n_reads]   start index into mins
    std::vector<int32_t>   cnt;        // [n_reads]   number of minimisers
    std::vector<Minimizer> mins;       // [sum(cnt)]  concatenated, pos-sorted

    // Total number of ordered pairs (i<j) we will score = the work size.
    long long num_pairs() const {
        const long long N = n_reads;
        return N * (N - 1) / 2;
    }
};

// ---------------------------------------------------------------------------
// OverlapResult: one scored read pair. The output is a flat array of these, one
// per (i<j) pair, in a DETERMINISTIC pair order (i outer, j inner) so the CPU
// and GPU produce identical arrays we can compare element by element.
//   read_i, read_j : the pair (i < j).
//   score          : best collinear chain score (integer; the overlap strength).
//   n_anchors      : how many shared minimisers the pair had (diagnostic).
// ---------------------------------------------------------------------------
struct OverlapResult {
    int32_t read_i;
    int32_t read_j;
    int32_t score;
    int32_t n_anchors;
};

// ---------------------------------------------------------------------------
// load_reads: parse the sketched dataset produced by scripts/make_synthetic.py.
//   Text format (documented fully in data/README.md):
//     line 1            : "<n_reads>"
//     then per read r   : "<read_len> <cnt>" followed by cnt lines "<pos> <hash>"
//                         (hash is an 8-hex-digit 32-bit value; pos ascending).
//   Throws std::runtime_error on a missing file or malformed record so demos
//   fail loudly instead of silently scoring nothing.
// ---------------------------------------------------------------------------
ReadSet load_reads(const std::string& path);

// ---------------------------------------------------------------------------
// pair_index: map an ordered pair (i, j), i < j, to its slot in the flat output
//   array, and back. We enumerate pairs row by row of the upper triangle:
//       (0,1)(0,2)...(0,N-1) (1,2)...(1,N-1) ... (N-2,N-1)
//   so the slot of (i,j) is  i*N - i*(i+1)/2 + (j - i - 1).  The GPU kernel uses
//   the inverse (slot -> (i,j)) so thread `t` knows which pair it owns. Keeping
//   both in one place guarantees CPU and GPU agree on the ordering.
// ---------------------------------------------------------------------------
inline long long pair_index(int i, int j, int n) {
    return static_cast<long long>(i) * n - static_cast<long long>(i) * (i + 1) / 2
           + (j - i - 1);
}

// ---------------------------------------------------------------------------
// chain_overlap_score: the heart of the computation. Given two reads' minimiser
//   slices, it (a) builds the shared-seed anchors by a linear merge over the two
//   pos-sorted, hash-bearing lists, then (b) runs the O(A^2) collinear chaining
//   DP from overlap_core.h, returning the best chain score and anchor count.
//
//   This is declared here and defined in reference_cpu.cpp; kernels.cu has a
//   device twin that calls the SAME overlap_core.h link function so the integer
//   scores match exactly. (We do NOT mark this OVL_HD because it allocates a
//   merge buffer; the GPU version uses fixed on-thread scratch instead.)
//
//   q_*  : read i's minimisers (slice + count)
//   t_*  : read j's minimisers (slice + count)
//   out_n_anchors : receives the number of shared-seed anchors used.
//   returns        : the best collinear chain score (0 if no usable anchors).
// ---------------------------------------------------------------------------
int chain_overlap_score(const Minimizer* q_min, int q_cnt,
                        const Minimizer* t_min, int t_cnt,
                        int* out_n_anchors);

// ---------------------------------------------------------------------------
// overlap_cpu: the trusted serial baseline. Loops over every ordered pair (i<j),
//   calls chain_overlap_score, and writes one OverlapResult per pair into `out`
//   in pair_index order. This is what the GPU result is verified against.
//   out is resized to rs.num_pairs().
// ---------------------------------------------------------------------------
void overlap_cpu(const ReadSet& rs, std::vector<OverlapResult>& out);
