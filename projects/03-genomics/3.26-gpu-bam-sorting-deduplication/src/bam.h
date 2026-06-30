// ===========================================================================
// src/bam.h  --  Shared (host + device) BAM-record primitives
// ---------------------------------------------------------------------------
// Project 3.26 : GPU BAM Sorting & Deduplication
//
// WHAT THIS PROJECT COMPUTES
//   After short reads are ALIGNED to a reference genome, a sequencing pipeline
//   must do two housekeeping steps before variant calling:
//
//     (1) COORDINATE SORT -- reorder the reads so they appear in genome order,
//         i.e. sorted by (reference/chromosome id, leftmost position, strand).
//         Downstream tools (pileup, variant callers, index builders) all assume
//         coordinate-sorted input. `samtools sort` does this on the CPU.
//
//     (2) DUPLICATE MARKING -- during library prep, PCR amplification can copy
//         the SAME original DNA fragment many times. Those copies map to the
//         exact same place and inflate apparent read depth, biasing variant
//         calls. We GROUP reads that share a "fragment signature"
//         (ref, position, strand, mate position) and keep ONE representative --
//         the highest-quality copy -- marking the rest as duplicates. This is
//         exactly what Picard MarkDuplicates / `samtools markdup` do.
//
//   This is a REDUCED-SCOPE TEACHING MODEL (CLAUDE.md §13): we operate on a
//   simple fixed-width read record (below) held in memory, not on a real
//   compressed BGZF/BAM file with CIGAR strings, clipping, and a full BAI/CSI
//   index. The catalog's research-grade tool is NVIDIA Parabricks `fq2bam`,
//   which fuses GPU sort + markdup into the alignment step. THEORY.md spells
//   out exactly what we simplify and how production differs.
//
// WHY A GPU
//   A whole-genome BAM holds ~10^9 reads. Coordinate sort is a RADIX SORT on an
//   integer key -- one of the few algorithms where a GPU's huge memory
//   bandwidth gives an order-of-magnitude win over a CPU comparison sort.
//   Duplicate marking is a HASH/GROUP-AGGREGATE: once reads are sorted by the
//   dedup signature, equal-signature reads are contiguous, so finding each
//   group's best-quality copy becomes a segmented reduction (reduce_by_key) --
//   again bandwidth-bound and ideal for the GPU.
//
// THE DETERMINISM TRICK (so CPU and GPU agree EXACTLY)
//   Every quantity we sort or compare is an INTEGER (ids, positions, quality
//   sums). Integer comparisons are exact and order-independent, so the GPU
//   radix sort and the CPU std::sort produce byte-identical orderings -- as
//   long as we make the sort keys TOTAL (no ties left to chance). We do that by
//   appending the read's original input index as the lowest-order tie-breaker.
//   The same total order drives duplicate marking, so "which copy is the
//   representative" is deterministic on both sides. No floating point appears
//   anywhere in the comparison path -- verification is therefore EXACT.
//
//   The helpers below are __host__ __device__ (BAM_HD) so the CPU reference and
//   the GPU kernels share the SAME key-packing and the SAME comparison math.
//   Keeping CUDA-only types (no __global__, no thrust) out of this header lets
//   the plain host compiler include it for reference_cpu.cpp (PATTERNS.md §2).
//
// READ THIS AFTER: nothing (start here), then reference_cpu.h, kernels.cuh.
// ===========================================================================
#pragma once

#include <cstdint>

// BAM_HD expands to __host__ __device__ under nvcc, and to nothing under the
// plain host compiler (which has never heard of those decorators). This is the
// "HD-macro idiom" -- one body, compiled for both CPU and GPU.
#ifdef __CUDACC__
#define BAM_HD __host__ __device__
#else
#define BAM_HD
#endif

// ---------------------------------------------------------------------------
// ReadRecord -- our minimal stand-in for one aligned BAM read.
//
//   A real BAM record carries 11+ mandatory fields (QNAME, FLAG, RNAME, POS,
//   MAPQ, CIGAR, RNEXT, PNEXT, TLEN, SEQ, QUAL) plus optional tags. For
//   teaching the sort+dedup algorithms we only need the fields those two steps
//   actually look at, so the record is a small POD (plain-old-data) struct:
//   trivially copyable to the GPU with a single cudaMemcpy, no pointers inside.
//
//   Field-by-field (all integers -> exact, deterministic comparisons):
//     ref_id   : reference sequence index (which chromosome). 0..numRefs-1.
//                Real BAM calls this RNAME (an index into the header @SQ list).
//     pos      : 0-based leftmost mapped coordinate on that reference (POS).
//     strand   : 0 = forward (+), 1 = reverse (-). Derived from the FLAG bit
//                0x10 in real BAM. Strand matters for dedup: a read and its
//                reverse-complement map to opposite strands and are NOT dups.
//     mate_pos : leftmost coordinate of the mate / the fragment's other end.
//                Together with (ref,pos,strand) this forms the duplicate
//                "signature": two reads from the same original fragment share
//                all four. (For single-end reads this would be the read's own
//                end; we keep it explicit for clarity.)
//     base_qual_sum : Picard's duplicate SCORE -- the sum of base qualities of
//                the read (higher = more confident bases). When several reads
//                share a signature, the one with the LARGEST score is kept as
//                the original and the rest are flagged as duplicates.
//     id       : the read's original index in the input (0..n-1). It is the
//                FINAL tie-breaker that makes every ordering total and therefore
//                reproducible. It also lets us report results in stable order.
// ---------------------------------------------------------------------------
struct ReadRecord {
    int32_t  ref_id;          // chromosome index (RNAME)
    int32_t  pos;             // leftmost mapped coordinate (POS), 0-based
    int32_t  strand;          // 0 = '+'(forward), 1 = '-'(reverse)
    int32_t  mate_pos;        // mate / fragment-end coordinate (PNEXT)
    int32_t  base_qual_sum;   // duplicate score = sum of base qualities
    int32_t  id;              // original input index; total-order tie-breaker
};

// ---------------------------------------------------------------------------
// coord_key -- pack (ref_id, pos, strand) into ONE unsigned 64-bit integer so a
//   single radix sort orders reads exactly the way `samtools sort` does.
//
//   WHY PACK INTO ONE KEY: thrust::sort_by_key does a high-throughput RADIX
//   sort on a scalar key. A radix sort processes the key bit-by-bit (or
//   digit-by-digit) and is NOT comparison-based, so it cannot call a custom
//   multi-field comparator cheaply. The standard trick is to fold the sort
//   fields into one wide integer whose natural unsigned order equals the
//   desired lexicographic order: most-significant bits = primary field.
//
//   LAYOUT (high bits -> low bits):
//     [63..40] ref_id   (24 bits)  -- primary: chromosome
//     [39..16] pos      (24 bits)  -- secondary: position within chromosome
//     [15.. 1] (unused, zero)
//     [    0 ] strand   ( 1 bit)   -- tertiary: + before -
//
//   24 bits hold positions up to ~16.7 Mb and ref ids up to ~16.7 M, which is
//   plenty for this teaching model (real CSI indexing uses 32-bit positions;
//   THEORY.md §real-world covers the extension). Inputs are validated by the
//   loader so they fit. We mask each field so a stray high bit cannot corrupt a
//   neighbour.
//
//   NOTE ON TOTAL ORDER: this key alone can tie (two reads at the same ref/pos/
//   strand). The sort uses `id` as a second key (sort_by_key is stable for ties
//   only if we make it so) -- see kernels.cu / reference_cpu.cpp, which carry
//   `id` alongside and break ties on it for a fully reproducible order.
// ---------------------------------------------------------------------------
BAM_HD inline uint64_t coord_key(int32_t ref_id, int32_t pos, int32_t strand) {
    const uint64_t r = static_cast<uint64_t>(ref_id)  & 0xFFFFFFull;   // 24 bits
    const uint64_t p = static_cast<uint64_t>(pos)     & 0xFFFFFFull;   // 24 bits
    const uint64_t s = static_cast<uint64_t>(strand)  & 0x1ull;        //  1 bit
    return (r << 40) | (p << 16) | s;
}

// Convenience overload taking a whole record.
BAM_HD inline uint64_t coord_key(const ReadRecord& rec) {
    return coord_key(rec.ref_id, rec.pos, rec.strand);
}

// ---------------------------------------------------------------------------
// dup_key -- pack the DUPLICATE SIGNATURE (ref_id, pos, strand, mate_pos) into
//   one 64-bit integer. Two reads with equal dup_key came (we infer) from the
//   same original DNA fragment and are PCR/optical duplicates of each other.
//
//   We reuse coord_key's (ref,pos,strand) packing and XOR-mix the mate position
//   into the low bits. Because mate_pos occupies a region coord_key left zero
//   (bits [15..1]) we instead place it in a SEPARATE 64-bit field pairing -- but
//   to keep a single scalar key for reduce_by_key we fold mate_pos into the
//   spare low bits with a bounded mask. To avoid collisions between the
//   24-bit (ref,pos) and the mate field we keep mate_pos to 15 bits here
//   (positions up to ~32k within a tile); the loader validates the range and
//   the synthetic generator respects it. THEORY.md explains why a real
//   implementation would use a 128-bit key or sort the four fields directly.
//
//   The ONLY property we need from dup_key is: equal signature <=> equal key,
//   for the inputs we generate. Grouping (reduce_by_key) then makes
//   equal-signature reads contiguous so we can pick each group's best score.
// ---------------------------------------------------------------------------
BAM_HD inline uint64_t dup_key(int32_t ref_id, int32_t pos, int32_t strand, int32_t mate_pos) {
    const uint64_t base = coord_key(ref_id, pos, strand);             // [63..16]+strand
    const uint64_t m    = static_cast<uint64_t>(mate_pos) & 0x7FFFull; // 15 bits
    return base | (m << 1);   // strand sits at bit 0; mate fills bits [15..1]
}

BAM_HD inline uint64_t dup_key(const ReadRecord& rec) {
    return dup_key(rec.ref_id, rec.pos, rec.strand, rec.mate_pos);
}

// ---------------------------------------------------------------------------
// coord_less -- the TOTAL coordinate-sort order used by BOTH the CPU reference
//   and the GPU sort's tie-break, so the two produce byte-identical orderings.
//
//   Order: by coord_key first (ref, then pos, then strand), and when those tie,
//   by original `id` (ascending). Appending `id` guarantees NO ties remain, so
//   the sorted sequence is unique -> deterministic and reproducible. Returns
//   true iff record `a` should come before record `b`.
// ---------------------------------------------------------------------------
BAM_HD inline bool coord_less(const ReadRecord& a, const ReadRecord& b) {
    const uint64_t ka = coord_key(a), kb = coord_key(b);
    if (ka != kb) return ka < kb;
    return a.id < b.id;        // total-order tie-break
}

// ---------------------------------------------------------------------------
// is_better_dup -- given two reads with the SAME duplicate signature, decide
//   which one is the "original" to KEEP (the other becomes a marked duplicate).
//   Picard's rule: keep the higher base-quality sum. We break exact-score ties
//   on the lower original `id`, so the choice is deterministic and matches
//   between CPU and GPU. Returns true iff `a` is the better (keep-worthy) copy.
// ---------------------------------------------------------------------------
BAM_HD inline bool is_better_dup(const ReadRecord& a, const ReadRecord& b) {
    if (a.base_qual_sum != b.base_qual_sum)
        return a.base_qual_sum > b.base_qual_sum;   // higher quality wins
    return a.id < b.id;                             // tie -> lower id wins
}
