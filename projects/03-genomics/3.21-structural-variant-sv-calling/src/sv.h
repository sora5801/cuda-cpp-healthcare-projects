// ===========================================================================
// src/sv.h  --  Shared (host + device) structural-variant primitives
// ---------------------------------------------------------------------------
// Project 3.21 : Structural Variant (SV) Calling  (REDUCED-SCOPE teaching version)
//
// WHAT THIS PROJECT COMPUTES
//   Given long reads that *cross* a structural-variant breakpoint, recover the
//   variant. A deletion of length L makes a read look like:
//
//       reference  : ...A C G T [ . . . . . L bases deleted . . . . . ] G C A...
//       read       : ...A C G T                                        G C A...
//
//   The read aligns to the reference in two pieces ("split read"): a left piece
//   ending at the deletion's left breakpoint, then a jump of L bases, then a
//   right piece. Each read gives a NOISY estimate of the breakpoint position and
//   the deleted length. A real caller must:
//     (1) RE-ALIGN each split-read candidate precisely to nail the breakpoint
//         (we use a tiny BANDED Smith-Waterman to score the left flank), and
//     (2) CLUSTER the per-read breakpoint estimates that agree into one SV call,
//         counting the supporting reads (the "read support").
//
//   This is exactly the catalog's GPU recipe for 3.21:
//     "Banded SW CUDA kernels for breakpoint realignment" + "read cluster sorting"
//     + "breakpoint clustering" + "genotype likelihood".
//
// WHY A GPU
//   A long-read SV callset starts from MILLIONS of candidate reads, and each
//   read's re-alignment + breakpoint refinement is INDEPENDENT of every other
//   read -> embarrassingly parallel (one read per GPU thread, PATTERNS.md §1
//   "independent jobs"). The clustering step is a SCATTER-REDUCTION: many reads
//   vote into the same breakpoint bin -> atomicAdd (PATTERNS.md §1 "parallel
//   assign + atomic reduce", exemplar 11.09). This header holds the per-read
//   math so the CPU reference and the GPU kernel run BYTE-IDENTICAL code
//   (PATTERNS.md §2, the HD-macro idiom).
//
// DETERMINISM (PATTERNS.md §3/§4)
//   Everything here is INTEGER arithmetic: the banded SW score (integer match/
//   mismatch/gap costs), the breakpoint bin index (integer division), and the
//   cluster vote counts (atomicAdd on unsigned int). Integer atomic adds commute,
//   so the GPU histogram equals the CPU histogram EXACTLY -- no floating-point
//   reordering, no tolerance needed for the support counts. The only float in the
//   pipeline is the genotype likelihood ratio, computed deterministically on the
//   host from those exact integer counts. Verification is therefore EXACT.
//
//   NOTE: keep CUDA-only constructs (__global__, <<<>>>) OUT of this header so the
//   host compiler (cl.exe / g++) can include it when building reference_cpu.cpp.
//   Only __host__ __device__ inline helpers live here.
//
// READ THIS BEFORE: reference_cpu.h, kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <cstdint>

// HD = "host+device". Under nvcc (__CUDACC__ defined) we decorate the helpers so
// they compile for BOTH the CPU and the GPU. Under the plain host compiler the
// decorators do not exist, so we expand HD to nothing. This single trick is what
// guarantees the CPU reference and the GPU kernel evaluate the SAME formulas.
#ifdef __CUDACC__
#define SV_HD __host__ __device__
#else
#define SV_HD
#endif

// ---------------------------------------------------------------------------
// Problem-size limits (compile-time, so device code can stack-allocate). A real
// caller streams arbitrarily long reads; this teaching version fixes small caps
// so each thread's working set fits comfortably in registers/local memory.
// ---------------------------------------------------------------------------
static const int   SV_FLANK      = 24;   // bases of read flank we re-align (left of breakpoint)
static const int   SV_BAND       = 4;    // banded-SW half-bandwidth (|i-j| <= BAND)
static const int   SV_SEARCH     = 12;   // +/- search window (bp) around the read's raw breakpoint guess

// Integer alignment scores (so SW is exact integer DP; matches DeepVariant-style
// affine-free scoring used in teaching). Values mirror common SW defaults.
static const int   SV_MATCH      =  2;   // reward for a matching base
static const int   SV_MISMATCH   = -1;   // penalty for a substitution
static const int   SV_GAP        = -2;   // penalty for an indel step (linear gap)

// Breakpoint clustering: we bin refined breakpoints onto a 1-bp grid and merge
// votes within +/- SV_MERGE bins into a single SV call (long-read breakpoints are
// fuzzy by a few bp; SURVIVOR uses ~1 kb for short reads, far less for long).
static const int   SV_MERGE      = 3;    // merge radius in bp for calling

// A genome base is encoded as a small integer 0..3 (A,C,G,T); N / unknown = 4.
// We keep reads and reference as int8 arrays of these codes (compact, branchless
// compares in the kernel). Encoding lives in reference_cpu.cpp (host-only I/O).

// ---------------------------------------------------------------------------
// sv_match_score: the per-cell substitution score for banded SW.
//   a, b : base codes (0..4). Returns SV_MATCH if equal & known, else SV_MISMATCH.
//   N (code 4) never "matches" -- an ambiguous base cannot confirm an alignment.
//   This is the one place the scoring scheme is defined, shared by CPU and GPU.
// ---------------------------------------------------------------------------
SV_HD inline int sv_match_score(int a, int b) {
    if (a == 4 || b == 4) return SV_MISMATCH;   // ambiguous base: treat as mismatch
    return (a == b) ? SV_MATCH : SV_MISMATCH;
}

// ---------------------------------------------------------------------------
// sv_banded_sw: banded Smith-Waterman LOCAL alignment score of two short flanks.
//   read_flank[0..n)  : the read's bases just left of its breakpoint guess
//   ref_flank [0..m)  : the reference's bases just left of a CANDIDATE breakpoint
//   n, m              : lengths (<= SV_FLANK)
//   Returns the best local-alignment score (>= 0). Higher = the read's left piece
//   ends here more cleanly -> this candidate breakpoint is better supported.
//
//   WHY BANDED: a true breakpoint shifts the alignment by at most a few bases, so
//   only DP cells near the diagonal (|i-j| <= SV_BAND) can be on the optimal path.
//   Skipping the rest turns the O(n*m) full matrix into O(n*BAND) work -- the same
//   trick real long-read aligners (minimap2, the SW in pbsv) use to stay fast.
//   We keep just two rows (prev, curr) -> O(m) memory, fits in registers/local.
//
//   This is INTEGER DP: every cell is max of integer terms, so the CPU and GPU
//   produce identical scores (PATTERNS.md §4 "exact").
// ---------------------------------------------------------------------------
SV_HD inline int sv_banded_sw(const signed char* read_flank, int n,
                              const signed char* ref_flank,  int m) {
    // Two rolling rows of the DP matrix. SV_FLANK+1 columns max.
    int prev[SV_FLANK + 1];
    int curr[SV_FLANK + 1];

    int best = 0;                          // Smith-Waterman tracks the global max cell
    for (int j = 0; j <= m; ++j) prev[j] = 0;   // local alignment: top row is all 0

    for (int i = 1; i <= n; ++i) {
        // Only columns within the band around row i are meaningful; clamp the
        // scan to [i-BAND, i+BAND] and zero the cells just outside so the
        // recurrence reads a defined value at the band edge.
        int jlo = i - SV_BAND; if (jlo < 1) jlo = 1;
        int jhi = i + SV_BAND; if (jhi > m) jhi = m;

        curr[0] = 0;                        // local alignment: left column is all 0
        if (jlo - 1 >= 0) curr[jlo - 1] = 0;  // guard the cell left of the band
        for (int j = jlo; j <= jhi; ++j) {
            // Standard SW recurrence with a floor of 0 (local alignment can
            // restart anywhere). diag = extend the alignment with (mis)match;
            // up/left = open a gap in one sequence.
            int diag = prev[j - 1] + sv_match_score(read_flank[i - 1], ref_flank[j - 1]);
            int up   = prev[j]     + SV_GAP;
            int left = curr[j - 1] + SV_GAP;
            int s = diag;
            if (up   > s) s = up;
            if (left > s) s = left;
            if (s < 0) s = 0;               // SW floor: never go negative
            curr[j] = s;
            if (s > best) best = s;          // remember the global maximum
        }
        // Roll curr -> prev for the next row (copy only the touched band + edges).
        for (int j = jlo - 1; j <= jhi; ++j) if (j >= 0) prev[j] = curr[j];
    }
    return best;
}

// ---------------------------------------------------------------------------
// sv_refine_breakpoint: slide a small window to find the best left breakpoint.
//   For a read whose RAW breakpoint guess is reference position `guess`, we try
//   every candidate breakpoint in [guess-SEARCH, guess+SEARCH], re-align the
//   read's left flank ending at that candidate, and keep the position with the
//   highest banded-SW score. This is the "rapid re-alignment of split-read
//   candidates to pinpoint breakpoints precisely" the catalog asks for.
//
//   read_left  : the read's SV_FLANK bases ending at the read-side break (codes)
//   ref        : whole reference (codes), length ref_len
//   ref_len    : reference length
//   guess      : raw breakpoint estimate (reference coordinate, 0-based)
//   out_score  : (optional) receives the winning SW score for QC
//   Returns the refined breakpoint reference coordinate (clamped in range).
//
//   Each read calls this ONCE and independently -> perfect GPU parallelism.
// ---------------------------------------------------------------------------
SV_HD inline int sv_refine_breakpoint(const signed char* read_left, int read_len,
                                      const signed char* ref, int ref_len,
                                      int guess, int* out_score) {
    int best_pos   = guess;
    int best_score = -1;
    // Try each candidate left-breakpoint position in the search window.
    for (int delta = -SV_SEARCH; delta <= SV_SEARCH; ++delta) {
        int bp = guess + delta;
        if (bp < read_len || bp > ref_len) continue;   // need read_len ref bases to the left
        // Reference flank = the read_len bases of reference ending AT bp.
        const signed char* ref_flank = ref + (bp - read_len);
        int score = sv_banded_sw(read_left, read_len, ref_flank, read_len);
        // Tie-break toward the position closest to the raw guess (smaller |delta|),
        // which we get for free by using strict '>' and scanning delta ascending
        // only if we also prefer smaller |delta|. To stay fully deterministic and
        // independent of scan direction, prefer higher score, then SMALLER bp.
        if (score > best_score || (score == best_score && bp < best_pos)) {
            best_score = score;
            best_pos   = bp;
        }
    }
    if (out_score) *out_score = best_score;
    return best_pos;
}

// ---------------------------------------------------------------------------
// sv_bin: map a refined breakpoint coordinate to its 1-bp histogram bin.
//   Trivial here (1 bp grid), but kept as a named function so the CPU and GPU
//   agree on the binning and so the grid resolution is changeable in one place.
// ---------------------------------------------------------------------------
SV_HD inline int sv_bin(int breakpoint) { return breakpoint; }

// ---------------------------------------------------------------------------
// sv_geno_from_vaf: integer-only genotype call from supporting/total read counts.
//   support : reads supporting the variant (alt allele)
//   total   : reads spanning the locus (alt + ref)
//   Returns 0 = 0/0 (hom-ref), 1 = 0/1 (het), 2 = 1/1 (hom-alt), using fixed
//   variant-allele-fraction (VAF) cutoffs. We compare 4*support against integer
//   multiples of `total` to avoid any floating point in the genotype decision,
//   so CPU and GPU agree exactly:
//       VAF < 1/8        -> 0/0   (4*support <  total/2  i.e. 8*support < total)
//       1/8 <= VAF < 3/4 -> 0/1
//       VAF >= 3/4       -> 1/1   (4*support >= 3*total)
//   These are teaching thresholds, not production likelihoods (see THEORY §6).
// ---------------------------------------------------------------------------
SV_HD inline int sv_geno_from_vaf(unsigned int support, unsigned int total) {
    if (total == 0) return 0;
    // 8*support < total  <=>  VAF < 1/8
    if (8u * support < total) return 0;          // 0/0 hom-ref (likely noise)
    // 4*support >= 3*total  <=>  VAF >= 3/4
    if (4u * support >= 3u * total) return 2;     // 1/1 hom-alt
    return 1;                                     // 0/1 het
}
