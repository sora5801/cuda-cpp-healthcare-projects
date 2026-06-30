// ===========================================================================
// src/bqsr.h  --  Shared (host + device) BQSR primitives & data layout
// ---------------------------------------------------------------------------
// Project 3.25 : Base Quality Score Recalibration (BQSR)
//
// WHAT THIS PROJECT COMPUTES (the reduced-scope teaching version)
//   A sequencer reports, for every base it calls, a PHRED QUALITY SCORE Q -- its
//   own estimate of the probability that the base is wrong: P_err = 10^(-Q/10).
//   These reported scores are SYSTEMATICALLY BIASED: the real error rate drifts
//   with the machine cycle (position along the read), the local sequence context
//   (the di-nucleotide ending at the base), and the reported Q itself. BQSR
//   measures that bias empirically and rewrites every quality score to match the
//   error rate actually observed. Variant callers downstream (GATK, DeepVariant)
//   trust these recalibrated scores, so getting them right matters.
//
//   The measurement works like this (GATK's BaseRecalibrator, simplified):
//     1. Walk every base of every aligned read.
//     2. SKIP bases that sit on a KNOWN VARIANT (dbSNP / Mills indels): a
//        mismatch there is a real biological difference, not a machine error, so
//        counting it would poison the error-rate estimate. This is the
//        "known-variant masking" step.
//     3. For every surviving base, decide if it is an ERROR (it disagrees with
//        the reference base) and drop it into a COVARIATE BIN keyed by
//        (reported-Q, cycle, di-nucleotide context). Each bin tallies two
//        integers: how many bases landed there (observations) and how many were
//        errors.
//     4. Per bin, the EMPIRICAL quality is the PHRED of the measured error rate,
//        with a +1 Yates / Laplace correction so a zero-error bin is finite:
//             Q_emp(bin) = round( -10 * log10( (errors+1) / (obs+1) ) )
//     5. RECALIBRATE: each base's new quality is the Q_emp of its bin.
//
// WHY A GPU
//   A 30x whole-genome BAM is ~1e9 reads / ~1e11 bases; the covariate scan is the
//   cost. Every base is INDEPENDENT until the tally, so we give each base its own
//   GPU thread (PATTERNS.md row "clustering / centroid accumulation"). The tally
//   is a SCATTER-REDUCTION: millions of threads add into a few thousand shared
//   bins -> atomicAdd. NVIDIA Parabricks does exactly this and matches GATK.
//
// DETERMINISM TRICK (same lesson as 5.01 and 11.09)
//   Float atomicAdd is order-dependent (non-associative) -> irreproducible. Here
//   the tallies are COUNTS, so we atomicAdd into UNSIGNED INTEGERS, which commute:
//   the GPU table is bit-identical run-to-run AND equals the CPU table exactly.
//   Empirical Q is then derived from those exact integer counts.
//
//   The covariate-binning math and the Q<->P conversions live here as
//   __host__ __device__ (BQSR_HD) inline functions so the CPU reference and the
//   GPU kernel run BYTE-FOR-BYTE identical math -> exact verification.
//
//   Keep CUDA-only constructs (__global__, kernel launches) OUT of this header so
//   the plain host compiler can include it too. READ THIS BEFORE: kernels.cuh,
//   reference_cpu.h.
// ===========================================================================
#pragma once

#include <cmath>     // std::log10, std::pow, std::lround
#include <cstdint>   // fixed-width integer types

// The HD-macro idiom (PATTERNS.md §2): under nvcc (__CUDACC__ defined) decorate
// the shared helpers as callable from both host and device; under the plain host
// compiler the decorators do not exist, so expand to nothing.
#ifdef __CUDACC__
#define BQSR_HD __host__ __device__
#else
#define BQSR_HD
#endif

// ---------------------------------------------------------------------------
// Covariate-space dimensions (the shape of the count table).
//
//   * Q is the reported PHRED score. Illumina reports 0..40-ish; we cap the table
//     at NUM_Q bins so Q in [0, NUM_Q) indexes directly.
//   * cycle is the 0-based position of the base within its read (the machine
//     "cycle"). Errors rise toward the end of a read, so cycle is a covariate.
//   * context is the di-nucleotide ending at this base: the previous reference
//     base (4 options A/C/G/T) and the current one (4) -> 16 contexts. The very
//     first base of a read has no previous base; we use a dedicated "no context"
//     slot (index NUM_CONTEXT-1) for it.
//
// The full GATK model also keys on read group; this teaching version fixes a
// single read group (one lane/sample) so the table stays small and legible. The
// table is a flat array of (obs,err) pairs indexed by covariate_index() below.
// ---------------------------------------------------------------------------
static const int NUM_Q       = 43;            // reported-quality bins: Q in [0,42]
static const int MAX_CYCLE   = 16;            // read length we bin (cycles 0..15)
static const int NUM_CONTEXT = 17;            // 16 di-nucleotides + 1 "no-context"
static const int NO_CONTEXT  = NUM_CONTEXT-1; // slot for the first base of a read

// Total number of covariate bins. One unsigned-int OBS counter and one ERR
// counter per bin live in two parallel arrays of this length.
static const int NUM_BINS = NUM_Q * MAX_CYCLE * NUM_CONTEXT;

// ---------------------------------------------------------------------------
// covariate_index: map a (Q, cycle, context) triple to a flat bin index.
//   Row-major over (Q, cycle, context). Callers guarantee each field is in range
//   (the loader clamps Q to [0,NUM_Q) and only emits cycles < MAX_CYCLE). Used
//   identically by the CPU reference and the GPU kernel so both hit the SAME bin.
// ---------------------------------------------------------------------------
BQSR_HD inline int covariate_index(int q, int cycle, int context) {
    return (q * MAX_CYCLE + cycle) * NUM_CONTEXT + context;
}

// ---------------------------------------------------------------------------
// base_code / dinuc_context: turn nucleotide letters into the context covariate.
//   base_code maps A,C,G,T -> 0,1,2,3 and anything else (N, lowercase) -> -1.
//   dinuc_context combines the previous and current reference base codes into a
//   0..15 index, or NO_CONTEXT if either base is unknown (e.g. read start or N).
//   We key context on the REFERENCE bases (not the read's) so the context of a
//   given genome position is the same for every read covering it -- exactly how
//   GATK defines the sequence-context covariate.
// ---------------------------------------------------------------------------
BQSR_HD inline int base_code(char b) {
    switch (b) {
        case 'A': return 0;
        case 'C': return 1;
        case 'G': return 2;
        case 'T': return 3;
        default:  return -1;   // 'N' or any non-ACGT: no usable context
    }
}

BQSR_HD inline int dinuc_context(char prev_ref, char cur_ref) {
    const int p = base_code(prev_ref);
    const int c = base_code(cur_ref);
    if (p < 0 || c < 0) return NO_CONTEXT;   // unknown -> dedicated slot
    return p * 4 + c;                        // 0..15
}

// ---------------------------------------------------------------------------
// PHRED <-> probability helpers (the core BQSR arithmetic).
//   phred_to_p : Q -> error probability  P = 10^(-Q/10).
//   p_to_phred : P -> Q = -10 log10(P), rounded to the nearest integer (quality
//                scores are stored as integers in BAM/SAM).
//   We compute in double so the CPU and GPU agree to the last bit (the kernel
//   uses the same double-precision log10/pow).
// ---------------------------------------------------------------------------
BQSR_HD inline double phred_to_p(int q) {
    return pow(10.0, -static_cast<double>(q) / 10.0);
}

BQSR_HD inline int p_to_phred(double p) {
    // Guard the log of zero (a bin with measured error rate exactly 0 should not
    // happen here because of the +1 correction, but be safe and cap at Q=93,
    // the SAM spec's maximum representable quality).
    if (p <= 0.0) return 93;
    long q = lround(-10.0 * log10(p));
    if (q < 0)  q = 0;
    if (q > 93) q = 93;
    return static_cast<int>(q);
}

// ---------------------------------------------------------------------------
// empirical_q: the recalibrated quality for a covariate bin from its integer
//   counts. This is the heart of BQSR. With `obs` observations and `err` errors
//   we estimate the error rate with a +1 Yates/Laplace correction (GATK does the
//   same) so an all-correct bin still yields a finite, conservative quality:
//
//        P_emp = (err + 1) / (obs + 1)   ->   Q_emp = -10 log10(P_emp)
//
//   A bin that was never observed (obs==0) has no evidence; we return the
//   sentinel -1 so the caller falls back to the base's original reported Q.
//   Both CPU and GPU recalibration call THIS function on the SAME integer table,
//   so their recalibrated scores are identical.
// ---------------------------------------------------------------------------
BQSR_HD inline int empirical_q(unsigned int obs, unsigned int err) {
    if (obs == 0u) return -1;                       // unseen bin: no recalibration
    const double p_emp = (static_cast<double>(err) + 1.0)
                       / (static_cast<double>(obs) + 1.0);
    return p_to_phred(p_emp);
}

// ---------------------------------------------------------------------------
// classify_base: the SINGLE per-base routine shared by the CPU loop and the GPU
//   kernel (HD-macro idiom). Given the flattened read arrays and a global base
//   index `g` (0 .. R*L-1), it decides whether this base contributes to the
//   covariate table and, if so, which bin it falls in and whether it is an error.
//
//   Pointers (not std::vector) so the SAME function works on host data
//   (vector.data()) and on device pointers inside the kernel.
//
//   Parameters:
//     g          : global base index; read = g / read_len, cycle = g % read_len.
//     read_len   : fixed bases-per-read (L).
//     ref        : the reference string (1 char/base), length ref_len.
//     ref_len    : length of `ref`, to bounds-check the lookup.
//     bases      : [R*L] called base letters.
//     quals      : [R*L] reported PHRED scores.
//     read_pos   : [R] reference start position of each read.
//     known      : [ref_len] known-variant mask (1 => skip this reference pos).
//   Outputs (only meaningful when the function returns true):
//     out_bin    : the covariate bin index (covariate_index(q,cycle,context)).
//     out_is_err : 1 if the called base disagrees with the reference, else 0.
//   Returns true if the base should be TALLIED, false if it must be SKIPPED.
//
//   Skip rules (mirror GATK): skip if the called base or reference base is not a
//   clean A/C/G/T (an 'N' has no defined error), skip if the reference position is
//   a KNOWN VARIANT (a mismatch there is biology, not machine error), and skip if
//   the reported quality is out of the table's range.
// ---------------------------------------------------------------------------
BQSR_HD inline bool classify_base(int g, int read_len,
                                  const char* ref, int ref_len,
                                  const char* bases, const int* quals,
                                  const int* read_pos,
                                  const unsigned char* known,
                                  int* out_bin, int* out_is_err) {
    const int read  = g / read_len;          // which read this base belongs to
    const int cycle = g - read * read_len;    // 0-based position within the read
    const int refp  = read_pos[read] + cycle; // reference coordinate of this base

    if (refp < 0 || refp >= ref_len) return false;   // off the reference -> skip
    if (known[refp] != 0)            return false;    // known variant -> mask out

    const char called = bases[g];
    const char refbase = ref[refp];
    if (base_code(called) < 0 || base_code(refbase) < 0) return false; // N -> skip

    const int q = quals[g];
    if (q < 0 || q >= NUM_Q) return false;            // out-of-range Q -> skip

    // Sequence context = di-nucleotide ending here: previous reference base (or
    // "no context" if this is the first cycle) and the current reference base.
    const char prev_ref = (cycle > 0 && refp - 1 >= 0) ? ref[refp - 1] : 'N';
    const int  context  = dinuc_context(prev_ref, refbase);

    *out_bin    = covariate_index(q, cycle, context);
    *out_is_err = (called != refbase) ? 1 : 0;        // mismatch == sequencing error
    return true;
}
