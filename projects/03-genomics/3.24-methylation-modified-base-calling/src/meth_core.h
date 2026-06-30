// ===========================================================================
// src/meth_core.h  --  The ONE TRUE per-event physics, shared by CPU and GPU
// ---------------------------------------------------------------------------
// Project 3.24 : Methylation / Modified-Base Calling
//
// WHY THIS HEADER EXISTS (the most important idea in the project; PATTERNS.md §2)
//   The CPU reference (reference_cpu.cpp, compiled by cl.exe) and the GPU kernel
//   (kernels.cu, compiled by nvcc) must compute *byte-for-byte identical* math so
//   that verification is EXACT, not "close enough". We guarantee that by putting
//   every per-element formula here, as `__host__ __device__` inline functions,
//   and including this single header from BOTH compilers. The host reference
//   loops these functions; each GPU thread calls the very same functions.
//
//   Keep this header CUDA-type-free (no __global__, no kernels, no <cuda_runtime>)
//   so the plain C++ host compiler can include it. The only CUDA-ism is the
//   `MC_HD` decorator, which expands to nothing under the host compiler.
//
// WHAT THE SCIENCE IS (read THEORY.md "The science" for the full story)
//   A nanopore sequencer threads a single DNA strand through a protein pore and
//   measures the ionic current. At any instant the current depends on the ~k
//   bases (a "k-mer") sitting in the pore's narrowest point. A trained "pore
//   model" gives, for each of the 4^k k-mers, the EXPECTED current (a Gaussian:
//   mean `level_mean`, std `level_stdv`). Base-calling/segmentation turns the raw
//   sample stream into a list of "events" (one mean current per dwell).
//
//   A cytosine that is METHYLATED (5mC) sits a little differently in the pore and
//   shifts the current for the k-mers that contain it. So we keep TWO pore models:
//   a CANONICAL model (unmodified C) and a METHYLATED model (5mC). To decide if a
//   given CpG site is methylated, we ask: under which model do the observed events
//   covering that site look more probable? That is a LOG-LIKELIHOOD RATIO (LLR).
//
// WHAT THIS FILE PROVIDES
//   * Kmer encoding (2-bit per base) and the per-k-mer Gaussian pore-model entry.
//   * gaussian_logpdf()     : log N(x | mean, stdv)   -- the emission log-prob.
//   * event_emission_logp() : log-prob of one event given the k-mer it aligns to.
//   * The band geometry helpers shared by the DP (so CPU/GPU index identically).
//
// READ THIS BEFORE: kernels.cuh, reference_cpu.h (both include this file).
// ===========================================================================
#pragma once

#include <cmath>     // std::log, std::sqrt, std::exp (host); device uses ::log etc.
#include <cstdint>   // fixed-width integer types for k-mer codes

// ---------------------------------------------------------------------------
// MC_HD: the host/device decorator. Under nvcc (__CUDACC__ defined) a function
// marked MC_HD is compiled for BOTH the CPU and the GPU. Under the plain host
// compiler the decorator must vanish (cl.exe has never heard of __host__).
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define MC_HD __host__ __device__
#else
#define MC_HD
#endif

// ---------------------------------------------------------------------------
// Problem-size constants (kept tiny and fixed so the teaching demo is legible
// and the DP band fits comfortably in registers / shared memory).
//
//   KMER_K     : pore-model k-mer length. Real ONT R10 models use k = 9; we use
//                k = 3 (64 k-mers) so the committed pore model is small and the
//                reader can read it by eye. THEORY.md "real world" explains the
//                step up to k = 9 (262144 entries) and why nothing else changes.
//   NUM_KMERS  : 4^KMER_K, the number of pore-model rows.
//   BAND_WIDTH : the adaptive band's half-width in reference positions. f5c uses
//                an adaptive band that follows the best diagonal; for teaching we
//                use a FIXED band of this half-width around the main diagonal,
//                which is the simplest correct version (THEORY.md "The algorithm"
//                describes the adaptive upgrade we deliberately defer).
// ---------------------------------------------------------------------------
#define KMER_K     3
#define NUM_KMERS  64          // 4^3
#define BAND_WIDTH 6           // band half-width (positions either side of diagonal)

// A single pore-model entry: the Gaussian a k-mer's current is expected to follow.
//   level_mean : expected event mean current (in normalized "pA-like" units)
//   level_stdv : expected event standard deviation (spread of the current)
// Two such tables exist per read set: canonical and methylated (see meth_core
// usage in main.cu). POD struct so it copies trivially to the device.
struct PoreModelEntry {
    float level_mean;   // expected current for this k-mer
    float level_stdv;   // expected current spread for this k-mer (> 0)
};

// ---------------------------------------------------------------------------
// base_to_code: map a DNA base character to its 2-bit code (A=0,C=1,G=2,T=3).
//   Returns 0 for any unexpected character so a malformed reference degrades
//   gracefully instead of indexing out of bounds. Host-only (parses text), so it
//   is a plain inline function, NOT MC_HD.
// ---------------------------------------------------------------------------
inline int base_to_code(char b) {
    switch (b) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default:            return 0;   // unknown -> treat as 'A'
    }
}

// ---------------------------------------------------------------------------
// kmer_code: pack KMER_K consecutive base-codes (most-significant base first)
// into an integer in [0, NUM_KMERS). This is the row index into a pore model.
//   `codes` points at the first base of the k-mer; the caller guarantees there
//   are at least KMER_K valid bases. MC_HD because the DP (device) re-derives
//   k-mer codes for the reference window it is aligning to.
// ---------------------------------------------------------------------------
MC_HD inline int kmer_code(const int* codes) {
    int c = 0;
    for (int i = 0; i < KMER_K; ++i) {
        c = (c << 2) | (codes[i] & 0x3);   // shift in 2 bits per base
    }
    return c;
}

// ---------------------------------------------------------------------------
// gaussian_logpdf: log of the Normal density,  log N(x | mu, sigma)
//      = -0.5*((x-mu)/sigma)^2 - log(sigma) - 0.5*log(2*pi).
//   This is the heart of the emission cost: how well does the observed event
//   current `x` match the pore model's expectation (mu, sigma) for a k-mer?
//   Higher (less negative) = better fit. We compute in DOUBLE so the CPU and GPU
//   reductions agree to machine precision (the inputs are float, the arithmetic
//   is double, the final compare uses a documented tolerance -- see main.cu).
//
//   MC_HD so the identical formula runs on both sides. We call ::log/::sqrt
//   (the global-namespace math) which resolves to the right host or device
//   intrinsic automatically.
// ---------------------------------------------------------------------------
MC_HD inline double gaussian_logpdf(double x, double mu, double sigma) {
    const double inv_sigma = 1.0 / sigma;          // sigma > 0 guaranteed by data
    const double z = (x - mu) * inv_sigma;          // standardized residual
    // -0.5*log(2*pi) as a literal so host and device share the exact constant.
    const double LOG_SQRT_2PI = 0.9189385332046727; // = 0.5*log(2*pi)
    return -0.5 * z * z - log(sigma) - LOG_SQRT_2PI;
}

// ---------------------------------------------------------------------------
// event_emission_logp: log-probability of observing event current `x` when the
// pore is sitting on k-mer index `kc`, under a given pore model table.
//   This is just gaussian_logpdf specialized to that k-mer's (mean, stdv). It is
//   the per-cell EMISSION term the banded DP accumulates as it threads events
//   onto reference positions. MC_HD: same call on CPU and GPU.
//     model : pointer to NUM_KMERS PoreModelEntry rows
//     kc    : k-mer code in [0, NUM_KMERS)
//     x     : observed event mean current
// ---------------------------------------------------------------------------
MC_HD inline double event_emission_logp(const PoreModelEntry* model, int kc, double x) {
    return gaussian_logpdf(x, (double)model[kc].level_mean, (double)model[kc].level_stdv);
}

// ---------------------------------------------------------------------------
// THE BANDED EVENT-ALIGNMENT DP (shared CPU/GPU core).
//
//   This is the f5c-style recurrence at the heart of methylation calling, written
//   ONCE here so the CPU reference and the GPU kernel run bit-identical math. It
//   aligns `n_events` observed event currents onto `n_kmers` reference k-mers and
//   returns the LOG-LIKELIHOOD of the best alignment path.
//
//   STATE.  dp[i][j] = best log-likelihood of aligning the first i events to the
//   first j reference k-mers, with event (i-1) emitted by k-mer (j-1) on a MATCH.
//   We allow three moves into cell (i,j):
//     * MATCH   (i-1, j-1) -> (i,j): consume event i, advance to k-mer j; emit.
//     * STAY    (i-1, j)   -> (i,j): consume event i, k-mer unchanged (the pore
//                 dwelled, producing an extra event on the same k-mer); emit +
//                 a small STAY penalty. (f5c's "insertion to event".)
//     * SKIP    (i,   j-1) -> (i,j): advance k-mer without consuming an event (a
//                 fast base skipped its event); a SKIP penalty, NO emission.
//   We take the MAX over the three predecessors (Viterbi-style best path, the log
//   of a max-product), so the result is the single most-likely alignment's logL.
//
//   THE BAND.  Only cells with |i - j| <= BAND_WIDTH are reachable; everything
//   outside the band is -infinity (NEG_INF). This is the adaptive-banded idea in
//   its simplest fixed-band form: the alignment cannot wander far from the
//   diagonal, which both reflects the biology (events track the reference roughly
//   1:1) and cuts the DP from O(n^2) to O(n * band). THEORY.md "The algorithm"
//   describes f5c's *adaptive* band that re-centers on the running best cell; we
//   defer that (CLAUDE.md §13) and document it.
//
//   MEMORY.  We need only the previous and current DP rows (the recurrence reaches
//   back one row), so we keep two length-(n_kmers+1) rows on the stack. With
//   n_kmers tiny (10 here) this is a handful of doubles per thread -- it lives in
//   registers/local memory on the GPU, no shared memory needed for this size.
//
//   PARAMETERS
//     events   : [n_events]  observed event mean currents
//     kmer_ids : [n_kmers]   reference k-mer codes (precomputed by the caller)
//     model    : [NUM_KMERS] the pore model to score under (canonical OR meth)
//     n_events : number of events  (<= MAX_DP_LEN)
//     n_kmers  : number of k-mers  (<= MAX_DP_LEN)
//   RETURNS the best-path log-likelihood (a double; more negative = worse fit).
//
//   MC_HD: identical on host and device. Bounded loops + a fixed-size scratch
//   array make it safe to call from a single GPU thread.
// ---------------------------------------------------------------------------

// Upper bound on the DP dimension, sizing the on-stack scratch rows. WINDOW_KMERS
// (defined in reference_cpu.h) is 10; we round up generously so the buffers are
// safe even if a learner widens the window in an exercise.
#define MAX_DP_LEN 64

// A very negative sentinel standing in for log(0) = -infinity (an unreachable
// state). Using a finite, large-magnitude value keeps the max() arithmetic free
// of NaNs/Inf propagation while being effectively "impossible".
#define MC_NEG_INF (-1.0e18)

MC_HD inline double banded_align_core(const float* events, const int* kmer_ids,
                                      const PoreModelEntry* model,
                                      int n_events, int n_kmers) {
    // Move penalties (log-prob of a non-match transition). Chosen small and
    // negative so the path PREFERS clean diagonal matches but can stay/skip when
    // the data demands it. These are constants shared by CPU and GPU verbatim.
    const double STAY_PENALTY = -1.0;   // log-prob of an extra event on same k-mer
    const double SKIP_PENALTY = -2.0;   // log-prob of skipping a k-mer's event

    // Two DP rows: prev = row (i-1), cur = row i. +1 for the j=0 boundary column.
    double prev[MAX_DP_LEN + 1];
    double cur[MAX_DP_LEN + 1];

    // ---- Initialize row i = 0 (no events consumed yet) --------------------
    // dp[0][0] = 0 (empty alignment, probability 1, logL 0). dp[0][j>0] would mean
    // skipping j k-mers before any event; allowed only inside the band, each skip
    // costing SKIP_PENALTY. Outside the band -> NEG_INF.
    for (int j = 0; j <= n_kmers; ++j) {
        if (j == 0)            prev[j] = 0.0;
        else if (j <= BAND_WIDTH) prev[j] = prev[j - 1] + SKIP_PENALTY;
        else                   prev[j] = MC_NEG_INF;
    }

    // ---- Fill rows i = 1..n_events ----------------------------------------
    for (int i = 1; i <= n_events; ++i) {
        const double x = (double)events[i - 1];   // the current event's mean current

        // j = 0 column: i>0 events consumed but 0 k-mers passed. There is no real
        // k-mer to emit from, so this state is unreachable -> NEG_INF. (Only
        // dp[0][0]=0 seeds the recurrence; every path must emit from a real k-mer.)
        cur[0] = MC_NEG_INF;

        for (int j = 1; j <= n_kmers; ++j) {
            // BAND GUARD: cells far from the diagonal are unreachable.
            if (j - i > BAND_WIDTH || i - j > BAND_WIDTH) {
                cur[j] = MC_NEG_INF;
                continue;
            }
            // Emission of event i under reference k-mer (j-1).
            const double emit = event_emission_logp(model, kmer_ids[j - 1], x);

            // Three candidate predecessors (Viterbi max). Guard each against the
            // NEG_INF sentinel so an unreachable predecessor stays unreachable.
            double best = MC_NEG_INF;
            // MATCH: came diagonally, emit here.
            double cand = prev[j - 1] + emit;
            if (cand > best) best = cand;
            // STAY: extra event on the same k-mer (j unchanged), emit + penalty.
            cand = prev[j] + emit + STAY_PENALTY;
            if (cand > best) best = cand;
            // SKIP: advanced k-mer without consuming an event; no emission.
            cand = cur[j - 1] + SKIP_PENALTY;
            if (cand > best) best = cand;

            cur[j] = best;
        }
        // Roll cur -> prev for the next event row.
        for (int j = 0; j <= n_kmers; ++j) prev[j] = cur[j];
    }

    // The best alignment must consume all events and all k-mers: dp[n_events][n_kmers].
    return prev[n_kmers];
}
