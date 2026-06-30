// ===========================================================================
// src/motif_core.h  --  The ONE TRUE per-window PWM score (shared CPU + GPU)
// ---------------------------------------------------------------------------
// Project 3.29 : Motif Finding in Genomic Sequences
//
// WHY THIS HEADER EXISTS (the HD-macro idiom -- PATTERNS.md sec 2)
//   The single most expensive step of MEME-style motif finding is the E-step:
//   score EVERY length-W window in EVERY sequence against the current motif
//   model (a position weight matrix, PWM). That score is a small dot product:
//
//       score(window) = sum_{p=0..W-1}  log2( PWM[p][base_p] / background[base_p] )
//
//   We want the CPU reference and the GPU kernel to compute this BYTE-FOR-BYTE
//   identically, so verification is EXACT, not approximate. The trick: put the
//   per-window formula in ONE inline function marked `__host__ __device__`, and
//   include this header from BOTH reference_cpu.cpp (host compiler) AND
//   kernels.cu (nvcc). The host loops it; the kernel calls it from one thread.
//
//   Keep this header free of any CUDA-only types (no `__global__`, no
//   `<<<>>>`), so the plain C++ compiler can include it unchanged.
//
// READ THIS AFTER: reference_cpu.h (the data model).  THEN: kernels.cu.
// ===========================================================================
#pragma once

// ---------------------------------------------------------------------------
// HD : expands to `__host__ __device__` when compiled by nvcc (so the function
//   exists on both sides), and to nothing for the host C++ compiler (which has
//   never heard of those decorators). This is the portable-parity idiom.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// The DNA alphabet is fixed at 4 letters. We encode bases as small integers so
// a window is just an array of indices into the PWM rows:
//   A=0, C=1, G=2, T=3.  Any other character (N, lowercase, gap) is encoded as
//   4 ("unknown") by the loader and makes a window un-scoreable (skipped).
#define MOTIF_ALPHABET 4   // |{A,C,G,T}|
#define MOTIF_BASE_N   4   // sentinel index meaning "not A/C/G/T"

// ---------------------------------------------------------------------------
// window_score : the per-window log-odds score (the "one true formula").
//
//   seq      : pointer to the encoded sequence (bytes in {0,1,2,3,4}).
//   start    : index of the window's first base inside `seq`.
//   w        : motif width W (number of PWM columns).
//   logodds  : [w * MOTIF_ALPHABET] row-major log-odds table. Entry
//              logodds[p*4 + b] = log2( PWM[p][b] / bg[b] ) -- precomputed once
//              per EM iteration on the host (see build_logodds in reference_cpu).
//              Using a PRECOMPUTED table (not raw probabilities) means the inner
//              loop is W additions of table lookups: cheap, branch-free, and --
//              crucially -- the SAME arithmetic on CPU and GPU.
//
//   returns  : sum of the W log-odds terms (a `float`; bits-per-position info).
//              A window containing any non-ACGT base returns -INF-ish via the
//              caller's masking; here we assume the caller already checked, so
//              this function is a clean dot product with no data-dependent
//              branches (good for the GPU: no warp divergence).
//
//   Determinism: the loop runs p = 0,1,...,W-1 in a FIXED order on both sides,
//   and float addition -- though not associative -- is deterministic for a
//   fixed order. CPU and GPU therefore produce IDENTICAL bits (THEORY sec
//   "How we verify correctness").
// ---------------------------------------------------------------------------
HD inline float window_score(const unsigned char* seq,
                             int start,
                             int w,
                             const float* logodds) {
    float s = 0.0f;                       // running log-odds sum for this window
    for (int p = 0; p < w; ++p) {         // walk the W motif columns in order
        int b = seq[start + p];           // encoded base at column p (0..3)
        // The caller guarantees b in {0,1,2,3} for scored windows, so this is a
        // pure table lookup + add: logodds[p*4 + b]. No branch -> no divergence.
        s += logodds[p * MOTIF_ALPHABET + b];
    }
    return s;
}
