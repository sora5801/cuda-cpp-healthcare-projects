// ===========================================================================
// src/sa_core.h  --  The ONE shared core for CPU + GPU suffix-array math
// ---------------------------------------------------------------------------
// Project 3.27 : Suffix Array / BWT / FM-Index Construction
//
// WHY THIS HEADER EXISTS  (PATTERNS.md section 2: the __host__ __device__ core)
//   The single most useful idiom in this repo: put the *per-element math* in ONE
//   header marked __host__ __device__, so the CPU reference (reference_cpu.cpp,
//   compiled by cl.exe) and the GPU kernels (kernels.cu, compiled by nvcc) run
//   BYTE-FOR-BYTE IDENTICAL arithmetic. Then "GPU == CPU" verification is EXACT
//   (tolerance 0), not "close enough". Every value here is an integer, so there
//   is no floating-point reordering to worry about at all.
//
//   The whole suffix-array algorithm is "prefix doubling": we repeatedly sort the
//   n suffixes by a *pair of ranks*, then renumber. The two pieces of math that
//   MUST match between CPU and GPU are:
//     (1) how we PACK a rank-pair (r1, r2) into one sortable 64-bit key, and
//     (2) how we DECIDE two adjacent suffixes share a rank (key equality).
//   Both live here as tiny inline functions, so neither side can drift.
//
//   This header must stay free of CUDA-only types and of __global__ (the host
//   compiler includes it too). The HD macro below expands to nothing for cl.exe
//   and to "__host__ __device__" for nvcc -- the standard HD-macro idiom.
//
// READ THIS BEFORE: reference_cpu.cpp (host loops these) and kernels.cu (device
//   threads call these). See ../THEORY.md for the full algorithm + GPU mapping.
// ===========================================================================
#pragma once

#include <cstdint>   // std::int32_t, std::uint64_t

// ---------------------------------------------------------------------------
// HD: the host/device decorator macro.
//   * Under nvcc (__CUDACC__ defined) it becomes "__host__ __device__", so the
//     same function compiles for BOTH the CPU and the GPU.
//   * Under the plain host compiler the decorators do not exist, so HD vanishes.
//   This is exactly the idiom from PATTERNS.md section 2, used by flagships
//   5.01, 6.04, 9.02, 10.02, 13.02, 14.02.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// Sentinel rank for "this suffix has no character k positions to the right".
//   In prefix doubling, suffix i is ranked by the pair (rank[i], rank[i+k]). If
//   i+k >= n the second half is the (implicit) end-of-string '$', which sorts
//   BEFORE every real character. We represent that missing rank as -1, the
//   smallest value, so a shorter suffix correctly sorts first. (-1 also encodes
//   the sentinel uniquely because every real rank is >= 0.)
static const int SA_RANK_SENTINEL = -1;

// ---------------------------------------------------------------------------
// pack_key: combine a rank pair (r1, r2) into a single unsigned 64-bit sort key.
// ---------------------------------------------------------------------------
//   The suffixes are ordered by (r1, r2) lexicographically: first by the rank of
//   the front half, then by the rank of the back half. A radix/lexicographic
//   sort is far easier if that pair is ONE scalar key, so we pack:
//
//       key = ( (r1 + 1) << 32 ) | (r2 + 1)
//
//   We add 1 to each rank before packing so the sentinel -1 maps to 0 (the
//   smallest bucket) and stays non-negative -- letting us use a plain UNSIGNED
//   comparison/radix that needs no sign handling. Ranks are < n <= 2^31, so
//   (r+1) fits in 32 bits and the two halves never collide.
//
//   This identical packing runs on the CPU (reference_cpu.cpp builds the keys in
//   a loop) and on the GPU (build_keys_kernel calls it per thread) -> the sort
//   keys are bit-identical on both sides, which is why the resulting suffix
//   arrays match exactly.
//
//   i      : the suffix index whose key we build
//   k      : the current doubling offset (1, 2, 4, ...)
//   n      : text length (including the appended '$' sentinel)
//   rank   : rank[i] for every suffix from the previous round (length n)
//   returns: the packed 64-bit key for suffix i
HD inline std::uint64_t pack_key(int i, int k, int n, const int* rank) {
    // r1: rank of the FRONT half (always a real rank in [0, n)).
    const std::uint64_t r1 = static_cast<std::uint64_t>(rank[i] + 1);
    // r2: rank of the BACK half, k positions over -- or the sentinel if off the
    //     end. The +1 shift maps SA_RANK_SENTINEL(-1) -> 0.
    const int r2_raw = (i + k < n) ? rank[i + k] : SA_RANK_SENTINEL;
    const std::uint64_t r2 = static_cast<std::uint64_t>(r2_raw + 1);
    // High 32 bits = front rank, low 32 bits = back rank: a lexicographic pair.
    return (r1 << 32) | r2;
}

// ---------------------------------------------------------------------------
// char_to_code: map a DNA base (or sentinel) to a small integer rank.
// ---------------------------------------------------------------------------
//   The INITIAL ranks (k = 0) are just the characters themselves, but we want a
//   compact, ORDER-PRESERVING code so '$' < 'A' < 'C' < 'G' < 'T'. The sentinel
//   '$' must be strictly smallest (code 0) for the suffix array to be well
//   defined (it guarantees all suffixes are distinct). This must agree between
//   CPU and GPU, so it lives here.
//
//   Any non-ACGT/'$' character returns a code above 'T' so it sorts last and is
//   visibly "other" -- but the loader rejects such input, so in practice we only
//   ever see the five valid symbols.
HD inline int char_to_code(char c) {
    switch (c) {
        case '$': return 0;   // sentinel: strictly smallest
        case 'A': return 1;
        case 'C': return 2;
        case 'G': return 3;
        case 'T': return 4;
        default:  return 5;   // anything else sorts after T (should not occur)
    }
}

#undef HD   // keep the macro local to this header (do not leak HD to includers)
