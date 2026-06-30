// ===========================================================================
// src/kernels.cuh  --  GPU suffix-array interface (declarations + the big idea)
// ---------------------------------------------------------------------------
// Project 3.27 : Suffix Array / BWT / FM-Index Construction
//
// THE BIG IDEA  (PATTERNS.md section 1: "radix-sort based SA construction")
//   The suffix array is built by PREFIX DOUBLING. Each round we must (a) form a
//   64-bit sort key for every suffix from its rank pair, (b) SORT all n suffixes
//   by that key, and (c) RENUMBER ranks from the sorted order. Step (b) is the
//   workhorse and is *embarrassingly parallel as a sort*: every suffix is an
//   independent (key, suffix-index) record. The catalog calls for
//   thrust::sort_by_key; to keep this a NO-BLACK-BOX teaching project (and to
//   avoid a host-compiler flag thrust needs on MSVC), we hand-roll the exact
//   primitive thrust would use: a least-significant-digit (LSD) RADIX SORT by
//   64-bit key, with the suffix indices carried along as the values.
//
//   So one doubling round on the GPU is three kernel families:
//     1. build_keys_kernel : one thread per suffix -> pack_key() (sa_core.h).
//        (Identical math to the CPU reference -> identical sort order.)
//     2. radix sort by key : 8 passes of 8-bit digits; each pass is
//        histogram -> exclusive scan -> scatter. Stable, so it is a correct LSD
//        radix sort and fully deterministic.
//     3. mark + scan + write_ranks : flag where the sorted key changes, prefix-
//        sum those flags to get each suffix's new rank, scatter ranks back.
//   Repeat until all ranks are distinct. The final value array IS the SA.
//
//   kernels.cu implements all of this; main.cu calls suffix_array_gpu().
//   This header is included only by .cu files (it declares __global__ kernels),
//   which is why the shared data types live in the pure-C++ reference_cpu.h.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, sa_core.h, reference_cpu.h.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "reference_cpu.h"   // SaResult (pure C++, safe inside a .cu)

// ---- Host wrapper ---------------------------------------------------------
// suffix_array_gpu: build SA + BWT + FM count entirely via GPU prefix doubling.
//   text      : the input T$ (sentinel already appended), length n.
//   pattern   : the query whose occurrences we count (FM backward search, host).
//   kernel_ms : out-param, total milliseconds spent INSIDE the doubling kernels
//               (CUDA-event timed; excludes host<->device copies and the host-
//               side BWT/FM postprocessing).
//   Returns a SaResult whose .sa is the GPU-computed suffix array; main.cu
//   compares it field-by-field against suffix_array_cpu(). The BWT and FM count
//   are derived on the host from the GPU's SA using the SHARED helpers
//   (bwt_from_sa / fm_count) so they cannot drift from the CPU path.
SaResult suffix_array_gpu(const std::string& text, const std::string& pattern, float* kernel_ms);

// ---- Device kernels (declared here, defined in kernels.cu) ----------------
// They are declared so the structure is documented in one place; main.cu does
// not call them directly (only suffix_array_gpu does).

// build_keys_kernel: thread p packs the rank pair of the suffix sitting at the
//   current sorted slot p (val[p]) into key[p]. Building keys for the MAINTAINED
//   order (rather than the identity) carries the previous round's tie-break, so
//   a stable sort reproduces the unique suffix array exactly (matches the CPU).
//   grid/block cover n slots; thread p = blockIdx.x*blockDim.x + threadIdx.x.
__global__ void build_keys_kernel(int n, int k, const int* __restrict__ val,
                                  const int* __restrict__ rank,
                                  std::uint64_t* __restrict__ key);

// histogram_kernel: count, per 8-bit digit value, how many keys have that digit
//   at the current radix pass (shift). Uses atomicAdd into 256 integer bins --
//   integer atomics are associative, so the histogram is deterministic.
__global__ void histogram_kernel(int n, int shift, const std::uint64_t* __restrict__ key_in,
                                 unsigned int* __restrict__ hist);

// scatter_kernel: stable-scatter (key,val) into their sorted slots for this pass
//   using the exclusive-scanned histogram as the per-digit running offset.
//   Run single-threaded over n for a guaranteed-stable, deterministic order
//   (teaching clarity over speed; see THEORY "Numerical considerations").
__global__ void scatter_kernel(int n, int shift,
                               const std::uint64_t* __restrict__ key_in,
                               const int* __restrict__ val_in,
                               std::uint64_t* __restrict__ key_out,
                               int* __restrict__ val_out,
                               unsigned int* __restrict__ offset);

// flag_kernel: flag[p] = 1 if the sorted key at position p differs from p-1.
//   (flag[0] = 0.) The exclusive prefix sum of flags gives each suffix's rank.
__global__ void flag_kernel(int n, const std::uint64_t* __restrict__ sorted_key,
                            int* __restrict__ flag);

// write_ranks_kernel: scatter new ranks back to suffix indices:
//   rank[ val[p] ] = prefix[p], where val[p] is the suffix at sorted slot p.
__global__ void write_ranks_kernel(int n, const int* __restrict__ val,
                                   const int* __restrict__ prefix,
                                   int* __restrict__ rank);
