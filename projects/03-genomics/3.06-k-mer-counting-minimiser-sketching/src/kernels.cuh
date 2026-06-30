// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 3.6 : k-mer Counting & Minimiser Sketching
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls count_kmers_gpu() and
//   sketch_gpu(); kernels.cu implements both the host wrappers and the device
//   kernels. Included only by .cu translation units (it declares __global__
//   kernels, so the plain C++ host compiler must never see it -- that is why the
//   CPU reference lives in the separate pure-C++ reference_cpu.h).
//
// THE TWO GPU PATTERNS THIS PROJECT TEACHES (PATTERNS.md)
//   (1) COUNTING = "parallel insert + atomic reduce" (like 11.09 k-means tally).
//       One thread per k-mer position. Each thread canonicalises its k-mer and
//       inserts it into an OPEN-ADDRESSING HASH TABLE on the device: it linear-
//       probes for the key, claiming an empty slot with atomicCAS, then bumps the
//       slot's counter with atomicAdd. Integer counts make the atomics commute,
//       so the (key->count) MAP is order-independent => deterministic and exactly
//       matches the CPU. This is a hand-rolled version of what Gerbil/Jellyfish
//       do with a lock-free hash table.
//
//   (2) SKETCHING = "independent windows + reduction" (like a sliding-window min).
//       One thread per minimiser window. Each thread scans its w consecutive
//       k-mer hashes and writes the window minimum. The host then sorts + dedups +
//       truncates to the bottom-s sketch. (A production kernel would do the
//       window-min with warp shuffles; we keep the per-thread loop for clarity
//       and explain the warp version in THEORY.md.)
//
//   Both kernels read a single flat `bases` buffer plus precomputed per-position
//   maps so every thread does O(1) index math -- the GPU-friendly layout built in
//   main.cu and described in reference_cpu.h (ReadSet).
//
// READ THIS AFTER: kmer.h, util/cuda_check.cuh, util/timer.cuh. Then kernels.cu.
// ===========================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "reference_cpu.h"   // ReadSet, KmerCount, Sketch (shared data model)

// ---------------------------------------------------------------------------
// count_kmers_gpu: GPU k-mer counting via a device open-addressing hash table.
//   Builds the SAME ascending-by-key histogram the CPU reference produces, so
//   main.cu can compare them entry-by-entry (exact match expected).
//
//   rs        : the read set to count (flat layout; see ReadSet).
//   kernel_ms : out-param, milliseconds spent in the counting kernel (CUDA events;
//               excludes H2D/D2H copies and host-side sorting). A teaching figure.
//   returns   : histogram sorted ascending by canonical k-mer key.
// ---------------------------------------------------------------------------
std::vector<KmerCount> count_kmers_gpu(const ReadSet& rs, float* kernel_ms);

// ---------------------------------------------------------------------------
// sketch_gpu: GPU minimiser sketching.
//   One thread per minimiser window computes that window's minimum k-mer hash;
//   the host then sorts/dedups/truncates to the bottom-s MinHash sketch -- the
//   same Sketch the CPU reference builds.
//
//   rs        : the read set to sketch.
//   s         : sketch size (number of smallest distinct hashes to keep).
//   kernel_ms : out-param, milliseconds spent in the minimiser kernel.
//   returns   : the bottom-s sketch (sorted ascending, distinct).
// ---------------------------------------------------------------------------
Sketch sketch_gpu(const ReadSet& rs, int s, float* kernel_ms);
