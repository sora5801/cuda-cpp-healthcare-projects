// ===========================================================================
// src/kernels.cuh  --  GPU interface for the pairwise distance-matrix phase
// ---------------------------------------------------------------------------
// Project 3.8 : Multiple Sequence Alignment (MSA)
//
// THE BIG IDEA  (catalog pattern: "one CUDA thread block per pairwise alignment")
//   STAGE 1 of progressive MSA scores EVERY pair of sequences with a global
//   Needleman-Wunsch alignment. There are P = N(N-1)/2 unordered pairs and the
//   alignments are completely INDEPENDENT of one another. So we map:
//
//        pair p  ->  one CUDA THREAD BLOCK
//
//   Each block computes a single NW score by calling the SHARED recurrence
//   nw_score_core() (nw_core.h) -- the exact same integer math the CPU reference
//   runs, so the GPU score matrix is BIT-IDENTICAL to the CPU one (PATTERNS.md §2,
//   §4: exact-integer verification).
//
//   Why a block per pair rather than a thread per pair? Because each NW alignment
//   needs O(L) scratch (two rolling DP rows). Giving the pair a whole block lets
//   us park those two rows in fast on-chip SHARED memory, and lets us reuse a
//   block's worth of resources per alignment. In THIS teaching version a single
//   thread of the block actually drives the serial recurrence (the DP rows are
//   in shared memory); a production tool would also parallelise WITHIN the pair
//   via the anti-diagonal wavefront of project 3.01. We keep the within-pair work
//   serial so the across-pairs parallelism -- the catalog's point -- stays clear.
//   (See THEORY.md "GPU mapping" and the Exercises for the wavefront upgrade.)
//
//        pairs laid out as a flat list, one block each:
//          block 0 -> (0,1)   block 1 -> (0,2)  ...  block P-1 -> (n-2,n-1)
//
//   This header is included only by .cu units (it declares a __global__). main.cu
//   calls distance_matrix_gpu(); the host wrapper builds the flat pair list,
//   uploads the sequences once, launches one block per pair, and copies the score
//   matrix back. STAGES 2-3 (center-star + assembly) then run on the host from
//   that matrix, identically to the CPU path.
//
// READ THIS AFTER: nw_core.h, util/cuda_check.cuh, util/timer.cuh. Then kernels.cu.
// ===========================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "reference_cpu.h"   // SeqSet (pure C++, safe to include in a .cu)

// ---- Device kernel --------------------------------------------------------
// nw_pairs_kernel: each BLOCK scores one pair (ia, ib) from the flat pair list.
//   d_data      : all sequences concatenated, encoded 0..3 (device)
//   d_off,d_len : [n] per-sequence offset/length into d_data (device)
//   d_pair_a    : [P] first  index of each pair (device)
//   d_pair_b    : [P] second index of each pair (device)
//   num_pairs   : P, number of pairs (guards the grid)
//   max_len     : longest sequence length -> shared-memory DP-row width
//   d_score     : [n*n] output score matrix (device); we write both (a,b),(b,a)
//   n           : sequence count (row stride of d_score)
// The DP rows live in dynamically-sized shared memory (2*(max_len+1) ints),
// requested at launch. See kernels.cu for the launch configuration + reasoning.
__global__ void nw_pairs_kernel(const uint8_t* __restrict__ d_data,
                                const int* __restrict__ d_off,
                                const int* __restrict__ d_len,
                                const int* __restrict__ d_pair_a,
                                const int* __restrict__ d_pair_b,
                                int num_pairs, int max_len,
                                int* __restrict__ d_score, int n);

// ---- Host wrapper ---------------------------------------------------------
// distance_matrix_gpu: GPU twin of distance_matrix_cpu() (STAGE 1 only).
//   Uploads the sequence set once, launches one block per pair, copies the score
//   matrix back, then derives the float distance matrix on the host (identical
//   normalisation to the CPU). Fills the diagonal (self-scores) on the host since
//   those need no alignment. Returns the kernel time via *kernel_ms.
//
//   s         : the loaded sequence set
//   raw_score : [n*n] out -- NW score per pair (the bit-exact GPU result)
//   D         : [n*n] out -- normalised distance in [0,1]
//   kernel_ms : out -- milliseconds spent in the kernel (CUDA-event timed)
void distance_matrix_gpu(const SeqSet& s,
                         std::vector<int>& raw_score,
                         std::vector<double>& D,
                         float* kernel_ms);
