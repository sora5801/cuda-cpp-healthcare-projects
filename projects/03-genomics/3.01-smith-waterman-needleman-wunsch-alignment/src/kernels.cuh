// ===========================================================================
// src/kernels.cuh  --  GPU interface for the anti-diagonal wavefront fill
// ---------------------------------------------------------------------------
// Project 3.01 : Smith-Waterman / Needleman-Wunsch Alignment
//
// THE BIG IDEA (contrast this with 1.12's "independent jobs"!)
//   The DP recurrence has DATA DEPENDENCIES: H[i][j] needs H[i-1][j-1],
//   H[i-1][j], H[i][j-1]. That looks fatally serial. The trick: every cell on a
//   single ANTI-DIAGONAL d = i+j depends only on diagonals d-1 and d-2, never on
//   another cell of diagonal d. So the cells of one anti-diagonal are mutually
//   independent and can be filled in parallel. We sweep the matrix as a
//   "wavefront" of anti-diagonals; each diagonal is one parallel kernel launch.
//
//        j ->                       anti-diagonals (i+j = const) are the
//      +----------------+           independent frontiers swept in order:
//   i  | 2  3  4  5  6  |             d=2  d=3  d=4 ...
//   |  | 3  4  5  6  7  |
//   v  | 4  5  6  7  8  |           a cell on d reads only cells on d-1, d-2
//      +----------------+
//
//   This header is included by .cu units (and pulls in reference_cpu.h for the
//   scoring constants and SeqPair). main.cu calls sw_gpu().
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, reference_cpu.h.
// ===========================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "reference_cpu.h"   // MATCH/MISMATCH/GAP, SeqPair (pure C++, safe in .cu)

// Device kernel: fill every cell on ONE anti-diagonal d in parallel.
//   q,t      : device sequences (encoded 0..3), lengths m, n
//   H        : device DP matrix, (m+1)*(n+1) ints, row stride (n+1)
//   d        : the anti-diagonal index (i + j), 2 <= d <= m+n
//   i_lo     : smallest valid row i on this diagonal
//   count    : number of cells on this diagonal
__global__ void sw_diagonal_kernel(const uint8_t* __restrict__ q,
                                   const uint8_t* __restrict__ t,
                                   int* __restrict__ H, int m, int n,
                                   int d, int i_lo, int count);

// Host wrapper: upload sequences, sweep all anti-diagonals (one kernel launch
// each), copy the filled matrix back. Returns the total GPU time of the sweep.
//   sp        : the loaded sequence pair
//   H         : resized to (m+1)*(n+1); filled with the SW score matrix
//   kernel_ms : out-param, total GPU time across the wavefront sweep (ms)
void sw_gpu(const SeqPair& sp, std::vector<int>& H, float* kernel_ms);
