// ===========================================================================
// src/kernels.cuh  --  GPU interface for the Nussinov anti-diagonal wavefront
// ---------------------------------------------------------------------------
// Project 3.10 : RNA Secondary-Structure Prediction  (Nussinov base-pair DP)
//
// THE BIG IDEA (contrast with 1.12's "independent jobs")
//   The Nussinov recurrence has DATA DEPENDENCIES: M[i][j] reads M[i+1][j],
//   M[i][j-1], M[i+1][j-1] and the bifurcation cells M[i][k], M[k+1][j]. That
//   looks fatally serial. The trick (the same one as Smith-Waterman 3.01): every
//   one of those cells has a SMALLER SPAN L = j - i than the cell being written.
//   So if we fill the matrix in order of increasing span -- a "wavefront" of
//   anti-diagonals -- then all the cells of ONE span are mutually independent and
//   can be computed in parallel. Each span is one parallel kernel launch.
//
//   Upper-triangular matrix (only i <= j is used); cells grouped by span L=j-i:
//
//        j ->   0   1   2   3   4                spans are the independent
//      +-----------------------+                 frontiers swept in order:
//   i  | 0 | .  L1  L2  L3  L4 |                    L=1, L=2, L=3, L=4 ...
//   |  | 1 |     .  L1  L2  L3 |
//   v  | 2 |         .  L1  L2 |                 a cell on span L reads only
//      | 3 |             .  L1 |                 cells of span < L (smaller)
//      | 4 |                 . |
//      +-----------------------+
//
//   For a length-n sequence there are n-1 span diagonals; span L has (n-L) cells.
//   We launch one kernel per span, threads-per-cell, so the whole upper triangle
//   is filled in n-1 dependent steps instead of O(n^2) serial cells.
//
//   This header is included only by .cu units (it declares a __global__). It
//   pulls in reference_cpu.h for the shared pairing rule + recurrence and the
//   RnaSeq type. main.cu calls nussinov_gpu().
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, reference_cpu.h.
// ===========================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "reference_cpu.h"   // RnaSeq, nussinov_cell, pair_score (pure C++, safe in .cu)

// ---- Device kernel -------------------------------------------------------
// nussinov_span_kernel: fill every cell of ONE span L = j - i in parallel.
//   Thread t -> cell (i = t, j = i + L). Because all cells written here share
//   span L and read only cells of span < L (finalised by previous launches),
//   there is no read-after-write hazard within the launch -- no atomics, no
//   __syncthreads. The per-cell math is the shared nussinov_cell() so the result
//   matches the CPU exactly.
//     s     : [n] device sequence (codes 0..3)
//     M     : [n*n] device DP matrix, row stride n (M[i*n + j])
//     n     : sequence length (= row stride)
//     L     : the span being filled (1 <= L <= n-1)
//     count : number of cells on this span (= n - L); guards the last block
__global__ void nussinov_span_kernel(const uint8_t* __restrict__ s,
                                     int* __restrict__ M, int n, int L, int count);

// ---- Host wrapper --------------------------------------------------------
// nussinov_gpu: upload the sequence + a zeroed matrix, sweep spans L=1..n-1 (one
//   kernel launch each), copy the filled matrix back. Returns the total GPU time
//   of the sweep (CUDA events). main.cu then runs traceback() on this matrix.
//     r         : the loaded RNA sequence
//     M         : resized to n*n; filled with the Nussinov pair-count matrix
//     kernel_ms : out-param, total GPU time across the wavefront sweep (ms)
void nussinov_gpu(const RnaSeq& r, std::vector<int>& M, float* kernel_ms);
