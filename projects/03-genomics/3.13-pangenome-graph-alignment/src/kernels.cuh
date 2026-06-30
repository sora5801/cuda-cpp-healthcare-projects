// ===========================================================================
// src/kernels.cuh  --  GPU interface for the per-node anti-diagonal wavefront
// ---------------------------------------------------------------------------
// Project 3.13 : Pangenome Graph Alignment
//
// THE BIG IDEA (how graph alignment becomes a GPU computation)
//   Linear Smith-Waterman fills ONE matrix as a wavefront of anti-diagonals: the
//   cells on diagonal d = i+j depend only on diagonals d-1 and d-2, so they are
//   mutually independent and fill in parallel (flagship 3.01). A pangenome graph
//   is a DAG of nodes; aligning to it fills ONE BLOCK PER NODE. We process nodes
//   in TOPOLOGICAL order (every predecessor finished first), and for each node we
//   run the SAME anti-diagonal wavefront over its (qlen+1) x (Lv+1) block.
//
//   The only cross-node coupling is a node's FIRST content column (j = 1), whose
//   "diagonal" and "left" neighbours live in the LAST column of the node's
//   predecessors (max over predecessors -- see reference_cpu.cpp
//   first_column_neighbours). We precompute those two short boundary vectors on
//   the host between launches and pass them to the kernel as diag_in[] / left_in[]
//   -- so the kernel itself never has to chase irregular graph pointers (that is
//   the part that makes graph alignment "hard to parallelise"; we localise it to
//   a tiny, cheap host reduction and keep the heavy DP regular and coalesced).
//
//        per node v:                 anti-diagonals d = i+j sweep the block:
//      j -> 0  1  2  3  (= Lv)         d=1 d=2 d=3 ...   each cell on d reads
//    i 0 [ boundary col j=0 ]          only cells on d-1, d-2 (already final),
//    | 1 |  .  .  .  .  |              so the cells of one diagonal are
//    v 2 |  .  .  .  .  |              independent -> one GPU thread per cell.
//      3 |  .  .  .  .  |
//
//   diag_in[i], left_in[i] feed column j=1; the kernel does ordinary SW elsewhere.
//
//   This header is included only by .cu units (it declares a __global__). It
//   pulls in reference_cpu.h for the shared scoring + cell_score + data structs.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, reference_cpu.h.
// ===========================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "reference_cpu.h"   // Problem, GraphDP, cell_score, scoring constants

// ---------------------------------------------------------------------------
// graph_sw_diagonal_kernel: fill every cell on ONE anti-diagonal d of ONE node's
// block, in parallel. One thread per cell.
//   q          : device query (codes 0..3), length qlen
//   node_seq   : device pointer to THIS node's segment bases (codes 0..3)
//   H          : device DP buffer (all node blocks); this node's block starts at
//                base and has row stride W = L+1
//   base, L, W : this node's block offset, segment length, and row stride
//   qlen       : query length (rows = qlen+1)
//   diag_in    : [qlen+1] precomputed diagonal-incoming scores for column j=1
//   left_in    : [qlen+1] precomputed left-incoming   scores for column j=1
//   d          : the anti-diagonal index (i + j), 2 <= d <= qlen + L
//   i_lo       : smallest valid row i on this diagonal
//   count      : number of cells on this diagonal
//
//   Thread k -> i = i_lo + k, j = d - i. Because every cell written lies on
//   diagonal d and only READS cells on d-1, d-2 (finalised by earlier launches)
//   plus the precomputed boundary vectors, there is no read-after-write hazard
//   within a launch -- no atomics or __syncthreads needed. Calls cell_score()
//   (shared with the CPU) so the block matches the reference exactly.
// ---------------------------------------------------------------------------
__global__ void graph_sw_diagonal_kernel(const uint8_t* __restrict__ q,
                                         const uint8_t* __restrict__ node_seq,
                                         int* __restrict__ H,
                                         int base, int L, int W, int qlen,
                                         const int* __restrict__ diag_in,
                                         const int* __restrict__ left_in,
                                         int d, int i_lo, int count);

// ---------------------------------------------------------------------------
// graph_sw_gpu: the host-callable "do the whole alignment on the GPU" function.
//   Uploads the query + graph once, then for each node (topological order) it
//   (a) reduces the node's predecessor last-columns into diag_in/left_in on the
//   host, uploads them, and (b) sweeps the node's block as an anti-diagonal
//   wavefront (one kernel launch per diagonal). Finally copies the full filled
//   H back so the host can trace the path. Returns total GPU sweep time (ms).
//
//   p         : the loaded problem (query + graph)
//   dp        : filled with the SAME flat block layout as graph_sw_cpu()
//   kernel_ms : out-param, total GPU time across all diagonals of all nodes (ms)
// ---------------------------------------------------------------------------
void graph_sw_gpu(const Problem& p, GraphDP& dp, float* kernel_ms);
