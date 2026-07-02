// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 6.20 : Coronary Autoregulation & Microvascular Perfusion
//
// THE GPU IDEA (PATTERNS.md §1: "iterative CG with sparse SpMV")
//   The heavy step is solving the sparse SPD system L p = b for nodal pressures,
//   once per autoregulation iteration. We solve it with CONJUGATE GRADIENT, whose
//   inner loop is dominated by ONE sparse matrix-vector product (SpMV) y = L p
//   plus a couple of dot-products and AXPYs. On the GPU:
//     * L is stored in CSR (compressed sparse row). Each ROW = one network node;
//       its nonzeros are the diagonal (sum of incident conductances) and one
//       -G_ij per incident segment.
//     * The SpMV kernel assigns ONE THREAD PER ROW (node): thread i walks its
//       CSR row and accumulates y_i. This is exactly what cuSPARSE's SpMV does
//       under the hood; we hand-roll it here so nothing is a black box (the
//       catalog names cuSPARSE -- THEORY §real-world shows the cuSPARSE call).
//     * Dot-products are block reductions summed by a tiny final kernel, in a
//       FIXED order so the result is deterministic (PATTERNS.md §3).
//   Conductances and the autoregulation radius update reuse coronary.h, so GPU
//   and CPU compute identical per-vessel physics.
//
//   Only the host-callable driver is declared here; the __global__ kernels are
//   internal to kernels.cu. This header is included by main.cu (host C++/nvcc).
//
// READ THIS AFTER: coronary.h, reference_cpu.h. Implementation: kernels.cu.
// ===========================================================================
#pragma once

#include "reference_cpu.h"   // Network, Solution (shared POD types)

// ---------------------------------------------------------------------------
// solve_gpu(net, n_autoreg, cg_tol, cg_max_iter, out, kernel_ms)
//   GPU counterpart of solve_cpu(): runs the identical autoregulation outer loop
//   and CG inner solves entirely on the device, then copies the final pressures
//   and flows back into `out`.
//
//   net         : network (radii are MUTATED across autoregulation, mirroring
//                 the CPU path so the two end on the same geometry)
//   n_autoreg   : number of outer autoregulation iterations (>= 1)
//   cg_tol      : CG relative-residual stopping tolerance
//   cg_max_iter : CG iteration cap
//   out         : receives final p[n_nodes], q[n_segs], and last CG stats
//   kernel_ms   : (out) total GPU time for all solves, measured with CUDA events
//                 (teaching artifact only, printed to stderr)
// ---------------------------------------------------------------------------
void solve_gpu(Network& net, int n_autoreg, double cg_tol, int cg_max_iter,
               Solution& out, float* kernel_ms);
