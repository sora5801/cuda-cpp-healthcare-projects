// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 3.15 : Hi-C / 3D Genome Contact Analysis
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls ice_balance_gpu() (the ICE
//   matrix-balancing driver). kernels.cu implements the host wrapper plus the
//   device kernels. Included only by .cu translation units (it declares
//   __global__ kernels, so the plain C++ host compiler must never see it -- that
//   is why the CPU reference lives in the separate pure-C++ reference_cpu.h).
//
// THE BIG IDEA
//   The ICE hot loop is a SPARSE REDUCTION: for each iteration we need the row
//   sum of the bias-corrected matrix at every bin. We store the matrix as a flat
//   array of nonzeros (COO). We launch ONE THREAD PER NONZERO; each thread reads
//   its entry (i, j, count), forms the balanced value count/(b_i b_j), quantizes
//   it to a fixed-point integer, and atomicAdd's it into row-sum bins i and j
//   (j only if off-diagonal). This is the "parallel scatter + atomic reduce"
//   pattern from docs/PATTERNS.md (cf. flagship 11.09 k-means accumulate).
//
//   Thread t (= blockIdx.x*blockDim.x + threadIdx.x) owns nonzero t. The atomic
//   target is shared by all nonzeros in the same row, so adds COLLIDE -> atomic.
//   We accumulate INTEGERS (hic_to_fixed) so the colliding adds commute and the
//   GPU result is deterministic and byte-identical to the serial CPU sum.
//
//   The cheap O(n) bias update and renormalisation stay on the HOST (shared
//   helper ice_update_bias) -- it is tiny, and keeping it host-side guarantees
//   the GPU and CPU apply EXACTLY the same update each iteration.
//
//   Real production tools (cooltools, Juicer, hicexplorer) instead express the
//   row sum as a cuSPARSE sparse matrix-vector product (SpMV: rowsum = |M'| * 1).
//   We hand-roll it here so nothing is a black box; THEORY.md "real world" shows
//   the cuSPARSE equivalent.
//
// READ THIS AFTER: hic.h, util/cuda_check.cuh, util/timer.cuh. Then kernels.cu.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // HicMatrix, ice_update_bias (shared host update)

// ---- Device kernel -------------------------------------------------------
// ice_rowsum_kernel: accumulate the fixed-point corrected row sums.
//   One thread per nonzero entry. Launch config: block of 256 threads (a good
//   occupancy default on sm_75..sm_89), grid = ceil(nnz / 256). Touches global
//   memory only; uses atomicAdd on 64-bit unsigned integers (the deterministic
//   tally). Declared here, defined and fully commented in kernels.cu.
//     ei,ej,ecount : device arrays [nnz] -- the COO entries (struct-of-arrays so
//                    the loads coalesce nicely)
//     nnz          : number of nonzeros (guards the ragged last block)
//     bias         : device array [n] -- current per-bin bias (read-only)
//     acc          : device array [n] -- fixed-point row-sum accumulators (out)
__global__ void ice_rowsum_kernel(const int* __restrict__ ei,
                                  const int* __restrict__ ej,
                                  const double* __restrict__ ecount,
                                  long long nnz,
                                  const double* __restrict__ bias,
                                  unsigned long long* __restrict__ acc);

// ---- Host wrapper --------------------------------------------------------
// ice_balance_gpu: run `iters` ICE iterations on the GPU and fill `bias` (size n).
//   Uploads the COO entries once, then each iteration: zero the accumulators ->
//   launch ice_rowsum_kernel -> copy the row sums back -> apply the SAME host
//   ice_update_bias() the CPU reference uses. Reports the total kernel time
//   (CUDA events, summed over iterations) via *kernel_ms. Returns the final
//   convergence variance (same metric as ice_balance_cpu) so main.cu can compare.
//
//   The result `bias` must match ice_balance_cpu(m, iters, ...) bit-for-bit:
//   identical fixed-point quanta + identical host update => identical output.
double ice_balance_gpu(const HicMatrix& m, int iters,
                       std::vector<double>& bias, float* kernel_ms);
