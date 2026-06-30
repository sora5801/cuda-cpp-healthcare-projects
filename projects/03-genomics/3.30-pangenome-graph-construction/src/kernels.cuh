// ===========================================================================
// src/kernels.cuh  --  GPU pangenome-layout interface
// ---------------------------------------------------------------------------
// Project 3.30 : Pangenome Graph Construction
//
// THE BIG IDEA  (PATTERN: parallel term evaluation + deterministic atomic reduce)
//   The ODGI-style 1-D layout is a SMACOF (Guttman-transform) optimisation. Each
//   sweep runs two tiny kernels:
//     * SCATTER kernel : one GPU thread per LAYOUT TERM. The thread reads its two
//       endpoint positions, calls the SHARED per-term contribution
//       (LO_term_numerator in layout.h -- identical to the CPU), and atomic-adds
//       the FIXED-POINT numerator (and the weight) onto BOTH endpoints'
//       accumulators. This is a SCATTER-REDUCTION: many terms touch the same node,
//       so the adds collide -> atomicAdd. Fixed-point integers make the adds
//       COMMUTE, so the reduction is deterministic AND equals the CPU bit-for-bit.
//     * APPLY kernel : one thread per NODE sets x[k] = numerator[k]/denominator[k]
//       (the weighted average -- the Guttman update). Because the scatter kernel
//       finishes (a launch boundary = barrier) before apply runs, and apply reads
//       only the accumulators, updating x[k] in place is a correct JACOBI step.
//   These two kernels run once per sweep for `iters` sweeps. SMACOF is monotone, so
//   no learning-rate schedule is needed.
//
//   kernels.cu defines the kernels + the host wrapper layout_gpu(). main.cu calls
//   layout_gpu() and compares its positions/stress against layout_cpu().
//
// WHY THIS MAPS THE CATALOG
//   Real ODGI runs this over millions of nodes and billions of terms and reports a
//   57.3x GPU speed-up. Our teaching version keeps the exact same parallel shape
//   (thread-per-term scatter + atomic node reduction) at a tiny, verifiable scale.
//
// READ THIS AFTER: layout.h, reference_cpu.h, util/cuda_check.cuh, util/timer.cuh.
// ===========================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "reference_cpu.h"   // LayoutProblem, LayoutTerm (pure C++, safe in .cu)

// SCATTER kernel: one thread per term. Reads x[i], x[j] (the sweep's source) and
// atomic-adds this term's fixed-point Guttman NUMERATOR to num[i], num[j] and its
// weight to den[i], den[j]. The accumulators are typed unsigned long long because
// CUDA's atomicAdd is defined for that type; we store the two's-complement bit
// pattern of a signed long long (signed addition is bit-identical under unsigned
// wraparound -- see kernels.cu).
__global__ void scatter_kernel(const double* __restrict__ x,
                               const LayoutTerm* __restrict__ terms, int num_terms,
                               unsigned long long* __restrict__ num,
                               unsigned long long* __restrict__ den);

// APPLY kernel: one thread per node. Sets x[k] = num[k]/den[k] (Guttman update);
// a node with no terms (den[k]==0) keeps its position.
__global__ void apply_kernel(double* __restrict__ x, int num_nodes,
                             const unsigned long long* __restrict__ num,
                             const unsigned long long* __restrict__ den);

// Host wrapper: run the full layout on the GPU. Fills `x` (final [N] positions)
// and returns the final stress; reports the GPU loop time via kernel_ms.
double layout_gpu(const LayoutProblem& p, std::vector<double>& x, float* kernel_ms);
