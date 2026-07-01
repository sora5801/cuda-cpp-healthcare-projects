// ===========================================================================
// src/reference_cpu.h  --  Prototype of the CPU reference gamma computation
// ---------------------------------------------------------------------------
// Project 5.9 -- Gamma-Index Dose Comparison
//
// WHY A SEPARATE HEADER
//   The CPU reference (reference_cpu.cpp) is compiled by the plain C++ compiler
//   and must NOT see any CUDA/__global__ syntax, so its prototype cannot live in
//   kernels.cuh. Both main.cu and reference_cpu.cpp include THIS pure-C++ header
//   so they agree on the function signature.
//
// THE CONTRACT
//   gamma_map_cpu() computes the gamma index at EVERY reference voxel by an
//   exhaustive, distance-limited search over the evaluated map -- the readable,
//   obviously-correct baseline that the GPU result is checked against. The
//   actual per-pair math it calls lives in the shared header gamma_core.h, so
//   the CPU and GPU compute bit-identical values (PATTERNS.md §2).
//
//   The CPU reference exists for two reasons (CLAUDE.md §5):
//     (a) it is the readable baseline that makes the GPU speed-up legible, and
//     (b) the demo runs BOTH and asserts they agree within tolerance.
//
// READ THIS AFTER: gamma_core.h, dose_problem.h. See kernels.cuh for the GPU twin.
// ===========================================================================
#pragma once

#include <vector>
#include "dose_problem.h"   // DoseProblem (the two dose maps + criteria + grid)

// ---------------------------------------------------------------------------
// gamma_map_cpu -- compute the gamma index at every reference voxel, serially.
//   prob      : the two dose maps, grid spacing, and acceptance criteria.
//   gamma_out : resized to prob.size(); gamma_out[i] is the gamma index at
//               reference voxel i (dimensionless; <= 1 means "passes").
//
//   Algorithm (see THEORY §3): for each reference voxel, scan every evaluated
//   voxel within a physical search radius, evaluate the squared gamma term
//   (gamma_core.h), keep the running minimum, then take one sqrt at the end.
//   Complexity: O(N * K) where N = #voxels and K = #voxels inside the search
//   window -- the same work the GPU parallelizes one-thread-per-reference-voxel.
// ---------------------------------------------------------------------------
void gamma_map_cpu(const DoseProblem& prob, std::vector<float>& gamma_out);
