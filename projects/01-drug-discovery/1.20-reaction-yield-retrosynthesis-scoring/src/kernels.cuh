// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for batched route scoring
// ---------------------------------------------------------------------------
// Project 1.20 : Reaction Yield / Retrosynthesis Scoring
//
// THE BIG IDEA
//   Scoring N candidate retrosynthetic routes is N INDEPENDENT jobs (route r's
//   score depends only on route r's steps), so we give EACH ROUTE its own GPU
//   thread -- the canonical "independent jobs" pattern (PATTERNS.md sec.1, shared
//   with 1.12 Tanimoto and 12.01 spectral search). Two design choices are the
//   teaching points here:
//     * the SHARED LOGISTIC MODEL (weights w + bias b) lives in CONSTANT memory:
//       every thread reads the same few floats and none writes them, so the
//       constant cache broadcasts them warp-wide in one transaction; and
//     * the per-route math is the SAME route_score() the CPU calls (route_score.h),
//       so CPU and GPU agree to ~1e-8 (single-precision expf/FMA rounding aside),
//       well inside the verification tolerance.
//   A grid-stride loop lets one modest grid cover a batch of any size (millions
//   of routes, as a real planner's MCTS would generate).
//
//   This header declares a __global__ kernel, so it is included ONLY by .cu
//   units. main.cu calls score_routes_gpu().
//
// READ THIS AFTER: route_score.h, util/cuda_check.cuh, util/timer.cuh,
// reference_cpu.h. Then read kernels.cu. The GPU-mapping is in ../THEORY.md.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // RouteSet, ROUTE_STRIDE (pure C++, safe in .cu)

// ---- Device kernel -------------------------------------------------------
// score_kernel: out[r] = route_score(route r). The shared model (w,b) is read
// from the __constant__ symbols defined in kernels.cu, NOT passed as parameters.
//   feats : [n * ROUTE_STRIDE] row-major device array of route feature blocks
//   avail : [n] device array of per-route availability factors
//   n     : number of candidate routes
//   out   : [n] device array of route scores (output)
__global__ void score_kernel(const float* __restrict__ feats,
                             const float* __restrict__ avail,
                             int n,
                             float* __restrict__ out);

// ---- Host wrapper --------------------------------------------------------
// score_routes_gpu: uploads the model to constant memory and the route batch to
// global memory, launches score_kernel, times ONLY the kernel (CUDA events), and
// returns the per-route scores.
//   rs        : the loaded batch (routes + shared model)
//   out       : resized to rs.n; filled with per-route scores
//   kernel_ms : out-param, GPU-measured kernel time in milliseconds
void score_routes_gpu(const RouteSet& rs, std::vector<float>& out, float* kernel_ms);
