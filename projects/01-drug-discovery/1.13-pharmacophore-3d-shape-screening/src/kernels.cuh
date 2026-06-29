// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for 3D shape screening
// ---------------------------------------------------------------------------
// Project 1.13 : Pharmacophore & 3D Shape Screening
//
// THE BIG IDEA
//   Scoring the query against N library conformers is N INDEPENDENT jobs, so we
//   give each conformer its own GPU thread (the same "independent jobs" pattern
//   as 1.12 Tanimoto and 12.01 spectral search -- PATTERNS.md sec 1). Two CUDA
//   features make it efficient and are the teaching points of this project:
//     * the QUERY molecule lives in CONSTANT memory: every thread reads all of
//       its atoms but none writes them, and it never changes during the launch,
//       so the constant cache broadcasts it warp-wide in one transaction; and
//     * each thread runs the SAME shared physics core (shape_overlap.h) the CPU
//       reference runs, so GPU and CPU agree to ~machine precision.
//   A grid-stride loop lets one modest grid cover an arbitrarily large library.
//
//   This header is included only by .cu units. It reuses the ConformerSet /
//   Molecule data model from reference_cpu.h (pure C++, safe inside nvcc).
//
// READ THIS AFTER: shape_overlap.h, reference_cpu.h, util/cuda_check.cuh.
// Then read kernels.cu. The GPU-mapping rationale is in ../THEORY.md sec 4.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // ConformerSet, Molecule (pure C++, safe in .cu)

// ---------------------------------------------------------------------------
// shape_screen_kernel: one thread scores one library conformer against the
// query (which it reads from the __constant__ symbol defined in kernels.cu).
//   d_lib   : [n] device array of library Molecules (POD, copied as raw bytes)
//   n       : number of library conformers
//   o_aa    : the query self-overlap O_AA, precomputed once on the host and
//             passed by value (identical for every thread -> no redundant work)
//   d_out   : [n] device array of Shape Tanimoto scores (output, double)
// The thread-to-data map and launch config are documented at the definition.
// ---------------------------------------------------------------------------
__global__ void shape_screen_kernel(const Molecule* __restrict__ d_lib, int n,
                                    double o_aa, double* __restrict__ d_out);

// ---------------------------------------------------------------------------
// shape_screen_gpu: host wrapper. Uploads the query to constant memory and the
// library to global memory, precomputes O_AA on the host (a single molecule
// self-overlap -- trivially cheap and avoids every thread recomputing it),
// launches the kernel, times ONLY the kernel with CUDA events, and returns the
// per-conformer scores.
//   set        : the loaded screening problem (query + n conformers)
//   out        : resized to n; filled with per-conformer Shape Tanimoto scores
//   kernel_ms  : out-param, GPU-measured kernel time in milliseconds
// ---------------------------------------------------------------------------
void shape_screen_gpu(const ConformerSet& set, std::vector<double>& out, float* kernel_ms);
