// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for batched 3D-CNN affinity scoring
// ---------------------------------------------------------------------------
// Project 1.15 : Protein-Ligand Binding Affinity Scoring (ML)
//
// THE BIG IDEA (the two stacked CUDA patterns this project teaches)
//   * BATCH over complexes: scoring N docked poses are N INDEPENDENT jobs, so we
//     give each complex its OWN THREAD BLOCK. This is the real-world "rescore
//     millions of docking poses" workload -- pure data parallelism across poses.
//   * STENCIL within a complex: the 3D convolution is the classic per-output-voxel
//     stencil. Inside a block, the threads cooperate over the GRID^3 voxels of one
//     complex, using a grid buffer in global memory and a shared-memory reduction
//     for the global-average pool.
//
//   So the launch is <<<n_complexes, BLOCK>>>: blockIdx.x selects the complex,
//   threadIdx.x is a worker that owns a stride of voxels. See kernels.cu for the
//   two device passes (voxelize, then conv+ReLU+pool+dense) and ../THEORY.md
//   "GPU mapping" for the occupancy / memory reasoning.
//
//   This header is included only by .cu units (it declares __global__ kernels).
//   The CPU reference uses reference_cpu.h instead. Both pull the shared per-
//   element math from scoring_core.h so results match (PATTERNS.md sec.2).
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, scoring_core.h.
// Then read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // ComplexSet, Atom (pure C++, safe to include in .cu)

// ---------------------------------------------------------------------------
// score_gpu: the host-callable "score the whole batch on the GPU" function.
//   It uploads the ragged atom array + offsets, allocates one reusable voxel/
//   feature grid per resident block, launches the batched scoring kernel, times
//   ONLY the kernel (CUDA events), and copies the per-complex pKd back.
//
//   cs        : the loaded batch (n complexes, CSR atom layout)
//   out       : resized to cs.n; filled with predicted pKd per complex (output)
//   kernel_ms : out-param, GPU-measured kernel time in milliseconds
//
//   main.cu calls exactly this; all CUDA bookkeeping is hidden inside.
// ---------------------------------------------------------------------------
void score_gpu(const ComplexSet& cs, std::vector<double>& out, float* kernel_ms);
