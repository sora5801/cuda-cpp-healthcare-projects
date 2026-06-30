// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 2.21 : Protein-Nucleic Acid Docking & Co-Folding (reduced-scope).
//
// THE BIG IDEA
//   The pose search is N INDEPENDENT scoring jobs (PATTERNS.md sec 1, the same
//   "score one query vs N items" pattern as 1.12 Tanimoto / 12.01 spectral
//   search). We give each candidate pose its own GPU thread: thread t decodes
//   flat pose index t -> (rotation, translation) with the SAME decode_pose()
//   the CPU uses, scores it with the SAME score_pose() core (docking_core.h),
//   and writes one int64 score. A grid-stride loop lets a modest grid cover an
//   arbitrarily large pose space.
//
//   Two CUDA features are the teaching points and mirror 1.12:
//     * the protein, ligand, and rotation set are read by EVERY thread but never
//       written -> read-only inputs; the tiny rotation set is an ideal candidate
//       for CONSTANT memory (broadcast cache), discussed in kernels.cu;
//     * because all arithmetic is integer, the per-thread score equals the CPU
//       score EXACTLY -- verification is integer equality, not a tolerance.
//
//   This header contains a __global__ declaration, so it is included only by
//   .cu units (the pure-C++ reference uses reference_cpu.h instead). It reuses
//   the shared types from reference_cpu.h so host and device agree on layout.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, docking_core.h,
//                  reference_cpu.h. Then read kernels.cu.
// ===========================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "reference_cpu.h"   // DockingProblem, decode_pose, score_pose (HD-safe)

// ---- Device kernel -------------------------------------------------------
// dock_kernel: one thread scores one pose (via a grid-stride loop).
//   protein/ligand : device arrays of Atom (Np / Nl entries), row-major.
//   Np, Nl         : atom counts.
//   grid           : the translational pose lattice (passed by value).
//   n_rot          : number of orientations (rotations live in constant memory).
//   sp             : scoring thresholds/weights (passed by value).
//   n_poses        : total poses = nx*ny*nz*n_rot (loop bound).
//   out            : device array [n_poses] of int64 scores (output).
// __restrict__ promises the pointers do not alias, so the compiler may cache
// the read-only protein/ligand atoms in registers across the inner loops.
__global__ void dock_kernel(const Atom* __restrict__ protein, int Np,
                            const Atom* __restrict__ ligand,  int Nl,
                            PoseGrid grid, int n_rot, ScoreParams sp,
                            long long n_poses,
                            long long* __restrict__ out);

// ---- Host wrapper --------------------------------------------------------
// dock_gpu: the host-callable "do the whole GPU search" function.
//   Uploads the rotation set to constant memory and the atoms to global memory,
//   launches dock_kernel, copies the per-pose scores back, and reports the
//   measured KERNEL time (CUDA events) via *kernel_ms. main.cu calls exactly
//   this; all CUDA bookkeeping is hidden here.
//
//   prob       : the loaded problem (protein + ligand + rots + grid + params).
//   out        : resized to prob.n_poses(); filled with per-pose int64 scores.
//   kernel_ms  : out-param, milliseconds spent in the kernel itself (not copies).
void dock_gpu(const DockingProblem& prob, std::vector<int64_t>& out,
              float* kernel_ms);
