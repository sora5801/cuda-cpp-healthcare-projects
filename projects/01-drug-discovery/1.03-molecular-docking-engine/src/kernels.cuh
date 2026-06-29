// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for rigid-body docking
// ---------------------------------------------------------------------------
// Project 1.3 : Molecular Docking Engine  (reduced-scope teaching version)
//
// THE BIG IDEA
//   Scoring a candidate ligand POSE is independent of every other pose, so we
//   give each pose its own GPU thread. With P poses we launch a grid-stride loop
//   over P threads; thread/iteration p decodes its pose (unrank_pose), scores it
//   with the SHARED docking_core.h::score_pose(), and the block then reduces to
//   the single best (lowest-energy) pose. Two CUDA features are the teaching
//   points:
//     * the per-pose score is a GATHER with trilinear interpolation over the
//       receptor energy grid (8 reads/atom) -- the grid lives in global memory
//       here and in TEXTURE memory in real docking engines; and
//     * finding the best pose is a MIN-REDUCTION that must also carry the WINNING
//       INDEX, done with warp shuffles + one atomic per block (deterministic
//       because ties break to the lower index, exactly like the CPU).
//
//   This header is included only by .cu units (it declares a __global__). The CPU
//   reference uses reference_cpu.h instead; both share docking_core.h so the
//   per-pose math is byte-identical. The science/GPU-mapping is in ../THEORY.md.
//
// READ THIS AFTER: docking_core.h, reference_cpu.h, util/cuda_check.cuh,
//   util/timer.cuh. Then read kernels.cu.
// ===========================================================================
#pragma once

#include "reference_cpu.h"   // DockingProblem, Pose, GridDims (pure C++, safe in .cu)

// ---------------------------------------------------------------------------
// dock_kernel: score poses and reduce to the per-block best.
//   One thread per pose (grid-stride). The block cooperatively reduces its
//   threads' (energy, pose-index) pairs to a single minimum, which thread 0
//   merges into the global best via an integer-keyed atomic CAS loop (see
//   kernels.cu for why an ordinary float atomicMin is not enough and is not
//   deterministic). Reads the ligand + grid from global memory; the grid
//   geometry and search space travel by value in the small POD structs.
//     d_grid   : [dims.count()] receptor energies (device)
//     dims     : grid geometry (by value)
//     d_lx/y/z : [n_atoms] ligand-local atom offsets (device, SoA)
//     d_w      : [n_atoms] per-atom weights (device)
//     n_atoms  : ligand atom count
//     space    : pose search space (by value)
//     n_poses  : total poses to score
//     d_best   : [1] packed best result (see kernels.cu PackedBest) -- output
__global__ void dock_kernel(const double* __restrict__ d_grid, GridDims dims,
                            const double* __restrict__ d_lx,
                            const double* __restrict__ d_ly,
                            const double* __restrict__ d_lz,
                            const double* __restrict__ d_w, int n_atoms,
                            SearchSpace space, long long n_poses,
                            unsigned long long* d_best);

// ---------------------------------------------------------------------------
// dock_gpu: host wrapper. Uploads the grid + ligand, launches dock_kernel,
//   reduces to the single best pose, copies it back, and reports the measured
//   KERNEL time (CUDA events) via *kernel_ms. main.cu calls exactly this.
//     prob       : the loaded docking problem (grid + ligand + search space)
//     out_energy : best (lowest) pose energy           (output)
//     out_index  : flat index of the best pose         (output)
//     kernel_ms  : milliseconds spent in the kernel    (output, not copies)
void dock_gpu(const DockingProblem& prob, double* out_energy,
              long long* out_index, float* kernel_ms);
