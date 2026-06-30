// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for per-residue MM-GBSA decomposition
// ---------------------------------------------------------------------------
// Project 2.23 : Protein-Ligand Interaction Energy Decomposition
//
// THE BIG IDEA  (PATTERNS.md sec 1: "the same work for many independent items")
//   The per-residue decomposition is M INDEPENDENT accumulations: residue m's
//   energy with the ligand, averaged over F frames, depends on nothing residue
//   m' computes. So we give each PROTEIN RESIDUE its own GPU thread; that thread
//   loops over all F frames and all L ligand atoms, accumulating its own three
//   energy components in registers, and writes one PerResidueEnergy at the end.
//
//   This is the most teachable mapping for the catalog's "N frames x M residues"
//   work: no atomics, no shared-memory reduction, no inter-thread communication.
//   Each thread's arithmetic is identical to the CPU reference's inner loops
//   (both call residue_frame_energy() from mmgbsa.h), so verification is exact up
//   to floating-point rounding. THEORY.md "GPU mapping" discusses the alternative
//   (frame x residue) tiling and the cuBLAS energy-matrix accumulation the
//   catalog mentions, and why this simpler mapping is the right teaching choice.
//
//   This header is included only by .cu units. main.cu calls decompose_gpu().
//
// READ THIS AFTER: mmgbsa.h, util/cuda_check.cuh, util/timer.cuh.
// Then read kernels.cu. The science/GPU-mapping is in ../THEORY.md.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // -> mmgbsa.h: MmgbsaSystem, PerResidueEnergy, HD core

// Device kernel: one thread per residue. Reads the uploaded system arrays and
// writes [M] per-residue decompositions. Declared here, defined in kernels.cu.
//   d_res     : [M] residue params           (device)
//   d_lig     : [L] ligand params            (device)
//   d_res_xyz : [F*M*3] residue coords       (device, flat row-major)
//   d_lig_xyz : [F*L*3] ligand  coords       (device, flat row-major)
//   F, M, L   : counts
//   cutoff2   : squared interaction cutoff (A^2)
//   d_out     : [M] output decompositions    (device)
__global__ void decompose_kernel(const ResidueParams* __restrict__ d_res,
                                 const LigandParams*  __restrict__ d_lig,
                                 const double* __restrict__ d_res_xyz,
                                 const double* __restrict__ d_lig_xyz,
                                 int F, int M, int L, double cutoff2,
                                 PerResidueEnergy* __restrict__ d_out);

// Host wrapper: uploads the system to the GPU, launches the kernel, times it
// (CUDA events), copies the [M] decompositions back, and frees device memory.
//   sys        : the loaded system (host)
//   out        : resized to M; filled with per-residue decompositions
//   kernel_ms  : out-param, GPU-measured kernel time in milliseconds
void decompose_gpu(const MmgbsaSystem& sys, std::vector<PerResidueEnergy>& out,
                   float* kernel_ms);
