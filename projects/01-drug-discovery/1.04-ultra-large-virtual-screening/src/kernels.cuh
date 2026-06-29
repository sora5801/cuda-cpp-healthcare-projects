// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for ultra-large virtual screening
// ---------------------------------------------------------------------------
// Project 1.4 : Ultra-Large Virtual Screening
//
// THE BIG IDEA
//   Screening one TARGET against N library LIGANDS is N INDEPENDENT jobs: each
//   ligand's filter + surrogate-dock score depends only on that ligand and the
//   (shared) target. So we give each ligand its OWN GPU thread. This is the same
//   "independent jobs + constant-memory query" pattern as project 1.12 (Tanimoto
//   search) -- the canonical embarrassingly-parallel GPU shape, and the literal
//   engine of billion-compound campaigns (AutoDock-GPU batches thousands of
//   ligands at once). Two CUDA features make it efficient and are the teaching
//   points here:
//     * the TARGET lives in CONSTANT memory -- read by every thread, written by
//       none during the launch -> the constant cache broadcasts it warp-wide in
//       one transaction instead of a global load per thread; and
//     * a GRID-STRIDE loop lets one modest grid cover an arbitrarily large
//       library (millions of ligands) with a fixed launch configuration.
//
//   The per-ligand math itself is NOT here -- it is in screen_core.h as the
//   shared __host__ __device__ score_ligand(), so the kernel and the CPU
//   reference run identical arithmetic. This header only declares the kernel and
//   the host wrapper. It is included only by .cu units (it names __global__).
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, reference_cpu.h.
// Then read kernels.cu. The science/GPU-mapping is in ../THEORY.md.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // LigandLibrary, Ligand, Target (pure C++, safe in .cu)

// ---------------------------------------------------------------------------
// screen_kernel: one logical thread per library ligand (via a grid-stride loop).
//   Thread (blockIdx.x, threadIdx.x) starts at i = block*blockDim + thread and
//   strides by the total thread count until i >= n. For each i it calls the
//   shared score_ligand(ligands[i], target) and writes the result to score[i].
//   The target is NOT a parameter -- it is read from the __constant__ symbol
//   defined in kernels.cu.
//     ligands : [n] device array of Ligand structs (the library)
//     n       : number of ligands
//     score   : [n] device array of per-ligand scores (output; REJECTED if the
//               ligand fails the filter cascade)
// ---------------------------------------------------------------------------
__global__ void screen_kernel(const Ligand* __restrict__ ligands, int n,
                              int* __restrict__ score);

// ---------------------------------------------------------------------------
// screen_gpu: the host-callable "do the whole GPU screen" wrapper.
//   Uploads the target to constant memory and the ligand library to global
//   memory, launches screen_kernel, times ONLY the kernel (CUDA events, not the
//   H2D/D2H copies), copies the scores back, and frees device memory. main.cu
//   calls exactly this; all CUDA bookkeeping is hidden here.
//     lib       : the loaded screening problem (target + N ligands)
//     score     : resized to lib.n(); filled with per-ligand scores
//     kernel_ms : out-param, GPU-measured kernel time in milliseconds
// ---------------------------------------------------------------------------
void screen_gpu(const LigandLibrary& lib, std::vector<int>& score, float* kernel_ms);
