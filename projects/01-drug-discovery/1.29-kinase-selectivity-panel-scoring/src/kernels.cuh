// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for kinase panel scoring
// ---------------------------------------------------------------------------
// Project 1.29 : Kinase Selectivity Panel Scoring
//
// THE BIG IDEA
//   Scoring ONE compound against N kinases is N INDEPENDENT jobs, so we give each
//   kinase its own GPU thread. Two CUDA features carry the teaching weight here
//   (the same "score one query vs N items" pattern as flagship 1.12):
//     * the QUERY COMPOUND's feature vector lives in CONSTANT memory -- it is read
//       by every thread but never written during the launch, so the constant
//       cache broadcasts it warp-wide in one transaction instead of NFEAT global
//       loads per thread; and
//     * the per-kinase physics is the SHARED __host__ __device__ score_kinase()
//       (selectivity_core.h), so the GPU thread and the CPU reference compute
//       bit-for-bit identical integers -> exact verification.
//   A grid-stride loop lets one modest grid cover a panel of any size.
//
//   This header contains a __global__ declaration, so it is included ONLY by .cu
//   units. main.cu calls score_panel_gpu(). The pure-C++ data model it shares
//   with the CPU side lives in reference_cpu.h.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, selectivity_core.h,
// reference_cpu.h. Then read kernels.cu. Science/GPU-mapping: ../THEORY.md.
// ===========================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "reference_cpu.h"   // KinasePanel, KinasePocket, NFEAT (pure C++, safe in .cu)

// ---- Device kernel -------------------------------------------------------
// score_kinase_kernel: one thread scores one kinase. The compound is read from
// the __constant__ symbol defined in kernels.cu (NOT a parameter).
//   pockets : [n] device array of KinasePocket (req[NFEAT] + bias + id), row i = kinase i
//   n       : number of kinases in the panel
//   pK_milli: [n] output predicted affinity * 1000 (exact integer)
//   hit     : [n] output 0/1 flag, 1 if bound above the selectivity threshold
__global__ void score_kinase_kernel(const KinasePocket* __restrict__ pockets, int n,
                                    int32_t* __restrict__ pK_milli,
                                    int32_t* __restrict__ hit);

// ---- Host wrapper --------------------------------------------------------
// score_panel_gpu: uploads the compound to constant memory and the pockets to
// global memory, launches the kernel, times ONLY the kernel (CUDA events), copies
// the per-kinase results back, and returns the integer S-count (sum of `hit`).
//   panel     : the loaded problem (compound + n kinase pockets)
//   pK_milli  : resized to n; filled with per-kinase predicted pK (milli-units)
//   hit       : resized to n; filled with per-kinase 0/1 hit flags
//   kernel_ms : out-param, GPU-measured kernel time in milliseconds
//   returns   : S-count = number of kinases bound above threshold (deterministic)
int32_t score_panel_gpu(const KinasePanel& panel,
                        std::vector<int32_t>& pK_milli,
                        std::vector<int32_t>& hit,
                        float* kernel_ms);
