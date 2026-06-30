// ===========================================================================
// src/kernels.cuh  --  GPU pharmacophore-screen interface (declarations + idea)
// ---------------------------------------------------------------------------
// Project 2.33 : Structure-Based Pharmacophore Modeling from MD Ensembles
//
// THE BIG IDEA  (PATTERNS.md §1: score ONE query vs N independent items)
//   One query pharmacophore is scored against N library molecules. Each score is
//   an independent Gaussian-overlap Tanimoto (pharmacophore.h, score_molecule),
//   so we give each library molecule its OWN GPU THREAD. The query is read by
//   every thread but never changes during the launch -> we keep it in CONSTANT
//   memory, whose hardware cache broadcasts each query feature warp-wide for free.
//   This is the same structure as project 1.12 (Tanimoto fingerprints) and 12.01
//   (spectral search), here over 3-D feature points instead of bit strings.
//
//   The variable-length library feature sets are passed as the flat CSR layout
//   from ScreenData (one coalesced buffer + an offset array), so the kernel reads
//   molecule k's features from lib_feats[ offset[k] .. offset[k+1] ).
//
//   kernels.cu implements the device kernel + the host wrapper; main.cu calls
//   screen_gpu(). Only .cu files may include this header (it declares __global__).
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, pharmacophore.h,
//                  reference_cpu.h. Then read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // ScreenData, Feature (pure C++, safe inside .cu)

// Maximum query-pharmacophore size that fits the constant-memory buffer (see
// kernels.cu). A real pharmacophore has ~4-10 features; 64 is generous headroom
// and still a tiny 64*24 B = 1.5 KB of the 64 KB constant bank.
static constexpr int MAX_QUERY_FEATS = 64;

// ---- Device kernel -------------------------------------------------------
// One thread computes the Tanimoto color score for one library molecule.
//   grid  : ceil(N / THREADS_PER_BLOCK) blocks
//   block : THREADS_PER_BLOCK threads
//   thread (blockIdx.x, threadIdx.x) -> library molecule index k
//
//   lib_feats : [total_feats] all molecules' Feature points concatenated (device)
//   offset    : [N+1] CSR offsets; molecule k = lib_feats[offset[k]..offset[k+1])
//   N         : number of library molecules (guards the ragged last block)
//   n_query   : number of query features (the query lives in __constant__ memory)
//   self_qq   : O_qq, the query self-overlap, precomputed once on the host
//   scores    : [N] output Tanimoto color scores (device)
__global__ void screen_kernel(const Feature* __restrict__ lib_feats,
                              const int* __restrict__ offset,
                              int N, int n_query, double self_qq,
                              float* __restrict__ scores);

// ---- Host wrapper --------------------------------------------------------
// screen_gpu: do the whole GPU screen. Uploads the query to constant memory,
//   copies the flat library + offsets to the device, launches screen_kernel,
//   copies the per-molecule scores back, and reports the measured KERNEL time
//   (CUDA events) via *kernel_ms. main.cu calls exactly this.
//
//   s         : the loaded screening problem (query + flat library + offsets)
//   self_qq   : O_qq, the query self-overlap (same value for every molecule)
//   scores    : host output, resized to N (output parameter)
//   kernel_ms : out-param, milliseconds spent in the kernel itself (not copies)
void screen_gpu(const ScreenData& s, double self_qq,
                std::vector<float>& scores, float* kernel_ms);
