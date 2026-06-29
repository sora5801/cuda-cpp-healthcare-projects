// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for Tanimoto similarity search
// ---------------------------------------------------------------------------
// Project 1.12 : Molecular Fingerprint Similarity Search
//
// THE BIG IDEA
//   Comparing the query against N library fingerprints is N INDEPENDENT jobs,
//   so we give each library molecule its own GPU thread. Two CUDA features make
//   this fast and are the teaching points of this project:
//     * the query lives in CONSTANT memory (read by every thread, never written
//       during the launch) -> the constant cache broadcasts it warp-wide; and
//     * __popcll() (64-bit population count) is a SINGLE hardware instruction,
//       so each word-pair costs ~2 popcounts + 2 boolean ops.
//   A grid-stride loop lets one modest grid cover millions of molecules.
//
//   This header is included only by .cu units (and by reference_cpu.h's data
//   model, which is pure C++). main.cu calls tanimoto_gpu().
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, reference_cpu.h.
// Then read kernels.cu. The science/GPU-mapping is in ../THEORY.md.
// ===========================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "reference_cpu.h"   // FP_WORDS, FingerprintSet (pure C++, safe in .cu)

// Device kernel: out[i] = Tanimoto(query, lib_i). The query is read from the
// __constant__ symbol defined in kernels.cu (not a parameter).
//   lib : [n * FP_WORDS] row-major device array of library fingerprints
//   n   : number of library fingerprints
//   out : [n] device array of similarity scores (output)
__global__ void tanimoto_kernel(const uint64_t* __restrict__ lib, int n,
                                float* __restrict__ out);

// Host wrapper: uploads the query to constant memory and the library to global
// memory, launches the kernel, times it (CUDA events), and returns the scores.
//   fps        : the loaded dataset (query + n library fingerprints)
//   out        : resized to n; filled with per-molecule Tanimoto scores
//   kernel_ms  : out-param, GPU-measured kernel time in milliseconds
void tanimoto_gpu(const FingerprintSet& fps, std::vector<float>& out, float* kernel_ms);
