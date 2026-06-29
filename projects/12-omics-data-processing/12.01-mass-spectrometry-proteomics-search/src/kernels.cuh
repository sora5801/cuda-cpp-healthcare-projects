// ===========================================================================
// src/kernels.cuh  --  GPU spectral-search interface
// ---------------------------------------------------------------------------
// Project 12.01 : Mass-Spectrometry Proteomics Search
//
// THE BIG IDEA (eleventh flagship pattern: BATCHED DOT-PRODUCT SCORING)
//   One observed spectrum (the query) is scored against N library spectra. Each
//   score is an independent normalized dot product, so we give each library
//   spectrum its own GPU thread. The query is read by every thread but never
//   changes -> CONSTANT memory (its cache broadcasts a value warp-wide). This is
//   the same shape as project 1.12's Tanimoto search, here with real-valued
//   intensities and a cosine score.
//
//   kernels.cu defines the kernel. main.cu calls cosine_gpu().
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // SpectralData (pure C++, safe in .cu)

// Maximum query length that fits in the constant-memory buffer (see kernels.cu).
static constexpr int MAX_BINS = 1024;

// Device kernel: scores[i] = cosine(query, lib_i). The query is read from the
// __constant__ buffer defined in kernels.cu (not a parameter).
//   lib     : [N*bins] library spectra, row-major (device)
//   libnorm : [N] precomputed L2 norms (device)
//   qnorm   : the query's L2 norm
__global__ void cosine_kernel(const float* __restrict__ lib, const double* __restrict__ libnorm,
                              int N, int bins, double qnorm, float* __restrict__ scores);

// Host wrapper: upload query (to constant memory) + library + norms, launch,
// copy the per-library scores back, report the kernel time.
void cosine_gpu(const SpectralData& s, double qnorm, const std::vector<double>& libnorm,
                std::vector<float>& scores, float* kernel_ms);
