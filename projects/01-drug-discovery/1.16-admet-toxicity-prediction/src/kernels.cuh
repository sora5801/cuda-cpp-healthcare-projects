// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for multi-task ADMET screening
// ---------------------------------------------------------------------------
// Project 1.16 : ADMET / Toxicity Prediction  (reduced-scope teaching version)
//
// THE BIG IDEA
//   Screening N molecules against M toxicity endpoints is an N x M grid of
//   INDEPENDENT logistic-regression predictions p_{i,t} = sigma(w_t . x_i + b_t).
//   We give each grid cell its own GPU thread -> a flat 1-D grid of N*M threads
//   (a grid-stride loop lets one modest launch cover any N). Two CUDA features
//   are the teaching points here, mirroring the 1.12 flagship:
//     * the M endpoint MODELS (weights + biases) live in CONSTANT memory: every
//       thread reads them, none writes them during the launch, and they are
//       identical across the grid -> the constant cache broadcasts them warp-wide
//       instead of forcing M*D global loads per thread.
//     * the reduction to "flagged molecules per endpoint" uses INTEGER atomicAdd
//       so it is deterministic and matches the CPU bit-for-bit (PATTERNS.md sec.3
//       -- a float atomic sum would reorder and drift).
//
//   This header is included only by .cu units (it declares __global__ kernels).
//   main.cu calls admet_screen_gpu(). The per-element math is shared with the
//   CPU via admet_core.h, so the two paths agree to ~machine precision.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, reference_cpu.h,
//   admet_core.h. Then read kernels.cu. The GPU-mapping is in ../THEORY.md.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // AdmetData, AdmetResult, ADMET_D/M (pure C++, safe in .cu)

// ---- Device kernels (declared here, defined + documented in kernels.cu) ----

// predict_kernel: one thread per (molecule, endpoint) cell. Reads molecule i's
// descriptor from global memory and endpoint t's model from CONSTANT memory,
// writes p_{i,t} into the [n*M] probs array (row-major: probs[i*M + t]).
//   desc : [n*ADMET_D] device descriptors, row-major
//   n    : molecule count
//   probs: [n*ADMET_M] device output probabilities
__global__ void predict_kernel(const double* __restrict__ desc, int n,
                               double* __restrict__ probs);

// flag_count_kernel: one thread per (molecule, endpoint) cell. Thresholds
// p_{i,t} into a 0/1 flag and atomically adds it (INTEGER add) into the
// per-endpoint counter d_flagged[t]. Integer atomics commute -> deterministic.
//   probs       : [n*ADMET_M] device probabilities (input)
//   n           : molecule count
//   d_flagged   : [ADMET_M] device counters (output, must be pre-zeroed)
__global__ void flag_count_kernel(const double* __restrict__ probs, int n,
                                  int* __restrict__ d_flagged);

// ---- Host wrapper --------------------------------------------------------
// admet_screen_gpu: run the whole GPU screen and return the SAME AdmetResult
// the CPU produces, so main.cu can verify them against each other.
//   * uploads the endpoint models to constant memory and the descriptors to
//     global memory,
//   * launches predict_kernel then flag_count_kernel,
//   * copies the probability matrix + per-endpoint counts back, and
//   * does the (tiny, serial, deterministic) "worst molecule" argmax on the host
//     from the GPU probabilities (an O(n*M) scan -- not worth a kernel, and the
//     host code is the obviously-correct version).
//
//   data       : the loaded problem (descriptors + models)
//   probs_out  : resized to n*M; filled with the GPU probability matrix
//   result_out : filled with the GPU-derived AdmetResult
//   kernel_ms  : out-param, GPU-measured time of the two kernels (CUDA events)
void admet_screen_gpu(const AdmetData& data,
                      std::vector<double>& probs_out,
                      AdmetResult& result_out,
                      float* kernel_ms);
