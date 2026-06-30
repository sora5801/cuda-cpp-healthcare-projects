// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for batched variant-effect scoring
// ---------------------------------------------------------------------------
// Project 3.19 : Variant Effect / Pathogenicity Prediction
//
// THE BIG IDEA
//   Scoring N variants is N INDEPENDENT model forward-pass PAIRS (one for the
//   reference window, one for the alternate window). The variants do not
//   interact, so we give each variant its own GPU thread -- the textbook
//   "batched inference over many inputs" pattern that real tools (AlphaMissense,
//   Enformer, ESM-1v) run at the scale of tens of millions of variants.
//
//   Two CUDA features carry the teaching weight, and they mirror project 1.12:
//     * the MODEL WEIGHTS live in CONSTANT memory. Every thread reads the same
//       fixed weights and none writes them, so the constant cache broadcasts one
//       address to a whole warp in a single transaction -- far cheaper than each
//       thread streaming the weights from global memory. The weight set is a
//       fixed-size struct (VepModel), which is exactly what constant memory wants.
//     * a GRID-STRIDE LOOP lets one modest grid cover an arbitrarily large batch.
//
//   This header is included only by .cu units (it declares a __global__). main.cu
//   calls score_variants_gpu(). The per-variant math itself is in vep_model.h,
//   shared verbatim with the CPU reference so the results match (THEORY "verify").
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, reference_cpu.h,
//   vep_model.h. Then read kernels.cu. Science/GPU-mapping is in ../THEORY.md.
// ===========================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "reference_cpu.h"   // VariantSet, VepModel (pure C++/HD, safe in .cu)

// ---- Device kernel -------------------------------------------------------
// score_variants_kernel: one logical thread per variant. The model is read from
// the __constant__ symbol defined in kernels.cu (NOT passed as a parameter).
//   ref : [n * VEP_WINDOW] device array of reference-window base codes (int8)
//   alt : [n * VEP_WINDOW] device array of alternate-window base codes (int8)
//   n   : number of variants
//   out : [n] device array of delta scores  effect = score(alt) - score(ref)
__global__ void score_variants_kernel(const int8_t* __restrict__ ref,
                                      const int8_t* __restrict__ alt,
                                      int n,
                                      double* __restrict__ out);

// ---- Host wrapper --------------------------------------------------------
// score_variants_gpu: uploads the model to constant memory and the windows to
// global memory, launches the kernel, times ONLY the kernel (CUDA events), and
// returns the per-variant delta scores.
//   m         : the fixed model (uploaded to __constant__ memory)
//   vs        : the loaded variant batch (ref/alt windows live on the host)
//   out       : resized to vs.n; filled with per-variant delta scores (double)
//   kernel_ms : out-param, GPU-measured kernel time in milliseconds
void score_variants_gpu(const VepModel& m, const VariantSet& vs,
                        std::vector<double>& out, float* kernel_ms);
