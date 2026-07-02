// ===========================================================================
// src/kernels.cuh  --  GPU Saltelli-evaluation interface
// ---------------------------------------------------------------------------
// Project 6.26 : Virtual Population Generation & Sensitivity Analysis
//
// THE PATTERN (ensemble evaluation, cf. flagships 9.02 SEIR and 13.02 PBPK)
//   Sobol/Saltelli sensitivity needs N*(k+2) INDEPENDENT model evaluations --
//   here, N*(k+2) virtual-patient AUC computations, one per Saltelli sample.
//   Each is a self-contained forward PK solve with no cross-talk, so we assign
//   ONE GPU THREAD PER EVALUATION: thread g decodes its Saltelli (block,row),
//   builds its parameter vector, integrates the PK model, and writes one AUC.
//   The shared per-sample math in vpop.h makes the GPU array match the CPU
//   reference to round-off; the (cheap, serial) Sobol reduction that turns the
//   AUC array into indices runs on the host for both arrays (reference_cpu.cpp).
//
//   Included only by .cu translation units (it declares a __global__ kernel, so
//   the plain C++ host compiler must never see it). The CPU-side declarations
//   live in the pure-C++ reference_cpu.h.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, vpop.h. Then kernels.cu.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // VpopParams (pure C++, safe to include in a .cu)

// ---- Device kernel -------------------------------------------------------
// evaluate_kernel: thread `g` computes one Saltelli model evaluation.
//   P    : population config passed BY VALUE (small POD -> copied into the
//          kernel's parameter space; every thread reads its own copy, no global
//          memory traffic for the scalars).
//   total: N*(k+2), the number of evaluations (guards the ragged last block).
//   out  : device pointer to `total` doubles; out[g] = AUC of evaluation g.
__global__ void evaluate_kernel(VpopParams P, long total, double* __restrict__ out);

// ---- Host wrapper --------------------------------------------------------
// evaluate_gpu: allocate the device output, launch one thread per Saltelli
// evaluation, copy the AUC array back, and report the measured KERNEL time
// (CUDA events) via *kernel_ms. main.cu calls exactly this; the Sobol reduction
// is done afterward on the host (compute_sobol) for both CPU and GPU arrays.
//   out       : host output, resized to N*(k+2) (output parameter)
//   kernel_ms : out-param, milliseconds spent in the kernel itself (not copies)
void evaluate_gpu(const VpopParams& P, std::vector<double>& out, float* kernel_ms);
