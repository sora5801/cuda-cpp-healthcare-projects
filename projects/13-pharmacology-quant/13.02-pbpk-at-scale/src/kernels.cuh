// ===========================================================================
// src/kernels.cuh  --  GPU PBPK population interface
// ---------------------------------------------------------------------------
// Project 13.02 : PBPK at Scale
//
// THE PATTERN (an ensemble-ODE variant, cf. 9.02)
//   Each virtual patient is an independent PBPK ODE solve, so each GPU thread
//   integrates one patient's full RK4 time loop in registers and writes one
//   exposure summary. The shared model + RNG (pbpk.h) make the GPU population
//   match the CPU reference to round-off. kernels.cu defines the kernel.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, pbpk.h, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // PbpkParams, PatientResult (pure C++, safe in .cu)

// Device kernel: thread `idx` integrates virtual patient idx -> results[idx].
__global__ void pbpk_kernel(PbpkParams P, PatientResult* __restrict__ results);

// Host wrapper: launch one thread per patient, copy the results back, time it.
void integrate_gpu(const PbpkParams& P, std::vector<PatientResult>& results, float* kernel_ms);
