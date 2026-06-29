// ===========================================================================
// src/kernels.cuh  --  GPU ensemble-integration interface
// ---------------------------------------------------------------------------
// Project 9.02 : Large-Scale Compartmental & Metapopulation Models
//
// THE BIG IDEA (eighth flagship pattern: ENSEMBLE ODE INTEGRATION)
//   Uncertainty quantification needs the SAME ODE solved for thousands of
//   parameter samples. Each solve is sequential in time but independent of the
//   others, so we give each ensemble member its own GPU thread: the thread runs
//   the full RK4 time loop (in registers) and writes one summary result. This is
//   how Monte Carlo / parameter-sweep epidemic studies scale on the GPU.
//
//   The RK4 integrator is shared with the CPU (seir.h), so the GPU results match
//   the reference to round-off. kernels.cu defines the kernel.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, seir.h, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // EnsembleConfig, MemberResult (pure C++, safe in .cu)

// Device kernel: thread `idx` integrates ensemble member idx and writes its
// MemberResult. Reads its (beta,gamma) from the sweep via member_params().
__global__ void ensemble_kernel(EnsembleConfig c, MemberResult* __restrict__ out);

// Host wrapper: launch one thread per member, copy the results back, time it.
void integrate_gpu(const EnsembleConfig& c, std::vector<MemberResult>& results, float* kernel_ms);
