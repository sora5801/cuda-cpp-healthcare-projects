// ===========================================================================
// src/kernels.cuh  --  GPU PK/PD population interface (declarations + the idea)
// ---------------------------------------------------------------------------
// Project 6.15 : PK/PD & PBPK Modeling
//
// THE PATTERN (ensemble ODE integration, PATTERNS.md §1; cf. flagships 9.02, 13.02)
//   Each virtual patient is an INDEPENDENT coupled PK/PD ODE solve, so each GPU
//   thread integrates one patient's full RK4 time loop in registers and writes
//   one PatientResult (PK exposure + PD effect). No inter-thread communication,
//   no shared memory, no atomics -- pure ensemble parallelism over the population.
//
//   The shared model + RNG (pkpd.h) make the GPU population match the CPU
//   reference to round-off (PATTERNS.md §2). kernels.cu defines the kernel and
//   the host wrapper below.
//
//   This header is included only by .cu units (it declares a __global__), so the
//   plain C++ compiler that builds reference_cpu.cpp never sees CUDA syntax -- the
//   pure-C++ config/loader prototypes live in reference_cpu.h instead.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, pkpd.h, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // PkPdParams, PatientResult (pure C++, safe in a .cu)

// ---- Device kernel -------------------------------------------------------
// pkpd_kernel: thread `idx` integrates virtual patient `idx` -> results[idx].
//   Launch config (chosen in integrate_gpu):
//     grid  = ceil(n_patients / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: idx = blockIdx.x * blockDim.x + threadIdx.x owns patient
//   idx. `P` is passed BY VALUE (a small POD struct) so every thread reads its
//   own copy from registers/constant, not global memory.
__global__ void pkpd_kernel(PkPdParams P, PatientResult* __restrict__ results);

// ---- Host wrapper --------------------------------------------------------
// integrate_gpu: launch one thread per patient, copy the results back, and
//   report the measured KERNEL time (CUDA events) via *kernel_ms. main.cu calls
//   exactly this; all device allocation/copy/free bookkeeping is hidden here.
//   results : resized to P.n_patients (output parameter).
//   kernel_ms : out-param, milliseconds spent in the kernel itself (not copies).
void integrate_gpu(const PkPdParams& P, std::vector<PatientResult>& results, float* kernel_ms);
