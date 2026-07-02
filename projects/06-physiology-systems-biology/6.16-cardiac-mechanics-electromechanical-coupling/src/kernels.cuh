// ===========================================================================
// src/kernels.cuh  --  GPU ensemble-integration interface
// ---------------------------------------------------------------------------
// Project 6.16 : Cardiac Mechanics & Electromechanical Coupling
//
// THE BIG IDEA  (batch ODE: one integration point per thread)
//   The full cardiac-electromechanics solver batches a stiff ODE over every
//   Gauss point of a finite-element mesh (catalog: "batch CVODE GPU for
//   per-Gauss-point ODE"). Our reduced-scope teaching version keeps that exact
//   PARALLEL STRUCTURE but makes each "integration point" a whole 0-D virtual
//   heart: we sweep contractility x afterload and give each virtual heart its
//   own GPU thread. The thread runs the FULL multi-beat RK4 time loop in
//   registers and writes ONE PV-loop summary (CycleResult). There is no
//   inter-thread communication -- pure embarrassing parallelism over hearts.
//
//   The ODE + RK4 are shared with the CPU (cardiac.h), so the GPU results match
//   the reference to round-off. kernels.cu defines the kernel + host wrapper.
//
//   This header contains a __global__ declaration, so ONLY .cu translation
//   units may include it (the plain C++ compiler must never see __global__ --
//   that is why the CPU reference lives in the pure-C++ reference_cpu.h).
//
// READ THIS AFTER: cardiac.h, reference_cpu.h, util/cuda_check.cuh, util/timer.cuh.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // EnsembleConfig, CycleResult (pure C++, safe in .cu)

// ---- Device kernel -------------------------------------------------------
// ensemble_kernel: thread `idx` integrates virtual-heart `idx` and writes its
//   PV-loop summary. It reads its (Tref, R_sys) from the sweep via
//   member_params() (defined in reference_cpu.h, shared host+device).
//     grid  : ceil(M / THREADS_PER_BLOCK) blocks
//     block : THREADS_PER_BLOCK threads
//     thread (blockIdx.x, threadIdx.x) -> heart index idx
__global__ void ensemble_kernel(EnsembleConfig c, CycleResult* __restrict__ out);

// ---- Host wrapper --------------------------------------------------------
// integrate_gpu: allocate the device output buffer, launch one thread per
//   heart, copy the CycleResults back, and report the measured KERNEL time
//   (CUDA events) via *kernel_ms. main.cu calls exactly this.
//     c         : the ensemble configuration (passed by value into the kernel)
//     results   : host output, resized to nT*nR (output parameter)
//     kernel_ms : out-param, milliseconds spent in the kernel (not the copies)
void integrate_gpu(const EnsembleConfig& c, std::vector<CycleResult>& results,
                   float* kernel_ms);
