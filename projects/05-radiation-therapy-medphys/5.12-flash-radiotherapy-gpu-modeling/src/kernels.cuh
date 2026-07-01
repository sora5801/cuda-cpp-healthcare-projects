// ===========================================================================
// src/kernels.cuh  --  GPU ensemble-integration interface
// ---------------------------------------------------------------------------
// Project 5.12 : FLASH Radiotherapy GPU Modeling
//
// THE BIG IDEA (the ENSEMBLE ODE INTEGRATION pattern; PATTERNS.md §1 row
// "the same ODE for many parameter sets", exemplified by flagships 9.02 & 13.02)
//   Screening the FLASH effect means solving the SAME per-voxel radiation-
//   chemistry ODE for many conditions -- here a sweep of oxygen tensions x two
//   delivery modes (conventional vs UHDR/FLASH). Each solve is sequential in
//   time but INDEPENDENT of the others, so we give each ensemble member its own
//   GPU thread: the thread runs the full RK4 pulse-train integration in
//   registers/local memory and writes one VoxelResult. No inter-thread
//   communication is needed -- embarrassingly parallel over voxels, which is how
//   a real FLASH map (millions of voxels) would be computed.
//
//   The integrator (integrate_voxel) is shared with the CPU via flash.h, so the
//   GPU results match the reference to round-off. kernels.cu defines the kernel.
//
// Included only by .cu translation units (it declares a __global__ kernel, so
// the plain host C++ compiler must never see it). EnsembleConfig / VoxelResult
// come from the pure-C++ reference_cpu.h and are safe to reuse here.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, flash.h, reference_cpu.h.
// Then read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // EnsembleConfig, VoxelResult, member_job (pure C++)

// ---- Device kernel -------------------------------------------------------
// ensemble_kernel: thread `idx` integrates ensemble member idx and writes its
// VoxelResult. It reads its (pO2, delivery mode) via member_job() -- the same
// mapping the CPU uses -- then runs the full pulse-train RK4 loop from flash.h.
//   c   : the ensemble config (passed BY VALUE, so it lands in every thread's
//         local memory -- it is small and read-only, no device pointer needed)
//   out : device array [ensemble_size(c)] of results, one slot per thread
__global__ void ensemble_kernel(EnsembleConfig c, VoxelResult* __restrict__ out);

// ---- Host wrapper --------------------------------------------------------
// integrate_gpu: launch one thread per ensemble member, copy the results back to
// the host, and report the measured KERNEL time (CUDA events) via *kernel_ms.
// main.cu calls exactly this; all CUDA bookkeeping (malloc/launch/memcpy/free)
// is hidden here.
//   c        : ensemble configuration (host side)
//   results  : host output, resized to ensemble_size(c) (output parameter)
//   kernel_ms: out-param, milliseconds spent in the kernel itself (not copies)
void integrate_gpu(const EnsembleConfig& c, std::vector<VoxelResult>& results,
                   float* kernel_ms);
