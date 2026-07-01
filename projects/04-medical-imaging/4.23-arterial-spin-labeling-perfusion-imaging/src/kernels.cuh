// ===========================================================================
// src/kernels.cuh  --  GPU per-voxel ASL fit interface
// ---------------------------------------------------------------------------
// Project 4.23 : Arterial Spin Labeling & Perfusion Imaging
//
// THE PATTERN (independent jobs / "same model, many parameter sets")
//   Every voxel's Buxton kinetic fit is INDEPENDENT of every other voxel, so we
//   give each voxel its own GPU thread. The thread reads that voxel's measured
//   delta-M curve, runs the shared Gauss-Newton solver (asl.h) entirely in
//   registers, and writes one AslFit. There is NO inter-thread communication --
//   pure data-parallelism across voxels (docs/PATTERNS.md rows 1 & 8). The PLD
//   schedule is read by EVERY thread but never changes, so we place it in CUDA
//   CONSTANT memory, whose broadcast cache is ideal for that access pattern
//   (the same trick project 1.12 uses for its query fingerprint).
//
//   Because the fit math is the shared __host__ __device__ asl_fit_voxel(), the
//   GPU result matches the CPU reference to round-off (verified in main.cu).
//
// This header is included only by .cu files (it declares a __global__ kernel).
// The pure-C++ types it needs (AslDataset, AslFit) come from reference_cpu.h,
// which is CUDA-free and therefore safe to include from a .cu.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, asl.h. Then kernels.cu.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // HostDataset, AslDataset, AslFit (pure C++, .cu-safe)

// Maximum PLDs we support in constant memory. Multi-delay ASL protocols use a
// handful of delays (typically 5-8, sometimes up to ~12); 32 is comfortably
// above any real schedule and keeps the constant-memory footprint tiny.
static constexpr int ASL_MAX_PLDS = 32;

// ---- Host wrapper --------------------------------------------------------
// fit_gpu: fit every voxel on the GPU (one thread per voxel) and copy the
//   AslFit results back to the host.
//   ds        : the loaded study (owns host buffers; we upload signal[] + PLDs)
//   fits      : host output, resized to ds.n_voxels (output parameter)
//   kernel_ms : out-param, milliseconds spent in the fit kernel (CUDA-event timed,
//               excludes the H2D/D2H copies -- see THEORY §"honest timing")
// All CUDA bookkeeping (constant-memory upload, malloc/copy/free) is hidden here.
void fit_gpu(const HostDataset& ds, std::vector<AslFit>& fits, float* kernel_ms);
