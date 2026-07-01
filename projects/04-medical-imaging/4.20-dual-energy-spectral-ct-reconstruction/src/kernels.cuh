// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for DECT material decomposition
// ---------------------------------------------------------------------------
// Project 4.20 : Dual-Energy / Spectral CT Reconstruction
//
// THE BIG IDEA
//   Decomposing a dual-energy sinogram is n INDEPENDENT 2x2 nonlinear solves
//   (one per bin), so we give each sinogram bin its own GPU thread. This is the
//   same "independent jobs" pattern as project 1.12 (PATTERNS.md §1), but the
//   per-item work is a full NEWTON ITERATION instead of a popcount. Two design
//   choices are the teaching points:
//     * the SpectralModel (both spectra + both attenuation curves) is read by
//       EVERY thread and never modified during the launch -> it lives in
//       __constant__ memory, whose hardware cache broadcasts one address to a
//       whole warp in a single transaction (ideal for uniform, read-only data);
//     * the per-bin math (forward model, Jacobian, Newton step) is the SHARED
//       __host__ __device__ core in dect.h, so the kernel and the CPU reference
//       run bit-identical arithmetic and verify EXACTLY.
//   A grid-stride loop lets one modest grid cover an arbitrarily large sinogram
//   (a real scan has ~10^8 bins).
//
//   Included only by .cu units. main.cu calls decompose_gpu().
//
// READ THIS AFTER: dect.h, reference_cpu.h, util/cuda_check.cuh, util/timer.cuh.
// Then read kernels.cu. The GPU mapping is in ../THEORY.md.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // DectSinogram, SpectralModel, solver constants

// ---------------------------------------------------------------------------
// decompose_gpu: host wrapper around the decomposition kernel.
//   Uploads the spectral model to constant memory and the measurements to global
//   memory, launches one-thread-per-bin, times ONLY the kernel (CUDA events),
//   and copies back the recovered path lengths + per-bin iteration counts.
//     sino      : the loaded dual-energy sinogram (n bins of m_lo, m_hi)
//     sm        : the shared scanner physics (spectra + attenuation curves)
//     t1,t2     : resized to n; filled with recovered basis-material path lengths
//     iters     : resized to n; per-bin Newton iteration counts
//     kernel_ms : out-param, GPU-measured kernel time in milliseconds
// ---------------------------------------------------------------------------
void decompose_gpu(const DectSinogram& sino, const SpectralModel& sm,
                   std::vector<double>& t1, std::vector<double>& t2,
                   std::vector<int>& iters, float* kernel_ms);
