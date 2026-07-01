// ===========================================================================
// src/kernels.cuh  --  GPU compute interface for DTI fitting + tractography
// ---------------------------------------------------------------------------
// Project 4.15 : Diffusion MRI & Tractography
//
// THE BIG IDEA
//   Two GPU kernels, each an instance of the "independent jobs" pattern:
//
//   (1) fit_gpu()  -- ONE THREAD PER VOXEL. Fitting the diffusion tensor is an
//       independent least-squares problem per voxel, so thread `v` fits voxel `v`
//       by calling the shared fit_voxel() (dti_core.h). The fixed OLS operator
//       Minv is read from CONSTANT memory: every thread reads the same NPARAM*NMEAS
//       matrix but never writes it, so the constant cache broadcasts it warp-wide
//       (like the query in flagship 1.12). No shared memory, no atomics -- outputs
//       are fully independent.
//
//   (2) tract_gpu() -- ONE THREAD PER SEED. Each streamline is an independent
//       walk through the principal-direction field, so thread `s` traces seed `s`
//       by calling the shared tractography stepping (tract_core.h). The per-step
//       lookup is a trilinear interpolation of the fitted direction field -- the
//       exact operation GPU TEXTURE hardware accelerates (we do it by hand here so
//       the math is visible; THEORY explains the texture-memory upgrade).
//
//   Both kernels produce output that is byte-identical to the CPU reference (the
//   physics/stepping is shared HD code in dti_core.h / tract_core.h), so main.cu
//   verifies GPU vs CPU exactly-ish (documented tolerance in main.cu / THEORY).
//
// READ THIS AFTER: dti_core.h, tract_core.h, util/cuda_check.cuh, util/timer.cuh,
// reference_cpu.h. Then read kernels.cu. Science/GPU-mapping in ../THEORY.md.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // DwiVolume, VoxelResult, Streamline (pure C++, safe in .cu)

// ---- Kernel 1: per-voxel tensor fit -------------------------------------
// fit_kernel: thread v fits voxel v. The OLS operator Minv is read from the
// __constant__ symbol defined in kernels.cu (not a parameter).
//   signal : [nvox * NMEAS] device array of raw signals (row-major per voxel)
//   nvox   : number of voxels
//   out    : [nvox] device array of VoxelResult (output)
__global__ void fit_kernel(const double* __restrict__ signal, int nvox,
                           VoxelResult* __restrict__ out);

// Host wrapper for kernel 1: uploads Minv to constant memory + the signals to
// global memory, launches fit_kernel, times it (CUDA events), returns results.
//   vol       : the loaded DWI volume
//   Minv      : the NPARAM*NMEAS pseudo-inverse (host, row-major)
//   out       : resized to nvox; filled with per-voxel fits
//   kernel_ms : out-param, GPU-measured kernel time (ms)
void fit_gpu(const DwiVolume& vol, const std::vector<double>& Minv,
             std::vector<VoxelResult>& out, float* kernel_ms);

// ---- Kernel 2: deterministic tractography -------------------------------
// tract_kernel: thread s traces the streamline seeded at seeds[3s..3s+2]. To keep
// the output a fixed-size array (no dynamic allocation on the device), each
// streamline writes up to `cap` points into a pre-sized slot and records its
// actual length in `lengths[s]`.
//   fit     : [nvox] device array of fitted VoxelResult
//   nx,ny,nz: grid dims
//   seeds   : [3 * nseeds] device array of seed voxel coordinates
//   nseeds  : number of seeds
//   cap     : max points per streamline (slot size)
//   step,fa_min,cos_min,max_steps : stepping parameters (see tract_core.h)
//   pts     : [nseeds * cap * 3] device output; slot s starts at s*cap*3
//   starts  : [nseeds] device output; first occupied slot index for seed s
//   lengths : [nseeds] device output; contiguous point count for seed s
//   The compact streamline for seed s is pts[(s*cap + starts[s]) .. +lengths[s]].
__global__ void tract_kernel(const VoxelResult* __restrict__ fit,
                             int nx, int ny, int nz,
                             const float* __restrict__ seeds, int nseeds,
                             int cap, int max_steps,
                             float step, float fa_min, float cos_min,
                             float* __restrict__ pts, int* __restrict__ starts,
                             int* __restrict__ lengths);

// Host wrapper for kernel 2: uploads the fit + seeds, launches tract_kernel,
// times it, and returns one Streamline per seed.
//   fit       : the fitted per-voxel results (host)
//   vol       : for the grid dimensions
//   seeds     : [3 * nseeds] host seed coordinates
//   max_steps,step,fa_min,cos_min : stepping parameters
//   lines     : resized to nseeds; filled with traced polylines (output)
//   kernel_ms : out-param, GPU-measured kernel time (ms)
void tract_gpu(const std::vector<VoxelResult>& fit, const DwiVolume& vol,
               const std::vector<float>& seeds,
               int max_steps, float step, float fa_min, float cos_min,
               std::vector<Streamline>& lines, float* kernel_ms);
