// ===========================================================================
// src/kernels.cuh  --  GPU SPME reciprocal-energy interface
// ---------------------------------------------------------------------------
// Project 1.2 : Particle-Mesh Ewald Electrostatics
//
// THE BIG IDEA (two flagship patterns combined)
//   The reciprocal-space PME energy is computed by a four-stage GPU pipeline:
//
//     (1) SPREAD   particle -> mesh : each atom scatters its charge onto an
//                  order^3 block of grid points using B-spline weights. Many
//                  atoms hit the same grid point, so this is an ATOMIC SCATTER
//                  (atomicAdd). To make it DETERMINISTIC and bit-match the CPU,
//                  we accumulate in FIXED-POINT integers (pme.h) -- the same
//                  trick as the k-means flagship 11.09.
//     (2) FFT      the 3D real charge grid via cuFFT (R2C). We USE THE LIBRARY,
//                  but kernels.cu documents exactly what it computes and the
//                  layout it expects -- NOT a black box (like flagship 8.03).
//     (3) CONVOLVE multiply each reciprocal-grid point by the Ewald influence
//                  function B(m)C(m) and form its energy contribution
//                  influence[m]*|F[m]|^2 (with the Hermitian multiplicity).
//     (4) REDUCE   sum the per-bin energies. We copy them to the host and sum in
//                  the SAME order as the CPU reference -> deterministic stdout,
//                  exact GPU==CPU agreement (PATTERNS.md §3).
//
//   kernels.cu defines the kernels + the pme_recip_gpu() wrapper. main.cu calls
//   pme_recip_gpu() and compares its energy to pme_recip_cpu() / the direct sum.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, pme.h, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // System, PmeParams (pure C++, safe in .cu)

// ---- Device kernels (documented fully in kernels.cu) ----------------------

// SPREAD: one thread per atom. Each atom computes its B-spline weights and
// atomically adds its fixed-point charge contributions onto the order^3 grid
// stencil (wrapped periodically). grid_fixed is unsigned long long (signed bits).
__global__ void spread_kernel(const double* __restrict__ x,
                              const double* __restrict__ y,
                              const double* __restrict__ z,
                              const double* __restrict__ q,
                              int n, int K, double box,
                              unsigned long long* __restrict__ grid_fixed);

// CONVERT: one thread per grid cell. Reinterpret the fixed-point accumulator as
// a signed integer and divide by the scale to recover the real charge density,
// written as the cufftReal input for the FFT.
__global__ void fixed_to_real_kernel(const unsigned long long* __restrict__ grid_fixed,
                                     int total, float* __restrict__ grid_real);

// CONVOLVE+ENERGY: one thread per reciprocal-grid bin. Compute
//   e[i] = mult[i] * influence[i] * |F[i]|^2
// where F is the cuFFT R2C output (float2), `influence` is B(m)C(m) (uploaded),
// and `mult` is 1 or 2 (Hermitian half-spectrum multiplicity). The per-bin
// energies are summed on the host for a deterministic result.
__global__ void energy_kernel(const float2* __restrict__ F,
                              const double* __restrict__ influence,
                              const double* __restrict__ mult,
                              int total, double* __restrict__ e);

// ---- Host wrapper ---------------------------------------------------------
// pme_recip_gpu: run the full reciprocal-space pipeline on the GPU and return
// E_recip. `influence` and `mult` are precomputed on the host (build_influence
// + the Hermitian multiplicity) and uploaded; passing them in keeps the GPU and
// CPU using identical coefficients. *kernel_ms receives the measured GPU time of
// the spread+FFT+convolve stages (CUDA events). The final reduction is on the
// host (summed in CPU order) so the energy is deterministic and CPU-matching.
double pme_recip_gpu(const System& s, const PmeParams& p,
                     const std::vector<double>& influence,
                     const std::vector<double>& mult,
                     float* kernel_ms);
