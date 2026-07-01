// ===========================================================================
// src/kernels.cuh  --  GPU S_N transport interface (declarations + the idea)
// ---------------------------------------------------------------------------
// Project 5.6 : GPU Boltzmann Transport (Deterministic Dose)
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls solve_sn_gpu(); kernels.cu
//   implements the host driver (the source-iteration loop) plus two device
//   kernels (the per-ordinate sweep and the deterministic angular reduction).
//   Included only by .cu units (it declares __global__ kernels), so the plain
//   C++ CPU reference lives in a separate pure-C++ header (reference_cpu.h).
//
// THE BIG IDEA (the pattern for this project: PARALLELISM ACROSS ORDINATES)
//   The discrete-ordinates transport SWEEP is a spatial recurrence: within one
//   direction, cell i+1's edge flux depends on cell i's -> that axis is
//   SEQUENTIAL and cannot be parallelized cheaply. What IS embarrassingly
//   parallel is the set of ordinates: the N angles are independent inside a
//   single source iteration. So we map ONE THREAD PER ORDINATE. Thread n sweeps
//   the whole slab for direction mu_n and writes its weighted contribution into
//   its OWN private row of a [nord x ncell] scratch buffer.
//
//   The scalar flux phi[i] = sum_n (that column) is then a reduction DOWN the
//   ordinate axis. We do it in a fixed order (integer-indexed loop over n, no
//   atomics) so the sum is byte-for-byte reproducible AND identical to the CPU's
//   ordinate-ordered accumulation (PATTERNS.md §3). The host drives the outer
//   source-iteration loop, relaunching the two kernels each iteration.
//
//   This is the honest small-scale twin of a production S_N code, whose sweep is
//   a wavefront across a 2-D/3-D spatial mesh (cuSPARSE upwind triangular solve)
//   with the angular flux tensor in global memory -- see THEORY §GPU mapping.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, boltzmann_sn.h. Then
//   read kernels.cu, and compare with reference_cpu.cpp (the CPU twin).
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // SlabProblem, SnQuadrature (pure C++, safe in .cu)

// ---- Device kernels ------------------------------------------------------
// sweep_kernel: thread n owns ordinate n. It sweeps the slab for direction
//   mu[n] (reading the lagged scalar flux d_phi) and writes w[n]*psi_avg per
//   cell into row n of d_contrib ([nord * ncell], row-major). No atomics: each
//   thread writes only its own row.
__global__ void sweep_kernel(int ncell, int nord, double h,
                             const double* __restrict__ d_mu,
                             const double* __restrict__ d_w,
                             const double* __restrict__ d_sigma_t,
                             const double* __restrict__ d_sigma_s,
                             const double* __restrict__ d_q,
                             const double* __restrict__ d_phi,     // lagged scalar flux
                             double psi_left_bc, double psi_right_bc,
                             double* __restrict__ d_contrib);      // [nord*ncell] out

// reduce_kernel: thread i owns cell i. It sums the nord per-ordinate
//   contributions for its column (a fixed-order loop over n) into d_phi_new[i].
//   Deterministic (fixed summation order) and matches the CPU exactly.
__global__ void reduce_kernel(int ncell, int nord,
                              const double* __restrict__ d_contrib,
                              double* __restrict__ d_phi_new);

// ---- Host driver ---------------------------------------------------------
// solve_sn_gpu: run the full source iteration on the GPU and return the
//   converged scalar flux plus the iteration count and the measured GPU time of
//   the iteration loop (CUDA events). Mirrors solve_sn_cpu() step for step so
//   the two results agree within tolerance.
//     p     the slab problem
//     quad  the ordinate set (uploaded to the device once)
//     phi   OUT: converged scalar flux per cell (length ncell)
//     iters OUT: source iterations taken
//     kernel_ms OUT: milliseconds spent in the iteration loop's kernels
void solve_sn_gpu(const SlabProblem& p, const SnQuadrature& quad,
                  std::vector<double>& phi, int& iters, float* kernel_ms);
