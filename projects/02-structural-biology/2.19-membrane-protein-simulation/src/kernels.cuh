// ===========================================================================
// src/kernels.cuh  --  GPU MD interface (one thread per bead)
// ---------------------------------------------------------------------------
// Project 2.19 : Membrane Protein Simulation   (reduced-scope teaching version)
//
// THE BIG IDEA (PATTERN: independent per-bead force gather + Verlet)
//   Molecular dynamics spends almost all its time computing FORCES: every bead
//   feels every other bead within a cutoff. During one step the force on bead i
//   does NOT depend on the force on bead j -- so we give each bead its own GPU
//   thread, which loops over all the others and SUMS its own force. With
//   N beads and a block of B threads we launch ceil(N/B) blocks; thread
//   (blockIdx.x, threadIdx.x) owns bead i = blockIdx.x*blockDim.x + threadIdx.x.
//   The integration (Verlet half-kicks, drift, Langevin) is a second
//   independent per-bead pass. No atomics, no races: every kernel reads the
//   shared state and writes only its own bead's slot.
//
//   The per-pair force, the Verlet update, and the deterministic random kick
//   are SHARED with the CPU reference (membrane.h), so the GPU trajectory
//   matches the CPU one to ~round-off (PATTERNS.md sections 2-4). The kernel's
//   force loop walks beads/bonds in the SAME index order as the CPU's, so even
//   the floating-point sums line up.
//
//   This header contains __global__ declarations, so only .cu files may include
//   it; the pure-C++ reference uses reference_cpu.h instead.
//
// READ THIS AFTER: membrane.h, util/cuda_check.cuh, util/timer.cuh, then kernels.cu.
// ===========================================================================
#pragma once

#include "reference_cpu.h"   // SimParams, Vec3, System (all pure C++, safe in .cu)

// ---- Device kernels (defined in kernels.cu) ------------------------------

// compute_forces_kernel: thread i computes the TOTAL conservative force on bead
// i (truncated LJ over all j + bonded springs where i is an endpoint), writing
// f[i]. Identical math/order to the CPU's compute_forces(). Read-only inputs;
// each thread writes only its own f[i] -> race-free.
__global__ void compute_forces_kernel(SimParams P,
                                       const Vec3* __restrict__ pos,
                                       const int*  __restrict__ type,
                                       const int*  __restrict__ bond_i,
                                       const int*  __restrict__ bond_j,
                                       int n_bonds,
                                       Vec3* __restrict__ f);

// kick_drift_kernel: Verlet step (A). For bead i, add the Langevin force to the
// conservative force f[i], do v += f/m*dt/2, then x += v*dt. Updates pos/vel.
__global__ void kick_drift_kernel(SimParams P, int step,
                                   const Vec3* __restrict__ f,
                                   const double* __restrict__ mass,
                                   const double* __restrict__ inv_mass,
                                   Vec3* __restrict__ pos, Vec3* __restrict__ vel);

// kick_kernel: Verlet step (C). For bead i, add the Langevin force to the NEW
// conservative force f[i], then v += f/m*dt/2. Updates vel only.
__global__ void kick_kernel(SimParams P, int step,
                            const Vec3* __restrict__ f,
                            const double* __restrict__ mass,
                            const double* __restrict__ inv_mass,
                            Vec3* __restrict__ vel);

// ---- Host wrapper (defined in kernels.cu) --------------------------------
// simulate_gpu: run the full MD loop on the GPU. Copies the System to the
// device, runs `P.steps` velocity-Verlet + Langevin steps, copies the final
// positions/velocities back into `sys` in place. Returns the GPU loop time
// (CUDA events) via *kernel_ms. main.cu calls exactly this.
void simulate_gpu(const SimParams& P, System& sys, float* kernel_ms);
