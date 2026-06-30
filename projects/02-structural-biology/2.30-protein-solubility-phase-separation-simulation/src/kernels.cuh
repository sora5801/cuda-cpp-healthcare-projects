// ===========================================================================
// src/kernels.cuh  --  GPU HPS coarse-grained MD interface
// ---------------------------------------------------------------------------
// Project 2.30 : Protein Solubility & Phase Separation Simulation
//
// THE BIG IDEA
//   In a residue-level MD step, the force on every bead is INDEPENDENT once the
//   current positions are fixed -- so we give each bead its own GPU thread. Each
//   velocity-Verlet iteration is two kernel launches:
//     1. force_kernel     : thread i GATHERS the force on bead i by looping over
//        all other beads j (the all-pairs O(N^2) HPS force from hps_model.h).
//        Pure reads of x/y/z + a private write of f[i] => NO atomics, NO races.
//     2. integrate_kernel : thread i half-kicks + drifts bead i with its force.
//   Both reuse the SHARED bead_force() in hps_model.h, so the GPU runs the
//   byte-identical physics the CPU reference does, in the identical fixed pair
//   order -- which is what makes the GPU-vs-CPU summaries match (THEORY.md
//   "verify correctness").
//
//   We launch the kernels n_steps times from the host. For a tiny teaching
//   system this is "launch-bound" and may be slower than the CPU -- exactly the
//   honest-timing lesson in docs/PATTERNS.md §7; the GPU's advantage appears as
//   N grows (the O(N^2) force dominates and the per-step launch cost amortizes).
//
//   Only .cu translation units may include this header (it declares __global__
//   kernels). The CPU reference uses the pure-C++ reference_cpu.h instead.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, hps_model.h, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // System, SimSummary (pure C++, safe to include in .cu)

// ---- Device kernels (defined in kernels.cu) ------------------------------

// force_kernel: thread i computes the total force (fx,fy,fz)[i] on bead i and
//   the half-pair energy it owns, by calling the shared bead_force(). Reads all
//   positions; writes only its own force/energy slot -> embarrassingly parallel.
__global__ void force_kernel(int N,
                             const double* __restrict__ x,
                             const double* __restrict__ y,
                             const double* __restrict__ z,
                             const double* __restrict__ lam,
                             const int* __restrict__ chain,
                             SimParams p,
                             double* __restrict__ fx,
                             double* __restrict__ fy,
                             double* __restrict__ fz,
                             double* __restrict__ u_half);

// integrate_kernel: thread i advances bead i by ONE velocity-Verlet sub-update.
//   `phase` selects which half of the step (see kernels.cu): phase 0 does the
//   first half-kick + drift (+ periodic wrap); phase 1 does the second half-kick.
__global__ void integrate_kernel(int N, double dt, double mass, double box,
                                 int phase,
                                 double* __restrict__ x,
                                 double* __restrict__ y,
                                 double* __restrict__ z,
                                 double* __restrict__ vx,
                                 double* __restrict__ vy,
                                 double* __restrict__ vz,
                                 const double* __restrict__ fx,
                                 const double* __restrict__ fy,
                                 const double* __restrict__ fz);

// ---- Host wrapper --------------------------------------------------------
// run_gpu: run the whole simulation on the device and fill `out`.
//   Copies the initial System to the GPU, runs n_steps velocity-Verlet steps as
//   pairs of kernel launches, copies the final positions/velocities back, and
//   computes the SAME SimSummary the CPU does (the order parameters via the host
//   order_params() on the returned positions). main.cu calls exactly this.
//     sys       : the initial system (by value; the device gets its own copy)
//     out       : receives the final-state summary
//     kernel_ms : out-param, total GPU kernel time over all steps (ms)
void run_gpu(System sys, SimSummary& out, float* kernel_ms);
