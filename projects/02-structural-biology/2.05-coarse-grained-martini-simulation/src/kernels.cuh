// ===========================================================================
// src/kernels.cuh  --  GPU coarse-grained MD interface (one thread per bead)
// ---------------------------------------------------------------------------
// Project 2.5 : Coarse-Grained / MARTINI Simulation
//
// THE BIG IDEA (the non-bonded N-body pattern)
//   The expensive part of molecular dynamics is the NON-BONDED PAIR force: each
//   bead is pushed/pulled by every other nearby bead. The total force on a bead
//   is an INDEPENDENT sum, so the natural mapping is ONE THREAD PER BEAD:
//
//       thread i  ->  loops over all j, accumulates the force on bead i,
//                     writes force[i].  (no two threads write the same slot
//                     -> no atomics, no races.)
//
//   The host drives TWO kernels per velocity-Verlet step:
//     1. kick_drift_kernel  : half-kick + drift every bead with the OLD force,
//     2. (recompute forces) force_kernel : the O(N^2) all-pairs sum,
//     3. kick_kernel        : second half-kick with the NEW force.
//   (The initial force is computed once before the loop.) Each thread calls the
//   SAME martini.h functions the CPU reference uses and sums pair contributions
//   in the SAME index order, so the GPU trajectory matches the CPU's to within
//   double round-off -> exact verification (THEORY section 6).
//
//   This header is included only by .cu files (it declares __global__ kernels),
//   so the plain C++ compiler that builds reference_cpu.cpp never sees it.
//
// READ THIS AFTER: martini.h, util/cuda_check.cuh, util/timer.cuh. Then kernels.cu.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // System, MdParams, Vec3 (pure C++, safe in .cu)

// ---- Device kernels (one thread per bead; defined in kernels.cu) ----------

// force_kernel: force[i] = sum over j != i of LJ force on bead i.
//   This is the O(N^2) hot loop -- the whole reason for using the GPU.
__global__ void force_kernel(MdParams P, const Vec3* __restrict__ pos,
                             const int* __restrict__ type,
                             Vec3* __restrict__ force);

// kick_drift_kernel: first half-kick + drift (uses the OLD force).
__global__ void kick_drift_kernel(MdParams P, Vec3* __restrict__ pos,
                                  Vec3* __restrict__ vel,
                                  const Vec3* __restrict__ force);

// kick_kernel: second half-kick (uses the NEW force).
__global__ void kick_kernel(MdParams P, Vec3* __restrict__ vel,
                            const Vec3* __restrict__ force);

// ---- Host wrapper ---------------------------------------------------------
// simulate_gpu: run the FULL velocity-Verlet time loop on the GPU. The system's
//   positions and velocities are updated in place to their final state. The
//   measured time of the kernel loop (CUDA events) is returned via *kernel_ms.
//   main.cu calls exactly this; all device bookkeeping is hidden inside.
void simulate_gpu(System& sys, float* kernel_ms);
