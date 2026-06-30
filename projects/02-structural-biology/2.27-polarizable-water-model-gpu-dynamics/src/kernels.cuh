// ===========================================================================
// src/kernels.cuh  --  GPU induced-dipole SCF interface (declarations + idea)
// ---------------------------------------------------------------------------
// Project 2.27 : Polarizable Water Model GPU Dynamics
//
// THE BIG IDEA (PATTERNS.md: iterative relaxation + N-body field eval)
//   The self-consistent induced-dipole problem (polar.h) is solved by JACOBI
//   ITERATION: each sweep recomputes every site's dipole from the PREVIOUS
//   sweep's dipoles. Within a sweep the N updates are INDEPENDENT, so we give
//   ONE GPU THREAD PER SITE. Each thread does an O(N) loop over the other sites
//   to gather the dipole field at its own site (an N-body field evaluation),
//   re-induces its dipole, and writes it to a SECOND buffer (ping-pong, so we
//   never read a half-updated array). The host drives the sweep loop and checks
//   convergence. This is the same "relax until self-consistent" structure as the
//   PBD flagship 10.02 and the stencil solvers, but the coupling is all-to-all.
//
//   Two design choices keep the GPU result DETERMINISTIC and matched to the CPU
//   (PATTERNS.md §3):
//     * Each thread sums the field over j in the SAME fixed index order the CPU
//       uses -> identical floating-point arithmetic -> dipoles agree to round-off.
//     * The two scalar reductions (max dipole change, total energy) are done in
//       FIXED-POINT integers via atomicAdd/atomicMax, so the answer does not
//       depend on the (nondeterministic) order in which threads finish.
//
//   Included only by .cu files (it declares __global__ kernels). PolarSystem /
//   SolveResult come from the pure-C++ reference_cpu.h so this stays consistent
//   with the CPU side.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, polar.h, reference_cpu.h.
// Then read kernels.cu.
// ===========================================================================
#pragma once

#include "reference_cpu.h"   // PolarSystem, SolveResult, Site, Vec3 (pure C++)

// ---------------------------------------------------------------------------
// solve_dipoles_gpu: the host-callable "do the whole SCF on the GPU" entry.
//   Uploads the geometry, computes the permanent field once, runs the Jacobi
//   sweep loop on the device (ping-ponging two dipole buffers) until the
//   max dipole change drops below sys.tol or sys.max_iters is hit, then reduces
//   the polarization energy. Returns the converged dipoles + diagnostics in the
//   same SolveResult shape as the CPU reference, so main.cu can compare them.
//
//   sys        : the system to solve (geometry, charges, polarizabilities, tol).
//   kernel_ms  : out-param, total GPU time across all sweep kernels (CUDA events).
//
// All CUDA bookkeeping (malloc/memcpy/launch/free) is hidden inside; main.cu
// only sees this one function, mirroring integrate_gpu() in the 9.02 flagship.
// ---------------------------------------------------------------------------
SolveResult solve_dipoles_gpu(const PolarSystem& sys, float* kernel_ms);
