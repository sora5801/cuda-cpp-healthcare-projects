// ===========================================================================
// src/kernels.cuh  --  GPU multi-scale (monodomain) simulation interface
// ---------------------------------------------------------------------------
// Project 6.14 : Multi-Scale Physiological Modeling
//
// THE BIG IDEA (two-level parallelism for scale coupling)
//   The catalog's GPU pattern is "CUDA grid over mesh elements, threads over the
//   per-element ODE RHS." On our 1-D cable that becomes: ONE GPU THREAD PER NODE.
//   Every global step, each thread:
//     (A) advances its own cell ODE (FHN via RK4)      -- the FINE scale, and
//     (B) applies the tissue diffusion stencil         -- the COARSE scale,
//   using the SAME __host__ __device__ routines the CPU reference uses
//   (multiscale.h), so GPU and CPU agree. This is operator splitting done in
//   lock-step across the whole mesh -- the GPU form of the heterogeneous
//   multiscale method (HMM). In a production VPH stack the sub-grid ODE solve
//   is SUNDIALS batch-CVODE; here we hand-roll RK4 so nothing is a black box.
//
//   Because the diffusion sub-step reads each node's neighbours, we cannot do it
//   in place (a thread might read a neighbour that another thread has already
//   overwritten). We use PING-PONG buffers (flagship 6.04 / 14.02): read the old
//   v field, write the new one, then swap. This makes the update a Jacobi sweep,
//   which is exactly what the CPU reference does with its snapshot -> results
//   match.
//
// READ THIS AFTER: multiscale.h, reference_cpu.h. kernels.cu defines the kernels.
// ===========================================================================
#pragma once

#include "reference_cpu.h"   // CableConfig, CableResult (pure C++, safe in .cu)

// Host wrapper: run the whole split-step simulation on the GPU (one thread per
//   node, ping-pong buffers, per-step reaction + diffusion kernels), copy back
//   the activation map + final field, and report the total GPU kernel time
//   (summed over all step launches) via *kernel_ms.
//
//   c         : the loaded cable configuration (by value; small + trivially copyable)
//   out       : filled with activation_time / v_final / w_final / summary metrics
//   kernel_ms : total on-device time spent in the step kernels (teaching artifact)
void simulate_gpu(const CableConfig& c, CableResult& out, float* kernel_ms);
