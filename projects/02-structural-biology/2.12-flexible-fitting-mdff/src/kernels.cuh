// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 2.12 : Flexible Fitting / MDFF
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls fit_gpu(); kernels.cu
//   implements the host wrapper plus the device kernel. Included only by .cu
//   translation units (it declares a __global__ kernel, so the plain C++
//   compiler must never see it -- that is why the CPU reference's prototypes
//   live in the separate pure-C++ reference_cpu.h).
//
// THE BIG IDEA  (one GPU thread per ATOM, density read-only, Jacobi iteration)
//   MDFF moves every atom along the local density gradient (plus a restraint).
//   Given the fixed density map and the CURRENT positions, each atom's update is
//   INDEPENDENT of the others, so we assign ONE THREAD PER ATOM:
//
//       atom index  i = blockIdx.x * blockDim.x + threadIdx.x
//
//   Each iteration the kernel reads x_old[i] (and the shared, read-only density),
//   computes the trilinear gradient there, and writes x_new[i]. We DOUBLE-BUFFER
//   x_old / x_new and swap pointers between iterations (the Jacobi / ping-pong
//   pattern, like projects 9.02 and 10.02): because every thread reads only from
//   x_old, there are NO data races and NO atomics. The density map lives once in
//   global memory and is shared by all threads across all iterations.
//
//   The per-atom math itself is NOT here -- it is in mdff.h's mdff_step_atom(),
//   a __host__ __device__ function the kernel and the CPU reference both call, so
//   the two paths are byte-for-byte identical (exact verification).
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, mdff.h. Then kernels.cu.
// ===========================================================================
#pragma once

#include <vector>

#include "mdff.h"   // Vec3, MdffParams, mdff_step_atom (shared host/device core)

// ---- Device kernel -------------------------------------------------------
// mdff_step_kernel: advance every atom by ONE fitting iteration.
//   __global__ marks an entry point launched from host, run on device.
//     rho   : device pointer to the density map [nx*ny*nz], READ-ONLY this run
//             (__restrict__ promises it does not alias the position arrays)
//     x_old : device pointer to current atom positions [natoms] (read)
//     x_ref : device pointer to restraint anchors      [natoms] (read)
//     x_new : device pointer to next atom positions    [natoms] (write)
//     P     : problem parameters, passed BY VALUE so each thread has its own copy
//             in registers (small struct; no pointer chasing on the device)
//   Thread-to-data map: i = blockIdx.x*blockDim.x + threadIdx.x owns atom i; the
//   ragged last block is guarded by `if (i < P.natoms)`.
__global__ void mdff_step_kernel(const double* __restrict__ rho,
                                 const Vec3* __restrict__ x_old,
                                 const Vec3* __restrict__ x_ref,
                                 Vec3* __restrict__ x_new,
                                 MdffParams P);

// ---- Host wrapper --------------------------------------------------------
// fit_gpu: run the whole MDFF fit on the GPU and return the fitted positions.
//   Allocates device buffers, uploads the density + atoms once, launches
//   mdff_step_kernel for P.iters iterations with pointer double-buffering, then
//   copies the final positions back. Reports the measured KERNEL time (the sum
//   over all iteration launches, via CUDA events) through *kernel_ms. main.cu
//   calls exactly this; all CUDA bookkeeping is hidden here.
//
//     P       : problem parameters (grid, weights, step, iters)
//     rho      : host density map [nx*ny*nz]
//     x0       : host starting positions [natoms]
//     x_ref    : host restraint anchors [natoms]
//     out      : host output, resized to natoms (the final fitted positions)
//     kernel_ms: out-param, milliseconds spent in the kernels (not H2D/D2H copies)
void fit_gpu(const MdffParams& P, const std::vector<double>& rho,
             const std::vector<Vec3>& x0, const std::vector<Vec3>& x_ref,
             std::vector<Vec3>& out, float* kernel_ms);
