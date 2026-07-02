// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 6.21 : Microcirculation & Oxygen Transport
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls solve_gpu(); kernels.cu
//   implements both the host wrapper and the device kernel. Included only by
//   .cu translation units (it contains a __global__ declaration, so the plain
//   C++ compiler must never see it -- that is why the problem containers and the
//   shared math live in the pure-C++ headers reference_cpu.h / oxygen.h).
//
// THE BIG IDEA (pattern: INDEPENDENT JOBS + GATHER, PATTERNS.md section 1)
//   The tissue PO2 field is a sum over sources evaluated at every grid point:
//       PO2_i = po2_inflow + sum_j q_j * G(|x_i - x_j|) - consumption.
//   Every grid point i is INDEPENDENT of every other, so we assign ONE GPU THREAD
//   PER GRID POINT. Each thread loops over all N_src sources (a "gather" of every
//   source's contribution) and writes one PO2 value. This is the O(N_grid*N_src)
//   direct sum -- the honest baseline that a fast multipole method accelerates
//   (see THEORY.md). The sources are read by EVERY thread, so we stage them in
//   fast on-chip SHARED MEMORY in tiles to cut global-memory traffic.
//
//   Both CPU and GPU call the SAME solve_point() (reference_cpu.h / oxygen.h),
//   with the source loop in a fixed order, so their double-precision results
//   agree to round-off.
//
// READ THIS AFTER: oxygen.h, reference_cpu.h, util/cuda_check.cuh,
//   util/timer.cuh. Then read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // TissueGrid, OxySource, solve_point (pure C++, safe in .cu)

// ---- Device kernel -------------------------------------------------------
// solve_field_kernel: thread `idx` computes the PO2 at grid point idx.
//   grid          : the tissue lattice + physiology (passed by value; POD, tiny).
//   sources       : device pointer to n_src OxySource records (__restrict__:
//                   promises no aliasing so the compiler can cache freely).
//   n_src         : number of capillary-segment sources.
//   po2_out       : device pointer to grid_size(grid) doubles (the result field).
// The kernel cooperatively loads sources into shared memory in tiles, then each
// thread superposes them via solve_point()'s math. See kernels.cu for detail.
__global__ void solve_field_kernel(TissueGrid grid,
                                    const OxySource* __restrict__ sources,
                                    int n_src,
                                    double* __restrict__ po2_out);

// ---- Host wrapper --------------------------------------------------------
// solve_gpu: the host-callable "do the whole GPU computation" function.
//   Allocates device buffers, copies the source list H2D, launches the kernel,
//   copies the PO2 field D2H, and reports the measured KERNEL time (CUDA events)
//   via *kernel_ms. main.cu calls exactly this; all CUDA bookkeeping is hidden.
//
//   problem   : the loaded grid + sources (host side).
//   po2       : host output, resized to grid_size (output parameter).
//   kernel_ms : out-param, milliseconds spent in the kernel itself (not copies).
void solve_gpu(const OxyProblem& problem, std::vector<double>& po2, float* kernel_ms);
