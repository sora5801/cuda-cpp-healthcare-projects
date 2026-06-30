// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 2.17 : Allosteric Network Analysis
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls dcc_matrix_gpu(); kernels.cu
//   implements both the host wrapper and the device kernel. Included only by
//   .cu translation units (it declares a __global__ kernel, so the plain C++
//   compiler must never see it -- that is why the CPU reference lives in the
//   separate pure-C++ reference_cpu.h).
//
// THE BIG IDEA (this project = the Dynamical Cross-Correlation matrix, DCC)
//   The expensive step of allosteric network analysis is computing the N x N
//   residue-residue correlation matrix from an MD trajectory of T frames. Each
//   matrix entry C[i][j] is an INDEPENDENT average over T frames (dcc_core.h's
//   dcc_pair()). Independence is the GPU's favorite property: we assign ONE
//   GPU THREAD PER MATRIX ENTRY using a 2-D grid of 2-D blocks. Thread
//   (row = blockIdx.y*blockDim.y + threadIdx.y, col = blockIdx.x*blockDim.x +
//   threadIdx.x) computes exactly C[row][col]. No thread touches another's
//   output, so there are no races, no atomics, and no synchronization -- a clean
//   "embarrassingly parallel" map over a matrix. This is PATTERNS.md's
//   "independent jobs" idiom lifted from 1-D (1.12 Tanimoto) to a 2-D matrix.
//
//   The serial cost is O(N^2 * T); the GPU does the N^2 entries concurrently so
//   the wall-clock work per entry is just the O(T) inner sum. See THEORY.md
//   "GPU mapping".
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, dcc_core.h. Then kernels.cu.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // Trajectory (we read its fields to size the launch)

// ---- Device kernel -------------------------------------------------------
// dcc_kernel: one thread fills one entry C[row][col] of the DCC matrix.
//   coords : device pointer to [T*N*3] flat trajectory (frame-major; dcc_core.h)
//   mean   : device pointer to [N*3] precomputed per-residue means (double)
//   T, N   : number of frames and residues (guard the ragged edge blocks)
//   C      : device pointer to [N*N] output matrix (row-major, float)
//   __restrict__ promises the pointers do not alias so loads can stay in regs.
__global__ void dcc_kernel(const float* __restrict__ coords,
                           const double* __restrict__ mean,
                           int T, int N,
                           float* __restrict__ C);

// ---- Host wrapper --------------------------------------------------------
// dcc_matrix_gpu: the host-callable "compute the whole DCC matrix on the GPU".
//   Allocates device buffers, copies the trajectory + means H2D, launches
//   dcc_kernel over a 2-D grid covering the N x N matrix, copies the matrix D2H,
//   and reports the measured KERNEL time (CUDA events) via *kernel_ms. main.cu
//   compares this matrix against dcc_matrix_cpu() for an exact-match verification.
//
//   traj      : the loaded trajectory (provides coords, N, T)
//   mean      : host [N*3] per-residue means (same ones the CPU reference used)
//   C         : host output, resized to N*N (output parameter, row-major float)
//   kernel_ms : out-param, milliseconds spent in the kernel itself (not copies)
void dcc_matrix_gpu(const Trajectory& traj, const std::vector<double>& mean,
                    std::vector<float>& C, float* kernel_ms);
