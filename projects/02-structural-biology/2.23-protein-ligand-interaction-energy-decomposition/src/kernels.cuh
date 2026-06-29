// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 2.23 -- Protein-Ligand Interaction Energy Decomposition   (template skeleton)
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls saxpy_gpu(); kernels.cu
//   implements both the host wrapper and the device kernel. Included only by
//   .cu translation units (it contains a __global__ declaration, so the plain
//   C++ compiler must never see it -- that is why the CPU reference lives in a
//   separate pure-C++ header).
//
// THE BIG IDEA (placeholder = SAXPY, out[i] = a*x[i] + y[i])
//   Every output element is independent, so we assign ONE GPU THREAD PER
//   ELEMENT. With n elements and a block of B threads, we launch
//   ceil(n / B) blocks; thread (blockIdx.x, threadIdx.x) owns element
//   i = blockIdx.x * blockDim.x + threadIdx.x. This "grid-of-1D-threads over a
//   1D array" is the most fundamental CUDA mapping and recurs everywhere.
//
//   TODO(impl): replace saxpy_kernel / saxpy_gpu with this project's real
//   kernel(s). Keep the launch-config reasoning in the comments (CLAUDE.md 6.1).
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh. Then read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>

// ---- Device kernel -------------------------------------------------------
// __global__ marks an entry point launched from host, run on device.
//   n   : number of elements (guards the ragged last block)
//   a   : scalar multiplier (passed by value -> lives in each thread's register)
//   x,y : device pointers to n input floats each (__restrict__ promises they do
//         not alias, letting the compiler keep loads in registers)
//   out : device pointer to n output floats
__global__ void saxpy_kernel(int n, float a,
                             const float* __restrict__ x,
                             const float* __restrict__ y,
                             float* __restrict__ out);

// ---- Host wrapper --------------------------------------------------------
// saxpy_gpu: the host-callable "do the whole GPU computation" function.
//   Allocates device buffers, copies inputs H2D, launches saxpy_kernel, copies
//   the result D2H, and reports the measured KERNEL time (CUDA events) via
//   *kernel_ms. main.cu calls exactly this; all CUDA bookkeeping is hidden here.
//
//   x, y : host inputs (length n)
//   out  : host output, resized to n (output parameter)
//   kernel_ms : out-param, milliseconds spent in the kernel itself (not copies)
void saxpy_gpu(int n, float a, const std::vector<float>& x,
               const std::vector<float>& y, std::vector<float>& out,
               float* kernel_ms);
