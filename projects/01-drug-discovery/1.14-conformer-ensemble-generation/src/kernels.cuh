// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 1.14 : Conformer Ensemble Generation
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls energies_gpu(); kernels.cu
//   implements both the host wrapper and the device kernel. Included only by
//   .cu translation units (it declares a __global__ kernel, which the plain C++
//   compiler must never see -- that is why the CPU reference lives behind the
//   pure-C++ reference_cpu.h).
//
// THE BIG IDEA -- conformer generation is embarrassingly parallel
//   We enumerate N_CONFORMER candidate conformers of a flexible chain. Each
//   conformer's energy depends ONLY on its own index (index -> torsions -> 3D
//   coordinates -> energy; all in conformer.h). There is zero coupling between
//   conformers, so we assign ONE GPU THREAD PER CONFORMER. With N conformers and
//   a block of B threads we launch ceil(N/B) blocks; thread
//   (blockIdx.x, threadIdx.x) owns conformer
//   c = blockIdx.x * blockDim.x + threadIdx.x. This is exactly the
//   "batch the embedding across many conformers simultaneously" the catalog
//   describes -- the GPU does on hundreds of conformers at once what the CPU does
//   one at a time.
//
//   The kernel calls the SAME conformer_energy() as the CPU reference, so the GPU
//   and CPU energies agree to ~machine precision (see THEORY "How we verify").
//
// READ THIS AFTER: conformer.h (the physics), util/cuda_check.cuh, util/timer.cuh.
//   Then read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>

// ---- Device kernel --------------------------------------------------------
// energies_kernel: one thread computes one conformer's energy.
//   n   : number of conformers (guards the ragged last block)
//   out : device pointer to n doubles; out[c] receives conformer c's energy
// The thread-to-data map and memory usage are documented at the definition in
// kernels.cu. No shared memory or atomics: outputs are fully independent.
__global__ void energies_kernel(long n, double* __restrict__ out);

// ---- Host wrapper ---------------------------------------------------------
// energies_gpu: the host-callable "do the whole GPU computation" function.
//   Allocates a device output buffer, launches energies_kernel over all
//   N_CONFORMER conformers, copies the energies back, and reports the measured
//   KERNEL time (CUDA events) via *kernel_ms. main.cu calls exactly this; all the
//   CUDA bookkeeping is hidden here.
//
//   energy    : host output, resized to N_CONFORMER (output parameter).
//   kernel_ms : out-param, milliseconds spent in the kernel itself (not copies).
void energies_gpu(std::vector<double>& energy, float* kernel_ms);
