// ===========================================================================
// src/kernels.cuh  --  GPU ensemble-of-cables interface (decls + teaching idea)
// ---------------------------------------------------------------------------
// Project 6.17 : Purkinje System & Conduction System Modeling
//
// THE BIG IDEA (ensemble-of-PDE-solvers pattern; docs/PATTERNS.md section 1 row
// "same ODE/PDE for many parameter sets" -> flagship 9.02)
//   The Purkinje tree is made of MANY 1-D cables that, during one heartbeat's
//   propagation model, we solve INDEPENDENTLY (their coupling is resolved
//   afterwards by the O(N) graph-delay pass). So we hand each cable to its OWN
//   GPU THREAD: thread `i` runs the full space x time PDE loop for cable i (in
//   per-thread LOCAL memory) and writes one CableResult. No inter-thread
//   communication -> embarrassingly parallel across cables. With N cables and a
//   block of B threads we launch ceil(N/B) blocks; thread
//   i = blockIdx.x*blockDim.x + threadIdx.x owns cable i.
//
//   The per-cable stepper (pk_simulate_cable in purkinje.h) is __host__ __device__
//   shared code, so this kernel and the CPU reference (reference_cpu.cpp) run
//   byte-for-byte identical arithmetic -> the GPU result matches the CPU to
//   round-off (in fact exactly for the integer activation steps). kernels.cu
//   defines the kernel + the host wrapper.
//
//   MEMORY NOTE: each thread needs three scratch arrays of PK_MAX_NODES doubles
//   (two ping-pong voltage buffers + one recovery buffer). Declared as local
//   arrays inside the kernel, these live in per-thread local memory (backed by
//   global memory / L1). This caps threads/block for occupancy but keeps the code
//   simple and correct; THEORY.md section "GPU mapping" discusses the alternative
//   one-thread-per-NODE tiling with shared-memory tridiagonal coefficients that
//   production solvers (MonoAlg3D_C) use.
//
// This header contains a __global__ declaration, so ONLY .cu files may include
// it. The CPU reference uses the pure-C++ reference_cpu.h instead.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, purkinje.h, reference_cpu.h.
// Then read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // PurkinjeTree, CableParams, CableResult (pure C++, safe in .cu)

// ---- Device kernel -------------------------------------------------------
// simulate_kernel: thread `i` simulates cable `params[i]` and writes out[i].
//   params : device array of N CableParams (read-only input)
//   n      : number of cables (guards the ragged last block)
//   out    : device array of N CableResult (one per cable)
__global__ void simulate_kernel(const CableParams* __restrict__ params, int n,
                                CableResult* __restrict__ out);

// ---- Host wrapper --------------------------------------------------------
// simulate_gpu: the host-callable "run the whole ensemble on the GPU" function.
//   Copies the CableParams array H2D, launches one thread per cable, copies the
//   CableResults D2H, and reports the measured KERNEL time (CUDA events) via
//   *kernel_ms. main.cu calls exactly this; all CUDA bookkeeping is hidden here.
//
//   t         : the Purkinje tree (its cables[] array is uploaded)
//   results   : host output, resized to N (output parameter)
//   kernel_ms : out-param, milliseconds spent in the kernel itself (not copies)
void simulate_gpu(const PurkinjeTree& t, std::vector<CableResult>& results,
                  float* kernel_ms);
