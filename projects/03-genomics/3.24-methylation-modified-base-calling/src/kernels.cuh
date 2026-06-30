// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 3.24 : Methylation / Modified-Base Calling
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls score_jobs_gpu(); kernels.cu
//   implements both the host wrapper and the device kernel. Included only by .cu
//   translation units (it carries a __global__ declaration, so the plain C++
//   compiler must never see it -- that is why the CPU reference + data model live
//   in the pure-C++ reference_cpu.h).
//
// THE BIG IDEA (PATTERNS.md §1 row "score one query vs N items, each independent")
//   This is f5c's GPU shape: a methylation call factorizes into thousands of
//   INDEPENDENT per-(read, site) alignment jobs. Each job runs a small banded
//   event-alignment DP twice (canonical vs methylated pore model) and emits one
//   log-likelihood ratio. Independence => assign ONE GPU THREAD PER JOB. With
//   num_jobs jobs and a block of B threads we launch ceil(num_jobs / B) blocks;
//   thread (blockIdx.x, threadIdx.x) owns job  j = blockIdx.x*blockDim.x + tx.
//
//   The TWO PORE MODELS are read by every thread but never change during the
//   launch -> they live in CONSTANT memory (the constant cache broadcasts a value
//   warp-wide), exactly like the constant-memory query in flagship 1.12. Each
//   thread keeps its tiny DP scratch (two length-WINDOW_KMERS rows) in registers/
//   local memory -- no shared memory needed at this size. See kernels.cu.
//
//   The per-job DP math (banded_align_core) is the SAME function the CPU calls,
//   from meth_core.h -> CPU and GPU agree to floating-point tolerance.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, meth_core.h. Then kernels.cu.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // MethData, Job, pore-model types (pure C++, safe in .cu)

// ---- Device kernel -------------------------------------------------------
// __global__ marks an entry point launched from host, run on device. One thread
// scores one job: derive its k-mer codes, run the banded DP under both pore
// models (read from constant memory), and write LLR = logL_meth - logL_canon.
//   jobs     : [num_jobs] device copy of the host Job array (POD, copies trivially)
//   num_jobs : guards the ragged last block
//   llr      : [num_jobs] device output, one log-likelihood ratio per job
// The pore models are NOT parameters: the kernel reads them from the __constant__
// tables defined in kernels.cu (uploaded once by the host wrapper).
__global__ void score_jobs_kernel(const Job* __restrict__ jobs, int num_jobs,
                                  float* __restrict__ llr);

// ---- Host wrapper --------------------------------------------------------
// score_jobs_gpu: the host-callable "do the whole GPU computation" function.
//   Uploads the two pore models to constant memory, copies the job array H2D,
//   launches score_jobs_kernel, copies the LLRs D2H, and reports the measured
//   KERNEL time (CUDA events) via *kernel_ms. main.cu calls exactly this; all
//   CUDA bookkeeping is hidden here.
//     d         : the loaded instance (provides jobs + both pore models)
//     llr       : host output, resized to num_jobs (output parameter)
//     kernel_ms : out-param, milliseconds spent in the kernel itself (not copies)
void score_jobs_gpu(const MethData& d, std::vector<float>& llr, float* kernel_ms);
