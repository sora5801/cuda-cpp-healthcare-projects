// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 3.25 : Base Quality Score Recalibration (BQSR)
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls bqsr_gpu(); kernels.cu
//   implements the host wrapper and the two device kernels. Included only by .cu
//   translation units (it declares __global__ kernels, so the plain C++ compiler
//   must never see it -- that is why the CPU reference API lives in the separate
//   pure-C++ header reference_cpu.h).
//
// THE BIG IDEA: parallel assign + atomic INTEGER reduction (PATTERNS.md row
//   "clustering / centroid accumulation", exemplified by flagship 11.09).
//     * accumulate_kernel : ONE THREAD PER BASE. Each thread classifies its base
//       (bqsr.h classify_base) and, if it survives masking, atomicAdd's 1 into the
//       observation counter of its covariate bin and `is_err` into the error
//       counter. Many bases share a bin -> the adds collide -> atomicAdd. Because
//       the counters are UNSIGNED INTEGERS, the adds commute: the table is
//       deterministic AND bit-identical to the CPU's table.
//     * recalibrate_kernel : ONE THREAD PER BASE. Reads the finished table and
//       writes each base's recalibrated quality = empirical_q(bin) (or keeps the
//       original quality for skipped / no-evidence bases).
//
// READ THIS AFTER: bqsr.h, util/cuda_check.cuh, util/timer.cuh. Then kernels.cu.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // Dataset, NUM_BINS, the shared covariate model

// ---------------------------------------------------------------------------
// accumulate_kernel: tally the covariate table on the GPU.
//   grid   : ceil(total_bases / block) blocks
//   block  : 256 threads (a good occupancy default on sm_75..sm_89)
//   thread : global base index g = blockIdx.x*blockDim.x + threadIdx.x
//   Touches: global memory (read-only read arrays + reference + mask), and
//            atomicAdd into the two global counter arrays d_obs / d_err.
//   Pointers are device pointers; layout matches the Dataset (see bqsr.h).
// ---------------------------------------------------------------------------
__global__ void accumulate_kernel(int total_bases, int read_len,
                                  const char* __restrict__ ref, int ref_len,
                                  const char* __restrict__ bases,
                                  const int* __restrict__ quals,
                                  const int* __restrict__ read_pos,
                                  const unsigned char* __restrict__ known,
                                  unsigned int* __restrict__ d_obs,
                                  unsigned int* __restrict__ d_err);

// ---------------------------------------------------------------------------
// recalibrate_kernel: write each base's recalibrated quality.
//   Same launch shape (one thread per base). Reads the finished d_obs / d_err
//   table and writes d_newq[g]. No atomics here: each thread owns one output.
// ---------------------------------------------------------------------------
__global__ void recalibrate_kernel(int total_bases, int read_len,
                                   const char* __restrict__ ref, int ref_len,
                                   const char* __restrict__ bases,
                                   const int* __restrict__ quals,
                                   const int* __restrict__ read_pos,
                                   const unsigned char* __restrict__ known,
                                   const unsigned int* __restrict__ d_obs,
                                   const unsigned int* __restrict__ d_err,
                                   int* __restrict__ d_newq);

// ---------------------------------------------------------------------------
// bqsr_gpu: the host-callable "do the whole GPU computation" function.
//   Uploads the dataset, launches accumulate_kernel then recalibrate_kernel,
//   downloads the covariate table (obs/err) and the recalibrated qualities, and
//   reports the measured kernel time (CUDA events) via *kernel_ms. main.cu calls
//   exactly this and compares the table + qualities against the CPU reference.
//
//   Outputs (resized inside): obs/err = the [NUM_BINS] integer table;
//   new_quals = [total_bases] recalibrated qualities. kernel_ms = ms in the two
//   kernels (not the copies) -- a teaching artifact, never a benchmark claim.
// ---------------------------------------------------------------------------
void bqsr_gpu(const Dataset& d,
              std::vector<unsigned int>& obs,
              std::vector<unsigned int>& err,
              std::vector<int>& new_quals,
              float* kernel_ms);
