// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 2.3 : Cryo-EM Single-Particle Reconstruction  (reduced-scope, 2D)
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls match_gpu() then
//   reconstruct_gpu(); kernels.cu implements both host wrappers and both device
//   kernels. Included only by .cu translation units (it declares __global__
//   kernels, so the plain C++ host compiler must never see it -- that is why the
//   data model and CPU reference live in the pure-C++ reference_cpu.h, which
//   THIS header includes to reuse Dataset / the geometry constants / the shared
//   __host__ __device__ physics).
//
// THE TWO GPU PATTERNS THIS PROJECT TEACHES
//   (1) PROJECTION MATCHING (the E-step) -- "score one item vs M templates,
//       independently, for each of N items". We give each PARTICLE its own
//       thread; that thread loops over all M reference projections (kept in
//       CONSTANT memory, broadcast to every thread) and keeps the best-scoring
//       angle. This is the same independent-jobs + constant-memory-query idiom
//       as project 1.12 (Tanimoto) -- and it is the step the catalog flags as
//       the cryo-EM walltime bottleneck (O(N*M)).
//   (2) BACK-PROJECTION (the M-step) -- "one thread per OUTPUT pixel, gather".
//       Each thread owns one density pixel and sums the contribution of every
//       particle's assigned profile (a gather, like CT back-projection in
//       project 4.01). No atomics: each pixel is written by exactly one thread,
//       and its internal sum runs in a fixed particle order -> deterministic and
//       bit-identical to the CPU reference.
//
//   The per-element math (project_sample / ncc_score / backproject_pixel) is NOT
//   duplicated here -- it is the shared __host__ __device__ core in
//   reference_cpu.h, so the kernels and the CPU reference compute identically.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, reference_cpu.h.
// Then read kernels.cu. The science/GPU-mapping is in ../THEORY.md.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // Dataset, IMG_SIZE/N_ANGLES/PROJ_LEN, HD physics

// ---- Device kernel 1: projection matching (E-step) -----------------------
// One thread per particle. The reference bank is read from the __constant__
// symbol defined in kernels.cu (not a parameter), so it is not in this list.
//   particles : [n*PROJ_LEN] device array of particle profiles, row-major
//   n         : number of particles (guards the ragged last block)
//   assign    : [n] output best reference-angle index per particle
//   best_score: [n] output best NCC score per particle
__global__ void match_kernel(const float* __restrict__ particles, int n,
                             int* __restrict__ assign,
                             float* __restrict__ best_score);

// ---- Device kernel 2: back-projection (M-step) ---------------------------
// One thread per OUTPUT pixel. ref_thetas is read from a __constant__ symbol.
//   particles : [n*PROJ_LEN] device particle profiles, row-major
//   assign    : [n] each particle's assigned angle index (from match_kernel)
//   n         : number of particles
//   recon     : [IMG_SIZE*IMG_SIZE] output density (row-major)
__global__ void backproject_kernel(const float* __restrict__ particles,
                                   const int* __restrict__ assign, int n,
                                   float* __restrict__ recon);

// ---- Host wrapper 1 ------------------------------------------------------
// match_gpu: upload the reference bank to constant memory + the particles to
//   global memory, launch match_kernel, time it (CUDA events), return the
//   per-particle assignment and score.
//   ds        : the loaded problem (uses ds.refs and ds.particles)
//   assign    : resized to n; filled with best angle index per particle
//   best_score: resized to n; filled with best NCC score per particle
//   kernel_ms : out-param, GPU-measured kernel milliseconds
void match_gpu(const Dataset& ds, std::vector<int>& assign,
               std::vector<float>& best_score, float* kernel_ms);

// ---- Host wrapper 2 ------------------------------------------------------
// reconstruct_gpu: upload assignments + per-angle thetas, launch
//   backproject_kernel, time it, return the reconstructed density.
//   ds        : the loaded problem (uses ds.particles)
//   assign    : the assignment produced by match_gpu (length n)
//   recon     : resized to IMG_SIZE*IMG_SIZE; the reconstructed density
//   kernel_ms : out-param, GPU-measured kernel milliseconds
void reconstruct_gpu(const Dataset& ds, const std::vector<int>& assign,
                     std::vector<float>& recon, float* kernel_ms);
