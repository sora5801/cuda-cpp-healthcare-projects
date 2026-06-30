// ===========================================================================
// src/kernels.cuh  --  GPU 3D-Variability (3DVA) interface
// ---------------------------------------------------------------------------
// Project 2.20 : Heterogeneous Cryo-EM Reconstruction (3D Variability)
//
// THE BIG IDEA (two GPU patterns in one project)
//   3DVA = PCA on a set of N particle volumes. We do it on the GPU in two
//   complementary styles, both taught in PATTERNS.md:
//
//   1. PER-ELEMENT KERNELS for the embarrassingly-parallel matrix building:
//        * mean volume       : one thread per voxel (sum over particles)
//        * N x N Gram matrix  : one thread per (i,j) entry -- each is an
//                               independent dot product (PATTERNS.md §1, "score
//                               one item vs many"). This is the work-heavy step:
//                               ~N^2/2 dot products of length D = G^3.
//        * volume PC + per-particle projections : one thread per voxel / particle.
//
//   2. A DENSE LINEAR-ALGEBRA LIBRARY for the eigenproblem:
//        * diagonalizing the small N x N Gram matrix is a solved problem with an
//          excellent GPU implementation, cuSOLVER's Dsyevd. We USE it (and
//          document exactly what it computes -- no black boxes, CLAUDE.md §6.1.6),
//          the same pattern as flagship 2.06 (PATTERNS.md §5).
//
//   main.cu calls run_3dva_gpu(); the heavy lifting is on the GPU, and the CPU
//   reference (reference_cpu.cpp) recomputes everything for verification.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, reference_cpu.h.
//   The science / GPU mapping is in ../THEORY.md.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // VolumeSet + shared HD math (pure C++, safe in .cu)

// ---------------------------------------------------------------------------
// GpuTimings: a small bag of per-stage GPU times (milliseconds) so main.cu can
//   print a clear, labeled breakdown to stderr. Times are a teaching artifact,
//   never a benchmark claim (CLAUDE.md §12).
// ---------------------------------------------------------------------------
struct GpuTimings {
    float mean_ms  = 0.0f;   // per-voxel mean kernel
    float gram_ms  = 0.0f;   // N x N Gram-matrix kernel (the heavy step)
    float eigen_ms = 0.0f;   // cuSOLVER Dsyevd
    float lift_ms  = 0.0f;   // lift Gram eigenvector -> volume-space PC
    float proj_ms  = 0.0f;   // per-particle projection kernel
};

// ---------------------------------------------------------------------------
// run_3dva_gpu: the full GPU 3DVA pipeline on `vs`.
//   Computes, entirely on the GPU (except the orchestration):
//     mean[D]            : per-voxel mean volume
//     eval[N]            : Gram eigenvalues, ascending (cuSOLVER)
//     pc1[D]             : volume-space principal component #1 (largest variance),
//                          unit-normalized, sign-fixed to match the CPU reference
//     z[N]               : each particle's latent coordinate along pc1
//     var_explained_pc1  : fraction of total variance captured by PC1
//                          ( = top eigenvalue / sum of eigenvalues )
//   `t` receives the per-stage GPU timings.
//
//   All outputs are double precision and use the SAME shared HD math as the CPU
//   reference, so main.cu can verify GPU == CPU to ~machine precision.
// ---------------------------------------------------------------------------
void run_3dva_gpu(const VolumeSet& vs,
                  std::vector<double>& mean,
                  std::vector<double>& eval,
                  std::vector<double>& pc1,
                  std::vector<double>& z,
                  double& var_explained_pc1,
                  GpuTimings& t);
