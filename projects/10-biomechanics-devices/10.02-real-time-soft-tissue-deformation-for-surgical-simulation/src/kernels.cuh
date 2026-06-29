// ===========================================================================
// src/kernels.cuh  --  GPU PBD interface (one thread per particle)
// ---------------------------------------------------------------------------
// Project 10.02 : Real-Time Soft-Tissue Deformation for Surgical Simulation
//
// THE BIG IDEA (ninth flagship pattern: PARALLEL CONSTRAINT PROJECTION)
//   PBD is a stencil-like solver, but the interesting twist is the JACOBI
//   constraint projection: each particle reads its neighbours' (read-only)
//   positions and computes its own correction, so all particles update
//   independently -> one thread per particle, double-buffered across iterations.
//   The host drives three kernels per step: predict, project (x iters), finalize.
//
//   The per-particle math is shared with the CPU (pbd.h), so the final mesh
//   matches the reference. kernels.cu defines the kernels.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, pbd.h, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // PbdParams, Vec3 (pure C++, safe in .cu)

// Predict positions under gravity: pa[i] = predict(x[i], v[i]).
__global__ void predict_kernel(PbdParams P, const Vec3* __restrict__ x,
                               const Vec3* __restrict__ v, const double* __restrict__ w,
                               Vec3* __restrict__ pa);

// One Jacobi projection iteration: dst[i] = src[i] + correction(i, src).
__global__ void constraint_kernel(PbdParams P, const Vec3* __restrict__ src,
                                  const double* __restrict__ w, Vec3* __restrict__ dst);

// Velocity update + commit: v[i] = (p[i]-x[i])/dt; x[i] = p[i].
__global__ void finalize_kernel(PbdParams P, const Vec3* __restrict__ p,
                                const double* __restrict__ w,
                                Vec3* __restrict__ x, Vec3* __restrict__ v);

// Host wrapper: run the full PBD time loop on the GPU; x and v are updated
// in place to the final mesh state. Returns total GPU time of the loop.
void simulate_gpu(const PbdParams& P, std::vector<Vec3>& x, std::vector<Vec3>& v,
                  const std::vector<double>& w, float* kernel_ms);
