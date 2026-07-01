// ===========================================================================
// src/kernels.cuh  --  GPU monodomain interface (declarations + the big idea)
// ---------------------------------------------------------------------------
// Project 6.1 : Cardiac Electrophysiology Simulation
//
// THE BIG IDEA (the STENCIL + per-cell ODE pattern, docs/PATTERNS.md section 1)
//   The heart is a grid of ~10^8 excitable cells. Two operations advance it:
//     REACTION  -- each cell integrates its own ODE; cells are INDEPENDENT, so
//                  we give each cell its own GPU thread (embarrassingly parallel,
//                  memory-bandwidth bound -- exactly what a GPU is built for).
//     DIFFUSION -- each cell couples to its 4 grid neighbours: a nearest-neighbour
//                  STENCIL. We again give each cell a thread, reading neighbours
//                  from a read-only buffer and writing to a second buffer, then
//                  PING-PONG the two buffers -- no races, no atomics.
//
//   The host runs the operator-split time loop, launching TWO kernels per step
//   (react then diffuse). The per-cell math is the shared react_step() and
//   diffuse_cell() in cardiac_cell.h, so the GPU reproduces the CPU result
//   byte-for-byte (the key to exact verification).
//
//   Production cardiac solvers (openCARP, MonoAlg3D) keep this reaction kernel
//   verbatim but replace the explicit-diffusion kernel with an IMPLICIT solve
//   (Crank-Nicolson + conjugate gradient via cuSPARSE/cuSOLVER) so they can take
//   larger, unconditionally-stable timesteps -- see ../THEORY.md "real world".
//
// This header is included only by .cu files (it declares __global__ kernels).
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, cardiac_cell.h. Then
// read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // MonodomainParams (pure C++, safe inside a .cu)

// ---- Device kernels -------------------------------------------------------

// react_kernel: thread (x,y) advances its own cell's (V,w) by the FHN reaction
//   ODE for one dt. Fully independent per cell; updates V and w in place.
__global__ void react_kernel(int nx, int ny, MonodomainParams p,
                             double* __restrict__ V, double* __restrict__ w);

// diffuse_kernel: thread (x,y) applies the 5-point Laplacian diffusion update,
//   reading the (post-reaction) field V_in and writing V_out (ping-pong).
__global__ void diffuse_kernel(int nx, int ny, MonodomainParams p,
                               const double* __restrict__ V_in,
                               double* __restrict__ V_out);

// ---- Host wrapper ---------------------------------------------------------

// monodomain_gpu: run the full operator-split time loop on the GPU and return
//   the final voltage field V (size nx*ny) and recovery field w. Reports the
//   total kernel time of the loop (CUDA events) via *kernel_ms. main.cu calls
//   exactly this; all CUDA bookkeeping is hidden inside.
void monodomain_gpu(const MonodomainParams& p,
                    std::vector<double>& V_final, std::vector<double>& w_final,
                    float* kernel_ms);
