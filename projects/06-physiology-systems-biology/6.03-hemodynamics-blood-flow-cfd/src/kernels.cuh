// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 6.3 : Hemodynamics / Blood-Flow CFD   (reduced-scope teaching version)
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls nse_gpu(); kernels.cu
//   implements the host time-loop wrapper plus the four per-step kernels. This
//   header is included only by .cu translation units (it declares __global__
//   kernels, which the plain C++ compiler must never see -- that is why the CPU
//   reference API lives in the pure-C++ reference_cpu.h).
//
// THE BIG IDEA (fractional-step NSE as a STENCIL, PATTERNS.md §1)
//   Chorin's projection method is a sequence of nearest-neighbour cell updates:
//     predictor -> divergence -> Jacobi pressure sweeps -> corrector.
//   Each is a pure stencil: a cell reads only its immediate neighbours and
//   writes only itself. So we give EACH GRID CELL ITS OWN GPU THREAD on a 2-D
//   thread grid (16x16 tiles), and the host drives the time loop, launching the
//   kernels once per step and PING-PONGING double buffers (read old, write new,
//   swap) exactly like the CPU reference. Because both call the SAME per-cell
//   functions in nse_channel.h, the GPU reproduces the CPU field to machine
//   precision. This mirrors the lattice-Boltzmann flagship 6.04 (stencil +
//   ping-pong) and the Jacobi projection of 10.02.
//
//   Production CFD (SimVascular, OpenFOAM, HemeLB) replaces the Jacobi pressure
//   solve with algebraic multigrid (AmgX) and the structured grid with an
//   unstructured mesh + cuSPARSE SpMV -- see ../THEORY.md "Where this sits".
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, nse_channel.h,
//                  reference_cpu.h. Then read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // ChannelParams (pure C++, safe inside a .cu)

// ---- Device kernels (one thread per grid cell) ---------------------------
// Each kernel maps thread (x,y) = (blockIdx*blockDim + threadIdx) to cell (x,y)
// and calls the matching shared per-cell function from nse_channel.h.

// Predictor: provisional velocity u* (advection + diffusion + body force).
__global__ void predictor_kernel(int nx, int ny, double h, double dt, double gx,
                                 double nu0, double nu_inf, double lambda,
                                 double n_cy, double a_cy,
                                 const double* __restrict__ u,
                                 const double* __restrict__ v,
                                 double* __restrict__ us,
                                 double* __restrict__ vs);

// Divergence -> Poisson RHS = (rho/dt) div(u*).
__global__ void divergence_kernel(int nx, int ny, double h, double scale,
                                  const double* __restrict__ us,
                                  const double* __restrict__ vs,
                                  double* __restrict__ rhs);

// One Jacobi sweep of the pressure Poisson equation (reads p_old, writes p_new).
__global__ void pressure_kernel(int nx, int ny, double h,
                                const double* __restrict__ p_old,
                                const double* __restrict__ rhs,
                                double* __restrict__ p_new);

// Corrector/projection: u = u* - (dt/rho) grad(p)  -> divergence-free velocity.
__global__ void corrector_kernel(int nx, int ny, double h, double dt, double rho,
                                 const double* __restrict__ us,
                                 const double* __restrict__ vs,
                                 const double* __restrict__ p,
                                 double* __restrict__ u_new,
                                 double* __restrict__ v_new);

// ---- Host wrapper --------------------------------------------------------
// nse_gpu: run the full time loop on the GPU and return the final velocity
//   fields (u,v), each size nx*ny row-major, plus the total kernel time of the
//   whole loop via *kernel_ms (CUDA events; H2D/D2H copies excluded). main.cu
//   calls exactly this; all CUDA bookkeeping is hidden here.
void nse_gpu(const ChannelParams& p,
             std::vector<double>& u_final,
             std::vector<double>& v_final,
             float* kernel_ms);
