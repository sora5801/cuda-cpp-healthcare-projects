// ===========================================================================
// src/kernels.cuh  --  GPU Poisson-Boltzmann solver interface
// ---------------------------------------------------------------------------
// Project 2.9 : Solvent-Accessible Surface & Poisson-Boltzmann Electrostatics
//
// THE PATTERN (a red-black STENCIL relaxation, cf. lattice-Boltzmann 6.04 and
//              reaction-diffusion 14.02)
//   The linearized Poisson-Boltzmann equation discretizes to a 7-point stencil
//   on a 3-D grid; we solve it with GAUSS-SEIDEL relaxation. Plain Gauss-Seidel
//   is inherently sequential (each cell uses already-updated neighbours), which
//   a GPU cannot parallelize. The fix is RED-BLACK COLOURING: colour cell
//   (x,y,z) by the parity of (x+y+z). Every red cell's six neighbours are black
//   and vice-versa, so:
//       * update ALL red cells in parallel (they read only black values),
//       * then update ALL black cells in parallel (they read the fresh reds).
//   That is two kernel launches per sweep, one thread per interior cell. No
//   races, and the arithmetic is identical to the serial red-black loop in
//   reference_cpu.cpp -- so the GPU field matches the CPU field (THEORY "verify").
//
//   Unlike reaction-diffusion (14.02) this needs NO ping-pong buffer: Gauss-
//   Seidel updates phi IN PLACE, and the colouring is exactly what makes the
//   in-place update safe in parallel.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, pbe.h, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // PbeProblem, GridParams (pure C++, safe in .cu)

// Host wrapper: solve the PBE on the GPU.
//   prob       : the built problem (eps/kappa2/rho grids + GridParams), input.
//   phi        : comes in zero-initialized (size n^3); updated in place to the
//                converged potential after prob.P.iters red-black sweeps.
//   kernel_ms  : out -- total GPU time across all sweeps (CUDA-event measured).
// After this returns, phi holds the GPU solution; main.cu compares it to the
// CPU reference. The two kernels (red, black) live in kernels.cu.
void solve_gpu(const PbeProblem& prob, std::vector<double>& phi, float* kernel_ms);
