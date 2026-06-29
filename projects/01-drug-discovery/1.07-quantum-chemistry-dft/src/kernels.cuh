// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (the O(N^4) ERIs + the eigensolve)
// ---------------------------------------------------------------------------
// Project 1.7 : Quantum Chemistry / DFT  (reduced-scope RHF/SCF -- see THEORY.md)
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. The GPU does the two expensive jobs of a
//   quantum-chemistry calculation:
//
//     (1) build_eri_gpu()  -- the TWO-ELECTRON INTEGRAL tensor (ij|kl). There are
//         N^4 of them and every one is independent, so we give EACH integral its
//         own GPU thread (a flat grid over the N^4 quartets). This is the project's
//         headline kernel and the catalog's named O(N^4) bottleneck. It calls the
//         SAME `eri_primitive()` formula (gaussian_integrals.h) the CPU uses, so
//         the GPU and CPU tensors are bitwise identical (verification is exact).
//
//     (2) cusolver_generalized() -- each SCF cycle must solve F C = S C eps, a
//         small DENSE generalized symmetric eigenproblem. That is a solved library
//         problem (PATTERNS.md §5 "use the library, but no black box"), so we hand
//         it to cuSOLVER's Dsygvd. We document exactly what it computes below.
//
//   main.cu orchestrates: it builds the cheap one-electron matrices on the CPU,
//   builds the ERI tensor BOTH ways (CPU + GPU) and verifies they match, then runs
//   the SCF loop with the cuSOLVER eigensolver. kernels.cu implements (1) and (2).
//
//   This header contains a __global__ declaration, so only .cu files may include
//   it (the plain C++ compiler must never see __global__) -- that is why the CPU
//   reference lives in the separate pure-C++ reference_cpu.h.
//
// READ THIS AFTER: gaussian_integrals.h (the shared formulas), util/cuda_check.cuh,
//   util/timer.cuh. Then read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // Basis / ContractedGaussian (pure C++, safe in .cu)

// ---------------------------------------------------------------------------
// build_eri_gpu: compute the full [N^4] two-electron repulsion tensor on the GPU.
//   The contracted basis is flattened into plain arrays (centers, exponents,
//   coefficients, and a per-function primitive offset) so it can be uploaded to
//   device memory and indexed by the kernel without any STL containers.
//     bs        : the contracted basis (host side; flattened internally)
//     N         : number of basis functions
//     eri       : OUTPUT, resized to N^4, eri[((i*N+j)*N+k)*N+l] = (ij|kl)
//     kernel_ms : OUTPUT, GPU time of the ERI kernel itself (CUDA events)
//   Mirrors build_eri_cpu() exactly; main.cu diffs the two tensors.
// ---------------------------------------------------------------------------
void build_eri_gpu(const Basis& bs, int N, std::vector<double>& eri, float* kernel_ms);

// ---------------------------------------------------------------------------
// cusolver_generalized: solve the generalized symmetric eigenproblem
//   F C = S C eps  with cuSOLVER's divide-and-conquer Dsygvd (itype=1).
//   F, S : [N*N] symmetric matrices (row-major; symmetric so layout-agnostic)
//   C    : OUTPUT [N*N], MO coefficients (column k = orbital k), ascending eps
//   eps  : OUTPUT [N], orbital energies ascending
//   This is the per-iteration eigensolve of the SCF loop, the GPU counterpart of
//   the CPU solve_generalized(). main.cu passes this as the SCF's eigensolver.
// ---------------------------------------------------------------------------
void cusolver_generalized(const std::vector<double>& F, const std::vector<double>& S,
                          int N, std::vector<double>& C, std::vector<double>& eps);
