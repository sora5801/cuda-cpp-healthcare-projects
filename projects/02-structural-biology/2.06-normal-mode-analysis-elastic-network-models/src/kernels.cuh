// ===========================================================================
// src/kernels.cuh  --  GPU eigendecomposition interface (cuSOLVER)
// ---------------------------------------------------------------------------
// Project 2.06 : Normal Mode Analysis / Elastic Network Models
//
// THE BIG IDEA (a NEW pattern: a DENSE LINEAR-ALGEBRA LIBRARY)
//   The heart of NMA is diagonalizing the 3N x 3N Hessian -- a dense symmetric
//   eigenvalue problem, O(N^3). That is a solved problem with an excellent GPU
//   implementation, cuSOLVER, so we USE IT (and document exactly what it does,
//   per the "no black boxes" rule). Unlike the per-element kernels of the other
//   flagships, here the GPU work is one big library call.
//
//   kernels.cu defines the wrapper. main.cu calls cusolver_eigen() and verifies
//   the eigenvalues against the CPU Jacobi reference.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // Protein (pure C++, safe in .cu)

// Eigendecompose the symmetric matrix H (n x n, row-major) with cuSOLVER's
// divide-and-conquer symmetric eigensolver (Dsyevd).
//   eig  : filled with n eigenvalues, ascending
//   evec : filled with n*n eigenvectors, column-major (column k = eigenvector k)
//   kernel_ms : GPU time of the eigensolver
void cusolver_eigen(const std::vector<double>& H, int n, std::vector<double>& eig,
                    std::vector<double>& evec, float* kernel_ms);
