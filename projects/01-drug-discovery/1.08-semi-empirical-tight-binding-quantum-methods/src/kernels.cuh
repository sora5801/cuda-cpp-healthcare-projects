// ===========================================================================
// src/kernels.cuh  --  GPU compute interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 1.8 : Semi-Empirical & Tight-Binding Quantum Methods
//
// ROLE IN THE PROJECT
//   The "what the GPU offers" header. main.cu calls tb_solve_batch_gpu(), which
//   runs the WHOLE batch on the device: build every molecule's Huckel matrix in
//   parallel (a custom kernel), then diagonalise the entire batch in ONE
//   cuSOLVER call. kernels.cu implements it. Included only by .cu translation
//   units (it declares a __global__ kernel), so the plain C++ compiler never
//   sees it -- that is why the CPU reference lives in its own pure-C++ header.
//
// THE BIG IDEA (TWO classic patterns combined; see docs/PATTERNS.md)
//   1) BATCHED INDEPENDENT JOBS + custom kernel (like 1.12 Tanimoto):
//      Each molecule's matrix is independent, so we build all of them at once.
//      We launch a 2-D grid where thread (mol, i, j) writes one matrix element
//      H_mol[i][j] using the SAME tb_hamiltonian_entry() the CPU uses -> the
//      device matrices are bit-identical to the host's.
//   2) A DENSE LINEAR-ALGEBRA LIBRARY, BATCHED (like 2.06 NMA, but batched):
//      Diagonalising a small symmetric matrix is a solved problem. cuSOLVER's
//      cusolverDnDsyevjBatched diagonalises an ENTIRE BATCH of equal-size
//      symmetric matrices in a single launch -- exactly the "thousands of small
//      molecules optimised simultaneously" pattern the catalog calls for. We USE
//      it and document precisely what it computes (no black boxes, CLAUDE.md §6).
//
//   This file declares the device kernel (for transparency) plus the one host
//   entry point main.cu needs.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, tight_binding.h,
//                  reference_cpu.h.  Then read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>

// ---- Device kernel (declared here for transparency; defined in kernels.cu) -
// build_hamiltonians_kernel: fill the padded Hamiltonian cube for the whole
// batch, one matrix element per thread.
//   d_adj  : [num_mol * n * n] device adjacency bytes (padded), row-major
//   d_nreal: [num_mol] device array of each molecule's TRUE atom count, so the
//            kernel knows which diagonal entries are padding (>= n_real)
//   d_H    : [num_mol * n * n] device output matrices (double), row/col-major
//            identical because each matrix is symmetric
//   num_mol : number of molecules; n : padded dimension (max atoms)
// Thread mapping: a 2-D block tiles the n x n matrix; blockIdx.z selects the
// molecule. Thread (mol, i, j) -> d_H[mol*n*n + i*n + j].
__global__ void build_hamiltonians_kernel(const unsigned char* __restrict__ d_adj,
                                          const int* __restrict__ d_nreal,
                                          double* __restrict__ d_H,
                                          int num_mol, int n);

// ---- Host wrapper --------------------------------------------------------
// tb_solve_batch_gpu: run the entire batch on the GPU.
//   Inputs:
//     adj      : host padded adjacency cube [num_mol * n * n] bytes
//     n_real   : host [num_mol] true atom count per molecule (drives padding)
//     num_mol  : number of molecules
//     n        : padded matrix dimension (max atoms across the batch)
//   Outputs:
//     eval     : resized to [num_mol * n], ascending eigenvalues per molecule
//                (block m occupies eval[m*n .. m*n + n - 1]); the first
//                n_real[m] of each block are the physical MO energies, the rest
//                are the large padding eigenvalues (~TB_PAD_DIAG).
//     build_ms : out-param, ms spent in the Hamiltonian-build kernel
//     solve_ms : out-param, ms spent in the cuSOLVER batched eigensolver
//   The function allocates device memory, builds the matrices on-device, calls
//   the batched eigensolver, copies eigenvalues back, and frees everything.
void tb_solve_batch_gpu(const std::vector<unsigned char>& adj,
                        const std::vector<int>& n_real, int num_mol, int n,
                        std::vector<double>& eval,
                        float* build_ms, float* solve_ms);
