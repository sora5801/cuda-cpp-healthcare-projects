// ===========================================================================
// src/reference_cpu.h  --  Protein model, ANM Hessian, CPU eigen reference
// ---------------------------------------------------------------------------
// Project 2.06 : Normal Mode Analysis / Elastic Network Models
//
// WHAT THIS PROJECT COMPUTES
//   The low-frequency NORMAL MODES of a protein -- its collective motions
//   (domain hinges, breathing) that underlie allostery and function. The
//   Anisotropic Network Model (ANM) represents the protein as Cα atoms joined by
//   Hookean springs within a cutoff, builds the 3N x 3N HESSIAN of the elastic
//   energy, and DIAGONALIZES it: eigenvectors = modes, eigenvalues = squared
//   frequencies. The lowest non-zero modes are the functional motions.
//
// WHY A GPU
//   Diagonalizing the dense 3N x 3N Hessian is an O(N^3) eigenvalue problem --
//   the bottleneck. This flagship hands it to the cuSOLVER library (a symmetric
//   eigensolver) on the GPU (kernels.cu), and verifies it against a transparent
//   CPU Jacobi eigensolver here.
//
//   Pure C++ header (no CUDA). kernels.cu reuses Protein.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

// A protein as N Cα atoms; coords[3*i .. 3*i+2] are atom i's (x,y,z) in Angstrom.
struct Protein {
    int N = 0;
    double cutoff = 0.0;       // spring cutoff distance (Angstrom)
    std::vector<double> coords; // [3*N]
};

// Load from the text format (data/README.md): "N cutoff" then N lines of "x y z".
Protein load_protein(const std::string& path);

// Build the ANM Hessian H (3N x 3N, row-major) with spring constant gamma. For a
// pair (i,j) within cutoff, the off-diagonal 3x3 block is -(gamma/d^2)*ΔΔ^T and
// the diagonal block accumulates the negative of every off-diagonal it sees.
void build_hessian(const Protein& p, double gamma, std::vector<double>& H);

// CPU reference: eigenvalues of the symmetric matrix A (n x n) by the cyclic
// JACOBI algorithm (transparently correct), returned sorted ascending. Takes A
// by value because Jacobi destroys it. The trusted baseline for cuSOLVER.
void jacobi_eigenvalues(std::vector<double> A, int n, std::vector<double>& eig);

// Per-residue mobility (predicted thermal fluctuation, ~ a B-factor) from the
// eigen-decomposition: sum over non-trivial modes k of (1/eig_k)*|v_k at residue i|^2.
//   eig  : [n] eigenvalues (ascending);  evec : [n*n] column-major eigenvectors
//   thr  : modes with eig <= thr are the ~zero rigid-body modes (excluded)
void mobility(const std::vector<double>& eig, const std::vector<double>& evec,
              int N, double thr, std::vector<double>& mob);
