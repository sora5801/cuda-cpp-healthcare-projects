// ===========================================================================
// src/reference_cpu.h  --  CPU reference interface: data model + the math
// ---------------------------------------------------------------------------
// Project 1.8 : Semi-Empirical & Tight-Binding Quantum Methods
//
// ROLE
//   Declares the plain-C++ "trusted baseline" that the GPU result is checked
//   against, plus the data structures shared across the project. Everything
//   here is pure host C++ (no CUDA) so it is safe to include from BOTH
//   reference_cpu.cpp (host compiler) and main.cu / kernels.cu (nvcc).
//
//   The pipeline for ONE molecule is:
//     1. build_hamiltonian()   -- adjacency -> N x N Huckel matrix H  (uses the
//                                 shared tb_hamiltonian_entry() so CPU==GPU)
//     2. jacobi_eigen()        -- diagonalise H  -> MO energies + coefficients
//     3. analyze_molecule()    -- fill electrons, total pi energy, HOMO-LUMO gap
//
//   The GPU path replaces step 2 with cuSOLVER's batched eigensolver (kernels.cu)
//   but reuses steps 1 and 3 verbatim, so the two paths share all the chemistry.
//
// READ THIS AFTER: tight_binding.h.   READ BEFORE: reference_cpu.cpp, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// MoleculeBatch
//   A "ragged" batch of molecules packed into flat arrays. cuSOLVER's batched
//   eigensolver requires every matrix in the batch to be the SAME dimension, so
//   we pad every molecule's adjacency/Hamiltonian to `max_n` (the largest atom
//   count in the batch). Padding atoms are isolated (no bonds, on-site energy
//   alpha): their orbitals decouple and contribute a known, ignorable block.
//
//   Layout (all row-major, molecule m's block starts at m*max_n*max_n):
//     adj   : [num_mol * max_n * max_n] bytes, padded adjacency matrices
//     n     : [num_mol] the TRUE atom count of each molecule (<= max_n)
//     name  : [num_mol] a human label for the report
// ---------------------------------------------------------------------------
struct MoleculeBatch {
    int num_mol = 0;                       // number of molecules in the batch
    int max_n   = 0;                       // padded matrix dimension (max atoms)
    std::vector<unsigned char> adj;        // [num_mol * max_n * max_n] adjacency
    std::vector<int>           n;          // [num_mol] true atom count per molecule
    std::vector<std::string>   name;       // [num_mol] molecule names
};

// ---------------------------------------------------------------------------
// MoleculeResult
//   The per-molecule physical observables we report and verify. Energies are in
//   units of |beta| (Huckel convention; see tight_binding.h).
// ---------------------------------------------------------------------------
struct MoleculeResult {
    double total_pi_energy = 0.0;   // sum over occupied MOs of (occupancy * energy)
    double homo_energy     = 0.0;   // energy of the Highest Occupied MO
    double lumo_energy     = 0.0;   // energy of the Lowest Unoccupied MO
    double homo_lumo_gap   = 0.0;   // lumo - homo  (a reactivity / stability proxy)
    int    n_atoms         = 0;     // true atom count (basis size) of this molecule
};

// ---- I/O -------------------------------------------------------------------
// load_batch: read the tiny committed sample (see data/README.md for the format)
//   into a padded MoleculeBatch. Throws std::runtime_error on a bad/empty file.
MoleculeBatch load_batch(const std::string& path);

// ---- Step 1: matrix construction ------------------------------------------
// build_hamiltonian: fill `H` (max_n x max_n, row-major, column-major-identical
//   because it is symmetric) for molecule `mol` of the batch, using the shared
//   tb_hamiltonian_entry(). Padding rows/cols beyond n[mol] get alpha on the
//   diagonal and 0 off-diagonal (isolated atoms).
void build_hamiltonian(const MoleculeBatch& batch, int mol, std::vector<double>& H);

// ---- Step 2: the CPU eigensolver (the reference) --------------------------
// jacobi_eigen: classic cyclic Jacobi diagonalisation of a real symmetric n x n
//   matrix `A` (row-major). On return:
//     eval : [n] eigenvalues in ASCENDING order (MO energies, low to high)
//     evec : [n*n] eigenvectors as COLUMNS, column k = MO k's coefficients,
//            column-major (evec[k*n + i] = coefficient of atom i in MO k) to
//            match cuSOLVER's output layout exactly.
//   This is the trusted baseline; it is O(n^3) per sweep and fully deterministic.
void jacobi_eigen(const std::vector<double>& A, int n,
                  std::vector<double>& eval, std::vector<double>& evec);

// ---- Step 3: chemistry post-processing (shared by CPU and GPU paths) ------
// analyze_molecule: given the ascending MO energies of one molecule (only the
//   first n_atoms are physical; the rest are padding), occupy them with the
//   Aufbau principle (2 electrons per MO from the bottom) and compute the total
//   pi energy, HOMO, LUMO, and the HOMO-LUMO gap.
MoleculeResult analyze_molecule(const std::vector<double>& eval, int n_atoms);
