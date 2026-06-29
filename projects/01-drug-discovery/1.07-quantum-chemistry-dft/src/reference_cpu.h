// ===========================================================================
// src/reference_cpu.h  --  Molecule + basis model, and the CPU SCF reference
// ---------------------------------------------------------------------------
// Project 1.7 : Quantum Chemistry / DFT
//   (REDUCED-SCOPE TEACHING VERSION -- see THEORY.md "Where this sits in the real
//    world": a full Kohn-Sham DFT code is research-grade; we implement the
//    transparent kernel of the same method, restricted Hartree-Fock (RHF) SCF on
//    a minimal STO-3G Gaussian basis, which exercises the identical pattern --
//    integral assembly + a self-consistent generalized eigensolve.)
//
// WHAT THIS PROJECT COMPUTES
//   The ground-state electronic energy of a small closed-shell molecule (e.g. H2)
//   from first principles: no fitted parameters, just the positions and charges of
//   the nuclei and the laws of quantum mechanics. We do it the way every
//   production code (PySCF, NWChem, TeraChem) does -- expand the molecular orbitals
//   in a fixed Gaussian BASIS SET, build the one- and two-electron integral
//   matrices, and solve the Roothaan equations  F C = S C eps  SELF-CONSISTENTLY
//   (the field F depends on the density, which depends on F, so we iterate).
//
// WHY A GPU
//   The cost is dominated by the TWO-ELECTRON repulsion integrals (ERIs): there
//   are O(N^4) of them for N basis functions, and each is independent. That is a
//   textbook "embarrassingly parallel" workload -- so kernels.cu gives EACH ERI
//   its own GPU thread, while the small generalized eigenproblem each iteration is
//   handed to cuSOLVER. This header holds the CPU reference for the same math.
//
//   Pure C++ header (no CUDA constructs). kernels.cu reuses Molecule/Basis and the
//   shared per-integral formulas in gaussian_integrals.h.
//
// READ THIS BEFORE: reference_cpu.cpp (the implementations), then kernels.cuh.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Atom: one nucleus. Z is the nuclear charge / atomic number (H=1, He=2).
//   Coordinates are in BOHR (atomic units of length). The molecule file stores
//   them directly in Bohr so no unit conversion clutters the physics.
// ---------------------------------------------------------------------------
struct Atom {
    int    Z = 0;                 // nuclear charge (also # protons)
    double x = 0, y = 0, z = 0;   // position in Bohr
};

// ---------------------------------------------------------------------------
// Molecule: the parsed input. A list of atoms plus the total electron count.
//   For closed-shell (RESTRICTED) HF we assume an even number of electrons filling
//   n_electrons/2 doubly-occupied spatial orbitals. n_electrons is the sum of
//   nuclear charges minus the net molecular charge (so charge +1 models a cation
//   such as HeH+, which has one fewer electron than protons).
// ---------------------------------------------------------------------------
struct Molecule {
    std::vector<Atom> atoms;
    int n_electrons = 0;          // total electrons (must be even for closed-shell RHF)
};

// ---------------------------------------------------------------------------
// ContractedGaussian: ONE basis function -- a fixed sum of K s-type primitives
//   sharing a center (the atom it sits on):
//       phi(r) = sum_k coef[k] * norm(exp[k]) * exp(-exp[k] |r - center|^2)
//   We use a MINIMAL basis: exactly one 1s function per atom (STO-3G, K=3). That
//   gives N = (number of atoms) basis functions -- small enough to print the whole
//   Fock matrix, large enough to teach every step. The primitive normalization is
//   folded into `coef` at build time, so callers treat each weight as one number.
// ---------------------------------------------------------------------------
struct ContractedGaussian {
    double x = 0, y = 0, z = 0;   // center (copied from the parent atom), Bohr
    std::vector<double> exp;      // primitive exponents alpha_k (Bohr^-2)
    std::vector<double> coef;     // contraction coeffs * primitive norm
};

// A Basis is just the ordered list of contracted functions (row/col order of all
// the matrices S, T, V, F, and the density matrix P).
using Basis = std::vector<ContractedGaussian>;

// ---------------------------------------------------------------------------
// Parsing & basis construction
// ---------------------------------------------------------------------------

// Load a molecule from the text format documented in data/README.md:
//   line 1: "<natoms> <charge>"   (charge: net molecular charge, e.g. 0 or 1)
//   then natoms lines: "<Z> <x> <y> <z>"   (Bohr)
// Throws std::runtime_error on malformed input so the demo fails loudly.
Molecule load_molecule(const std::string& path);

// Build the minimal STO-3G basis for a molecule: one contracted 1s per atom. The
// STO-3G exponents/coefficients for H and He are hard-coded (they are public,
// standard numbers; see reference_cpu.cpp). Throws if an unsupported element
// appears -- this teaching build ships only H and He (enough for H2, He, HeH+).
Basis build_basis(const Molecule& mol);

// ---------------------------------------------------------------------------
// One-electron matrices (cheap; built on the CPU for both the reference and the
// GPU path -- the GPU only accelerates the O(N^4) ERIs).
//   S = overlap (the metric), Hcore = T + V (kinetic + nuclear attraction).
//   All are N x N, row-major.
// ---------------------------------------------------------------------------
void build_overlap(const Basis& bs, int N, std::vector<double>& S);
void build_core_hamiltonian(const Basis& bs, const Molecule& mol, int N,
                            std::vector<double>& Hcore);

// ---------------------------------------------------------------------------
// build_eri_cpu: the CPU reference for the O(N^4) two-electron integral tensor.
//   Fills eri[((i*N + j)*N + k)*N + l] = (ij|kl) for all i,j,k,l in [0,N).
//   This is the function the GPU kernel in kernels.cu replaces; main.cu compares
//   the two tensors element-by-element to verify the GPU.
// ---------------------------------------------------------------------------
void build_eri_cpu(const Basis& bs, int N, std::vector<double>& eri);

// ---------------------------------------------------------------------------
// nuclear_repulsion: the classical proton-proton Coulomb energy
//   sum_{A<B} Za*Zb / R_AB. It does not depend on the electrons, so it is added
//   once to the electronic energy to get the total. In Hartree.
// ---------------------------------------------------------------------------
double nuclear_repulsion(const Molecule& mol);

// ---------------------------------------------------------------------------
// ScfResult: everything the SCF loop produces, so main.cu can print and verify.
// ---------------------------------------------------------------------------
struct ScfResult {
    double e_total = 0.0;         // total energy = electronic + nuclear (Hartree)
    double e_electronic = 0.0;    // electronic part only
    double e_nuclear = 0.0;       // nuclear repulsion
    int    iterations = 0;        // SCF cycles taken to converge
    bool   converged = false;     // did |dE| fall below the threshold?
    std::vector<double> orbital_energies;  // [N] eps, ascending (MO energies)
    double homo = 0.0, lumo = 0.0;// highest occupied / lowest unoccupied MO energy
};

// ---------------------------------------------------------------------------
// run_scf: the full restricted-Hartree-Fock self-consistent field loop. It takes
//   the precomputed S, Hcore, and ERI tensor (so the SAME tensor -- CPU- or
//   GPU-built -- can drive it; that is how we prove GPU and CPU agree on the final
//   energy, not just on the integrals).
//     S, Hcore : [N*N] one-electron matrices
//     eri      : [N^4] two-electron tensor (from build_eri_cpu OR the GPU)
//     n_occ    : number of doubly-occupied orbitals = n_electrons / 2
//     max_iter, e_tol : convergence controls
//   The eigensolve is pluggable: `use_cusolver=false` uses the transparent Jacobi
//   path here; main.cu also calls it with the cuSOLVER solver for the GPU run.
// ---------------------------------------------------------------------------
ScfResult run_scf(const std::vector<double>& S, const std::vector<double>& Hcore,
                  const std::vector<double>& eri, int N, int n_occ,
                  double e_nuclear, int max_iter, double e_tol);

// ---------------------------------------------------------------------------
// Small dense-linear-algebra helpers used by the CPU SCF (transparent reference
// implementations; the GPU path uses cuSOLVER for the same generalized eigensolve).
//   symmetric_eigen: eigenpairs of a symmetric NxN matrix by cyclic Jacobi,
//     returned ascending. evec column-major (column k = eigenvector k).
//   solve_generalized: solve F C = S C eps via symmetric (Loewdin) orthogonal-
//     ization (X = S^-1/2), returning MO coefficients C and energies eps.
// ---------------------------------------------------------------------------
void symmetric_eigen(std::vector<double> A, int N,
                     std::vector<double>& eval, std::vector<double>& evec);
void solve_generalized(const std::vector<double>& F, const std::vector<double>& S,
                       int N, std::vector<double>& C, std::vector<double>& eps);

// ---------------------------------------------------------------------------
// build_fock: assemble the Fock matrix F = Hcore + G(P) from the density matrix P
//   and the two-electron tensor. G_{ij} = sum_{k,l} P_{kl} [ (ij|kl) - 0.5(ik|jl) ]
//   is the Coulomb-minus-exchange operator (the mean field the electrons feel).
//   Exposed here because both the CPU and GPU SCF loops reuse it (the GPU only
//   replaces the ERI *tensor*; the contraction into F is the same cheap O(N^4)
//   step). build_density forms P from the occupied MO coefficients.
// ---------------------------------------------------------------------------
void build_density(const std::vector<double>& C, int N, int n_occ,
                   std::vector<double>& P);
void build_fock(const std::vector<double>& Hcore, const std::vector<double>& P,
                const std::vector<double>& eri, int N, std::vector<double>& F);
