// ===========================================================================
// src/reference_cpu.cpp  --  CPU reference: basis, integrals, and the SCF loop
// ---------------------------------------------------------------------------
// Project 1.7 : Quantum Chemistry / DFT  (reduced-scope RHF/SCF -- see THEORY.md)
// Compiled by the host C++ compiler only. See reference_cpu.h for the contract and
// gaussian_integrals.h for the per-integral formulas shared with the GPU kernel.
//
// This file is the transparent baseline: every step a production code performs
// (PySCF, NWChem, TeraChem) is here in a few hundred readable lines, so the GPU
// kernel can be verified against something you can fully understand.
// ===========================================================================
#include "reference_cpu.h"
#include "gaussian_integrals.h"   // overlap/kinetic/nuclear/eri primitives (HD)

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
// next_data_line: read the next NON-COMMENT, non-blank line from a stream into an
//   istringstream the caller can extract tokens from. Lines whose first
//   non-whitespace character is '#' are skipped, so the molecule file can carry
//   explanatory comments (see data/sample/h2.txt). Returns false at end of file.
bool next_data_line(std::istream& in, std::istringstream& iss) {
    std::string line;
    while (std::getline(in, line)) {
        // Find the first non-whitespace character to classify the line.
        size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) continue;     // blank line
        if (line[first] == '#') continue;             // comment line
        iss.clear();
        iss.str(line);
        return true;
    }
    return false;
}
}  // namespace

// ---------------------------------------------------------------------------
// load_molecule: parse the tiny text format (see data/README.md).
//   line 1: "<natoms> <charge>";  then natoms lines "<Z> <x> <y> <z>" (Bohr).
//   '#' comment lines and blank lines are ignored anywhere in the file.
// ---------------------------------------------------------------------------
Molecule load_molecule(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open molecule file: " + path);

    std::istringstream iss;
    int natoms = 0, charge = 0;
    if (!next_data_line(in, iss) || !(iss >> natoms >> charge) || natoms <= 0)
        throw std::runtime_error("bad header (expected '<natoms> <charge>') in " + path);

    Molecule mol;
    mol.atoms.resize(natoms);
    int total_Z = 0;                       // sum of nuclear charges = #electrons if neutral
    for (int i = 0; i < natoms; ++i) {
        Atom& a = mol.atoms[i];
        if (!next_data_line(in, iss) || !(iss >> a.Z >> a.x >> a.y >> a.z))
            throw std::runtime_error("atom line truncated in " + path);
        if (a.Z < 1) throw std::runtime_error("nonpositive nuclear charge in " + path);
        total_Z += a.Z;
    }
    // Electrons = protons minus the net molecular charge (charge>0 => fewer e-).
    mol.n_electrons = total_Z - charge;
    if (mol.n_electrons <= 0)
        throw std::runtime_error("non-positive electron count (check charge) in " + path);
    if (mol.n_electrons % 2 != 0)
        throw std::runtime_error("odd electron count: this teaching build does CLOSED-SHELL "
                                 "(restricted) HF only; supply an even-electron species");
    return mol;
}

// ---------------------------------------------------------------------------
// STO-3G parameters for the 1s function of H and He (public, standard numbers).
//   STO-3G fits one Slater-type 1s orbital with THREE Gaussians. The exponents
//   below are the tabulated values for hydrogen and helium; the d[] are the shared
//   contraction coefficients of the 1s contraction. Source: the original STO-3G
//   work (Hehre, Stewart, Pople 1969) -- the same numbers PySCF/Gaussian ship.
// ---------------------------------------------------------------------------
namespace {
// Shared 1s contraction coefficients (same for every element in STO-3G's 1s).
const double STO3G_1S_COEF[3] = {0.444635, 0.535328, 0.154329};
// Element-specific 1s exponents (Bohr^-2).
const double H_1S_EXP[3]  = {0.168856, 0.623913, 3.42525};
const double HE_1S_EXP[3] = {0.480844, 1.776691, 9.753934};
}  // namespace

// ---------------------------------------------------------------------------
// build_basis: one contracted 1s per atom. We FOLD the primitive normalization
//   into the stored coefficient (coef = d_k * N(alpha_k)) so the integral routines
//   can treat each primitive as a single weighted Gaussian. This keeps the hot
//   loops (and the GPU kernel) free of pow() calls.
// ---------------------------------------------------------------------------
Basis build_basis(const Molecule& mol) {
    Basis bs;
    bs.reserve(mol.atoms.size());
    for (const Atom& at : mol.atoms) {
        const double* expo = nullptr;
        if      (at.Z == 1) expo = H_1S_EXP;
        else if (at.Z == 2) expo = HE_1S_EXP;
        else throw std::runtime_error("unsupported element Z=" + std::to_string(at.Z) +
                                      "; this teaching build ships only H and He");
        ContractedGaussian g;
        g.x = at.x; g.y = at.y; g.z = at.z;
        for (int k = 0; k < 3; ++k) {
            g.exp.push_back(expo[k]);
            // coef = contraction coefficient * primitive normalization constant.
            g.coef.push_back(STO3G_1S_COEF[k] * gauss_norm(expo[k]));
        }
        bs.push_back(std::move(g));
    }
    return bs;
}

// ---------------------------------------------------------------------------
// Contracted integrals = double/quadruple sums over primitives of the shared
//   primitive formulas, weighted by the contraction coefficients. These small
//   helpers keep build_overlap/build_core_hamiltonian/build_eri_cpu readable.
// ---------------------------------------------------------------------------
namespace {

// Overlap between contracted functions A and B: sum over primitive pairs.
double contracted_overlap(const ContractedGaussian& A, const ContractedGaussian& B) {
    double s = 0.0;
    for (size_t p = 0; p < A.exp.size(); ++p)
        for (size_t q = 0; q < B.exp.size(); ++q)
            s += A.coef[p] * B.coef[q] *
                 overlap_primitive(A.exp[p], A.x, A.y, A.z, B.exp[q], B.x, B.y, B.z);
    return s;
}

// Kinetic energy between contracted functions A and B.
double contracted_kinetic(const ContractedGaussian& A, const ContractedGaussian& B) {
    double t = 0.0;
    for (size_t p = 0; p < A.exp.size(); ++p)
        for (size_t q = 0; q < B.exp.size(); ++q)
            t += A.coef[p] * B.coef[q] *
                 kinetic_primitive(A.exp[p], A.x, A.y, A.z, B.exp[q], B.x, B.y, B.z);
    return t;
}

// Nuclear attraction between A and B, summed over every nucleus C in the molecule.
double contracted_nuclear(const ContractedGaussian& A, const ContractedGaussian& B,
                          const Molecule& mol) {
    double v = 0.0;
    for (const Atom& C : mol.atoms)
        for (size_t p = 0; p < A.exp.size(); ++p)
            for (size_t q = 0; q < B.exp.size(); ++q)
                v += A.coef[p] * B.coef[q] *
                     nuclear_primitive(A.exp[p], A.x, A.y, A.z, B.exp[q], B.x, B.y, B.z,
                                       static_cast<double>(C.Z), C.x, C.y, C.z);
    return v;
}

}  // namespace

void build_overlap(const Basis& bs, int N, std::vector<double>& S) {
    S.assign(static_cast<size_t>(N) * N, 0.0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            S[static_cast<size_t>(i) * N + j] = contracted_overlap(bs[i], bs[j]);
}

void build_core_hamiltonian(const Basis& bs, const Molecule& mol, int N,
                            std::vector<double>& Hcore) {
    Hcore.assign(static_cast<size_t>(N) * N, 0.0);
    // Hcore = T + V: the energy of ONE electron in the field of the bare nuclei,
    // ignoring the other electrons (those enter later through G(P) in the Fock op).
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            Hcore[static_cast<size_t>(i) * N + j] =
                contracted_kinetic(bs[i], bs[j]) + contracted_nuclear(bs[i], bs[j], mol);
}

// ---------------------------------------------------------------------------
// build_eri_cpu: the reference O(N^4) two-electron tensor. Quadruple loop over
//   basis functions; inside, a quadruple loop over their primitives (the analytic
//   contraction). This is the function the GPU kernel parallelizes -- there, the
//   outer four loops become one thread per (i,j,k,l) quartet.
// ---------------------------------------------------------------------------
void build_eri_cpu(const Basis& bs, int N, std::vector<double>& eri) {
    eri.assign(static_cast<size_t>(N) * N * N * N, 0.0);
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j)
        for (int k = 0; k < N; ++k)
          for (int l = 0; l < N; ++l) {
              const ContractedGaussian& A = bs[i];
              const ContractedGaussian& B = bs[j];
              const ContractedGaussian& C = bs[k];
              const ContractedGaussian& D = bs[l];
              double val = 0.0;
              // Contract over all primitive quartets (3^4 = 81 terms for STO-3G).
              for (size_t pa = 0; pa < A.exp.size(); ++pa)
                for (size_t pb = 0; pb < B.exp.size(); ++pb)
                  for (size_t pc = 0; pc < C.exp.size(); ++pc)
                    for (size_t pd = 0; pd < D.exp.size(); ++pd)
                        val += A.coef[pa] * B.coef[pb] * C.coef[pc] * D.coef[pd] *
                               eri_primitive(A.exp[pa], A.x, A.y, A.z,
                                             B.exp[pb], B.x, B.y, B.z,
                                             C.exp[pc], C.x, C.y, C.z,
                                             D.exp[pd], D.x, D.y, D.z);
              eri[(((static_cast<size_t>(i) * N + j) * N + k) * N + l)] = val;
          }
}

// ---------------------------------------------------------------------------
// nuclear_repulsion: classical Coulomb energy between the bare nuclei.
// ---------------------------------------------------------------------------
double nuclear_repulsion(const Molecule& mol) {
    double e = 0.0;
    const auto& a = mol.atoms;
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = i + 1; j < a.size(); ++j) {
            const double dx = a[i].x - a[j].x, dy = a[i].y - a[j].y, dz = a[i].z - a[j].z;
            const double R = std::sqrt(dx * dx + dy * dy + dz * dz);
            e += static_cast<double>(a[i].Z) * a[j].Z / R;
        }
    return e;
}

// ---------------------------------------------------------------------------
// symmetric_eigen: cyclic Jacobi eigensolver for a symmetric matrix (transparent
//   reference; the GPU uses cuSOLVER for the same job). Returns eigenvalues
//   ascending and the matching eigenvectors as COLUMNS (column-major evec).
//   Clear and obviously correct -- exactly what a reference should be.
// ---------------------------------------------------------------------------
void symmetric_eigen(std::vector<double> A, int N,
                     std::vector<double>& eval, std::vector<double>& evec) {
    // V starts as the identity and accumulates every rotation -> eigenvectors.
    std::vector<double> V(static_cast<size_t>(N) * N, 0.0);
    for (int i = 0; i < N; ++i) V[static_cast<size_t>(i) * N + i] = 1.0;

    const int MAX_SWEEPS = 100;
    for (int sweep = 0; sweep < MAX_SWEEPS; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < N; ++p)
            for (int qq = p + 1; qq < N; ++qq)
                off += A[static_cast<size_t>(p) * N + qq] * A[static_cast<size_t>(p) * N + qq];
        if (off < 1e-24) break;                       // off-diagonals ~ 0 => done

        for (int p = 0; p < N; ++p) {
            for (int qq = p + 1; qq < N; ++qq) {
                const double apq = A[static_cast<size_t>(p) * N + qq];
                if (std::fabs(apq) < 1e-300) continue;
                const double app = A[static_cast<size_t>(p) * N + p];
                const double aqq = A[static_cast<size_t>(qq) * N + qq];
                const double theta = (aqq - app) / (2.0 * apq);
                const double t = (theta >= 0 ? 1.0 : -1.0) /
                                 (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;
                // Apply the Givens rotation J^T A J (columns then rows).
                for (int kk = 0; kk < N; ++kk) {
                    const double akp = A[static_cast<size_t>(kk) * N + p];
                    const double akq = A[static_cast<size_t>(kk) * N + qq];
                    A[static_cast<size_t>(kk) * N + p] = c * akp - s * akq;
                    A[static_cast<size_t>(kk) * N + qq] = s * akp + c * akq;
                }
                for (int kk = 0; kk < N; ++kk) {
                    const double apk = A[static_cast<size_t>(p) * N + kk];
                    const double aqk = A[static_cast<size_t>(qq) * N + kk];
                    A[static_cast<size_t>(p) * N + kk] = c * apk - s * aqk;
                    A[static_cast<size_t>(qq) * N + kk] = s * apk + c * aqk;
                }
                // Accumulate the rotation into V (eigenvector columns p, qq).
                for (int kk = 0; kk < N; ++kk) {
                    const double vkp = V[static_cast<size_t>(kk) * N + p];
                    const double vkq = V[static_cast<size_t>(kk) * N + qq];
                    V[static_cast<size_t>(kk) * N + p] = c * vkp - s * vkq;
                    V[static_cast<size_t>(kk) * N + qq] = s * vkp + c * vkq;
                }
            }
        }
    }

    // Collect (eigenvalue, original column) and sort ascending by eigenvalue.
    std::vector<std::pair<double,int>> order(N);
    for (int i = 0; i < N; ++i) order[i] = {A[static_cast<size_t>(i) * N + i], i};
    std::sort(order.begin(), order.end(),
              [](const std::pair<double,int>& a, const std::pair<double,int>& b) {
                  return a.first < b.first;
              });

    eval.resize(N);
    evec.assign(static_cast<size_t>(N) * N, 0.0);   // column-major: col k = vector k
    for (int k = 0; k < N; ++k) {
        eval[k] = order[k].first;
        const int src = order[k].second;
        for (int row = 0; row < N; ++row)
            evec[static_cast<size_t>(k) * N + row] = V[static_cast<size_t>(row) * N + src];
    }
}

// ---------------------------------------------------------------------------
// solve_generalized: solve F C = S C eps. The basis is non-orthogonal (S != I),
//   so we first ORTHOGONALIZE: form X = S^-1/2 (symmetric/Loewdin orthogonal-
//   ization), transform F' = X^T F X (now an ORDINARY eigenproblem), diagonalize
//   F' = C' eps C'^T, and back-transform C = X C'. This is exactly the recipe in
//   Szabo & Ostlund and in every textbook HF code.
// ---------------------------------------------------------------------------
void solve_generalized(const std::vector<double>& F, const std::vector<double>& S,
                       int N, std::vector<double>& C, std::vector<double>& eps) {
    // 1) Diagonalize S = U s U^T to build X = U s^-1/2 U^T.
    std::vector<double> s_val, s_vec;     // s_vec column-major
    symmetric_eigen(S, N, s_val, s_vec);
    std::vector<double> X(static_cast<size_t>(N) * N, 0.0);   // X = S^-1/2 (row-major)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                const double uik = s_vec[static_cast<size_t>(k) * N + i];  // U[i,k]
                const double ujk = s_vec[static_cast<size_t>(k) * N + j];  // U[j,k]
                sum += uik * (1.0 / std::sqrt(s_val[k])) * ujk;
            }
            X[static_cast<size_t>(i) * N + j] = sum;
        }

    // 2) Fp = X^T F X. (X is symmetric, so X^T = X, but we keep it explicit.)
    std::vector<double> FX(static_cast<size_t>(N) * N, 0.0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k)
                sum += F[static_cast<size_t>(i) * N + k] * X[static_cast<size_t>(k) * N + j];
            FX[static_cast<size_t>(i) * N + j] = sum;
        }
    std::vector<double> Fp(static_cast<size_t>(N) * N, 0.0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k)
                sum += X[static_cast<size_t>(k) * N + i] * FX[static_cast<size_t>(k) * N + j];
            Fp[static_cast<size_t>(i) * N + j] = sum;
        }

    // 3) Diagonalize Fp -> orbital energies eps and transformed coeffs Cp.
    std::vector<double> Cp;               // column-major
    symmetric_eigen(Fp, N, eps, Cp);

    // 4) Back-transform C = X Cp  (column k of C = MO k in the original basis).
    C.assign(static_cast<size_t>(N) * N, 0.0);   // column-major
    for (int k = 0; k < N; ++k)
        for (int row = 0; row < N; ++row) {
            double sum = 0.0;
            for (int m = 0; m < N; ++m)
                sum += X[static_cast<size_t>(row) * N + m] * Cp[static_cast<size_t>(k) * N + m];
            C[static_cast<size_t>(k) * N + row] = sum;
        }
}

// ---------------------------------------------------------------------------
// build_density: P = 2 * sum_{a in occupied} C_a C_a^T. The factor 2 is the two
//   electrons (spin up + down) in each doubly-occupied spatial orbital. P_{ij} is
//   the one-particle density matrix in the basis -- it is what couples the
//   electrons to each other through the Fock operator.
//   C is column-major (column k = MO k); we use the lowest n_occ columns.
// ---------------------------------------------------------------------------
void build_density(const std::vector<double>& C, int N, int n_occ,
                   std::vector<double>& P) {
    P.assign(static_cast<size_t>(N) * N, 0.0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int a = 0; a < n_occ; ++a)
                sum += C[static_cast<size_t>(a) * N + i] * C[static_cast<size_t>(a) * N + j];
            P[static_cast<size_t>(i) * N + j] = 2.0 * sum;
        }
}

// ---------------------------------------------------------------------------
// build_fock: F = Hcore + G(P), where the two-electron term is
//   G_{ij} = sum_{k,l} P_{kl} [ (ij|kl) - 0.5 (ik|jl) ].
//   The first term is the COULOMB repulsion (electrons feel the average charge
//   cloud); the second is EXCHANGE (a purely quantum effect from antisymmetry, the
//   exact-HF analogue of DFT's exchange-correlation functional). This contraction
//   is O(N^4) but uses the ALREADY-COMPUTED tensor, so it is cheap relative to
//   building the integrals.
// ---------------------------------------------------------------------------
void build_fock(const std::vector<double>& Hcore, const std::vector<double>& P,
                const std::vector<double>& eri, int N, std::vector<double>& F) {
    F = Hcore;                                        // start from the one-electron part
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double g = 0.0;
            for (int k = 0; k < N; ++k)
                for (int l = 0; l < N; ++l) {
                    const double coulomb  = eri[(((static_cast<size_t>(i) * N + j) * N + k) * N + l)];
                    const double exchange = eri[(((static_cast<size_t>(i) * N + k) * N + j) * N + l)];
                    g += P[static_cast<size_t>(k) * N + l] * (coulomb - 0.5 * exchange);
                }
            F[static_cast<size_t>(i) * N + j] += g;
        }
}

// ---------------------------------------------------------------------------
// run_scf: the self-consistent field loop (the heart of HF/DFT).
//   The Fock operator depends on the density, which depends on the orbitals,
//   which come from diagonalizing the Fock operator -- a chicken-and-egg problem
//   solved by ITERATION:
//     0. start from the core guess (P = 0  =>  F = Hcore)
//     repeat:
//       1. solve F C = S C eps          (generalized eigenproblem)
//       2. build P from the lowest n_occ orbitals
//       3. rebuild F = Hcore + G(P)
//       4. electronic energy E = 0.5 * sum_ij P_ij (Hcore_ij + F_ij)
//       until |E - E_prev| < e_tol
//   We use a transparent Jacobi-based generalized eigensolve here; main.cu runs
//   the SAME loop but swapping in cuSOLVER, and checks both converge to the same E.
// ---------------------------------------------------------------------------
ScfResult run_scf(const std::vector<double>& S, const std::vector<double>& Hcore,
                  const std::vector<double>& eri, int N, int n_occ,
                  double e_nuclear, int max_iter, double e_tol) {
    ScfResult res;
    res.e_nuclear = e_nuclear;

    std::vector<double> F = Hcore;        // core guess: ignore electron-electron at first
    std::vector<double> C, eps, P;
    double e_elec_prev = 0.0;

    for (int iter = 1; iter <= max_iter; ++iter) {
        // 1. Orbitals from the current Fock matrix.
        solve_generalized(F, S, N, C, eps);
        // 2. New density from the occupied orbitals.
        build_density(C, N, n_occ, P);
        // 3. New Fock matrix from that density.
        build_fock(Hcore, P, eri, N, F);
        // 4. Electronic energy with the updated P and F.
        double e_elec = 0.0;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                e_elec += 0.5 * P[static_cast<size_t>(i) * N + j] *
                          (Hcore[static_cast<size_t>(i) * N + j] + F[static_cast<size_t>(i) * N + j]);

        res.iterations = iter;
        res.orbital_energies = eps;
        res.e_electronic = e_elec;
        if (std::fabs(e_elec - e_elec_prev) < e_tol) {
            res.converged = true;
            break;
        }
        e_elec_prev = e_elec;
    }

    res.e_total = res.e_electronic + e_nuclear;
    // HOMO / LUMO: highest occupied & lowest unoccupied MO energies (the frontier
    // orbitals; their gap is a first estimate of optical/chemical reactivity).
    if (n_occ - 1 >= 0 && n_occ - 1 < N) res.homo = res.orbital_energies[n_occ - 1];
    if (n_occ < N)                       res.lumo = res.orbital_energies[n_occ];
    return res;
}
