// ===========================================================================
// src/reference_cpu.cpp  --  The trusted CPU baseline (no CUDA here)
// ---------------------------------------------------------------------------
// Project 1.8 : Semi-Empirical & Tight-Binding Quantum Methods
//
// WHAT THIS FILE DOES
//   Implements the whole tight-binding pipeline on the CPU so we have a
//   reference to verify the GPU against:
//     load_batch()        -- parse the tiny committed sample into a padded batch
//     build_hamiltonian() -- adjacency -> Huckel matrix (via the shared core)
//     jacobi_eigen()      -- diagonalise it (classic cyclic Jacobi)
//     analyze_molecule()  -- fill electrons, total pi energy, HOMO-LUMO gap
//
//   Compiled by the PLAIN host C++ compiler (cl.exe / g++), so it must contain
//   NO CUDA syntax. It includes tight_binding.h for the exact same matrix-build
//   formula the GPU uses -- that shared core is what makes verification exact.
//
// READ THIS AFTER: tight_binding.h, reference_cpu.h.  THEN: kernels.cu, main.cu.
// ===========================================================================
#include "reference_cpu.h"
#include "tight_binding.h"     // tb_hamiltonian_entry, tb_num_pi_electrons, TB_ALPHA

#include <algorithm>   // std::sort, std::max
#include <cmath>       // std::sqrt, std::fabs
#include <cstddef>     // std::size_t
#include <fstream>     // std::ifstream
#include <sstream>     // std::istringstream
#include <stdexcept>   // std::runtime_error
#include <string>
#include <utility>     // std::pair
#include <vector>

// ===========================================================================
// load_batch  --  parse the committed sample file
// ---------------------------------------------------------------------------
// SAMPLE FORMAT (whitespace/line based; see data/README.md):
//   * Lines beginning with '#' are comments and are skipped.
//   * First non-comment token: NUM_MOL  (the molecule count).
//   * Then, for each molecule:
//       NAME  N  NBONDS
//       NBONDS lines, each "i j"  (a 0-based bond between atoms i and j)
//
//   We read every molecule's bond list, find max_n across the batch, allocate
//   the padded adjacency cube, and stamp each molecule's bonds (symmetrically)
//   into its block. Padding atoms (index >= n) keep an all-zero row/col, i.e.
//   they are isolated.
// ===========================================================================
MoleculeBatch load_batch(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open input file: " + path);

    // --- helper: pull the next non-comment, non-blank line into `line` -----
    auto next_line = [&](std::string& line) -> bool {
        while (std::getline(in, line)) {
            // strip a trailing '\r' so Windows-CRLF files parse on any OS
            if (!line.empty() && line.back() == '\r') line.pop_back();
            // find first non-space to test for blank / comment
            std::size_t p = line.find_first_not_of(" \t");
            if (p == std::string::npos) continue;          // blank
            if (line[p] == '#') continue;                  // comment
            return true;
        }
        return false;
    };

    std::string line;
    if (!next_line(line)) throw std::runtime_error("empty sample file: " + path);

    int num_mol = 0;
    {
        std::istringstream ss(line);
        if (!(ss >> num_mol) || num_mol <= 0)
            throw std::runtime_error("bad molecule count in: " + path);
    }

    // First pass: read each molecule's (name, n, bond-list) into temporaries.
    std::vector<std::string>                      names(num_mol);
    std::vector<int>                              atoms(num_mol);
    std::vector<std::vector<std::pair<int,int>>>  bonds(num_mol);
    int max_n = 0;

    for (int m = 0; m < num_mol; ++m) {
        if (!next_line(line))
            throw std::runtime_error("unexpected EOF reading molecule header");
        std::istringstream ss(line);
        int n = 0, nb = 0;
        if (!(ss >> names[m] >> n >> nb) || n <= 0 || nb < 0)
            throw std::runtime_error("bad molecule header: " + line);
        atoms[m] = n;
        max_n = std::max(max_n, n);
        bonds[m].reserve(nb);
        for (int b = 0; b < nb; ++b) {
            if (!next_line(line))
                throw std::runtime_error("unexpected EOF reading bonds");
            std::istringstream bs(line);
            int i = 0, j = 0;
            if (!(bs >> i >> j) || i < 0 || j < 0 || i >= n || j >= n || i == j)
                throw std::runtime_error("bad bond line: " + line);
            bonds[m].push_back({i, j});
        }
    }

    // Second pass: allocate the padded cube and stamp bonds symmetrically.
    MoleculeBatch batch;
    batch.num_mol = num_mol;
    batch.max_n   = max_n;
    batch.name    = names;
    batch.n       = atoms;
    // zero-initialise so padding and non-bonds are 0 automatically.
    batch.adj.assign((std::size_t)num_mol * max_n * max_n, (unsigned char)0);

    for (int m = 0; m < num_mol; ++m) {
        unsigned char* A = &batch.adj[(std::size_t)m * max_n * max_n];
        for (const auto& e : bonds[m]) {
            A[(std::size_t)e.first  * max_n + e.second] = 1;   // i->j
            A[(std::size_t)e.second * max_n + e.first ] = 1;   // j->i (symmetric)
        }
    }
    return batch;
}

// ===========================================================================
// build_hamiltonian  --  adjacency -> Huckel matrix for one molecule
// ---------------------------------------------------------------------------
// We fill the FULL padded max_n x max_n matrix. For atom indices within the
// molecule (< n_real), tb_hamiltonian_entry() returns alpha on the diagonal and
// beta on a bonded off-diagonal. For PADDING indices (>= n_real) it returns the
// large TB_PAD_DIAG on the diagonal so those orbitals' eigenvalues sit far above
// the physical spectrum and never interleave with it (see tight_binding.h).
//
// The matrix is symmetric, so its row-major and column-major storage are byte-
// identical -- which is why the GPU can upload this same buffer straight into
// cuSOLVER's column-major expectation with no transpose.
// ===========================================================================
void build_hamiltonian(const MoleculeBatch& batch, int mol, std::vector<double>& H) {
    const int N      = batch.max_n;          // padded leading dimension
    const int n_real = batch.n[mol];         // this molecule's true atom count
    const unsigned char* A = &batch.adj[(std::size_t)mol * N * N];
    H.assign((std::size_t)N * N, 0.0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            H[(std::size_t)i * N + j] = tb_hamiltonian_entry(i, j, A, N, n_real);
}

// ===========================================================================
// jacobi_eigen  --  cyclic Jacobi diagonalisation of a real symmetric matrix
// ---------------------------------------------------------------------------
// WHY JACOBI (and not, say, QR): it is short, self-contained, numerically
// robust for small symmetric matrices, and DETERMINISTIC -- exactly what a
// teaching reference wants. It repeatedly applies Givens rotations that zero an
// off-diagonal pair, accumulating the rotations into the eigenvector matrix. We
// sweep all (p,q) pairs until the off-diagonal mass is negligible.
//
// Complexity: O(n^3) per sweep, a handful of sweeps -> O(n^3) overall.
//
// OUTPUT LAYOUT: we deliver eigenpairs sorted ascending and store eigenvectors
// COLUMN-major (evec[k*n + i]) so the layout matches cuSOLVER's Dsyevj output
// exactly -- making a future eigenvector comparison apples-to-apples.
// ===========================================================================
void jacobi_eigen(const std::vector<double>& A, int n,
                  std::vector<double>& eval, std::vector<double>& evec) {
    // Work on a mutable copy `a` (row-major). `v` accumulates eigenvectors.
    std::vector<double> a = A;
    std::vector<double> v((std::size_t)n * n, 0.0);
    for (int i = 0; i < n; ++i) v[(std::size_t)i * n + i] = 1.0;   // V = identity

    const int    MAX_SWEEPS = 100;       // ample; convergence is quadratic
    const double EPS        = 1e-300;    // floor to skip already-zero pairs

    for (int sweep = 0; sweep < MAX_SWEEPS; ++sweep) {
        // off = sqrt(sum of squares of upper-triangle off-diagonals). When this
        // is ~0 the matrix is (numerically) diagonal and we are done.
        double off = 0.0;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q)
                off += a[(std::size_t)p * n + q] * a[(std::size_t)p * n + q];
        off = std::sqrt(off);
        if (off < 1e-15) break;          // converged

        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const double apq = a[(std::size_t)p * n + q];
                if (std::fabs(apq) < EPS) continue;   // already (near) zero
                const double app = a[(std::size_t)p * n + p];
                const double aqq = a[(std::size_t)q * n + q];
                // Compute the Jacobi rotation that zeros a[p][q].
                //   theta = (aqq-app)/(2*apq);  t = sign(theta)/(|theta|+sqrt(theta^2+1))
                const double theta = (aqq - app) / (2.0 * apq);
                const double t = (theta >= 0.0 ? 1.0 : -1.0) /
                                 (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);   // cos
                const double s = t * c;                          // sin

                // Apply the rotation to columns p,q of `a`.
                for (int k = 0; k < n; ++k) {
                    const double akp = a[(std::size_t)k * n + p];
                    const double akq = a[(std::size_t)k * n + q];
                    a[(std::size_t)k * n + p] = c * akp - s * akq;
                    a[(std::size_t)k * n + q] = s * akp + c * akq;
                }
                // Apply the rotation to rows p,q of `a`.
                for (int k = 0; k < n; ++k) {
                    const double apk = a[(std::size_t)p * n + k];
                    const double aqk = a[(std::size_t)q * n + k];
                    a[(std::size_t)p * n + k] = c * apk - s * aqk;
                    a[(std::size_t)q * n + k] = s * apk + c * aqk;
                }
                // Accumulate the rotation into the eigenvector matrix V.
                for (int k = 0; k < n; ++k) {
                    const double vkp = v[(std::size_t)k * n + p];
                    const double vkq = v[(std::size_t)k * n + q];
                    v[(std::size_t)k * n + p] = c * vkp - s * vkq;
                    v[(std::size_t)k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    // Eigenvalues sit on the diagonal of the (now diagonal) `a`.
    // Build an index permutation that sorts them ascending (deterministic; ties
    // broken by original index so the order is stable run-to-run).
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int x, int y) {
        const double ex = a[(std::size_t)x * n + x];
        const double ey = a[(std::size_t)y * n + y];
        if (ex != ey) return ex < ey;
        return x < y;
    });

    eval.assign(n, 0.0);
    evec.assign((std::size_t)n * n, 0.0);
    for (int k = 0; k < n; ++k) {
        const int src = order[k];
        eval[k] = a[(std::size_t)src * n + src];
        // store eigenvector column-major: column k holds MO k's coefficients
        for (int i = 0; i < n; ++i)
            evec[(std::size_t)k * n + i] = v[(std::size_t)i * n + src];
    }
}

// ===========================================================================
// analyze_molecule  --  fill electrons, compute energy and HOMO-LUMO gap
// ---------------------------------------------------------------------------
// Aufbau filling: pi electrons go two-at-a-time into the lowest MOs. With
// n_atoms physical MOs and n_atoms electrons (one per carbon), there are
// n_atoms/2 doubly-occupied MOs (plus one singly-occupied MO if n_atoms is odd,
// a radical). We sum 2*energy over fully-occupied MOs (+ 1*energy for a half-
// filled one), and read HOMO (last occupied) and LUMO (first empty) for the gap.
//
// Only the FIRST n_atoms eigenvalues are physical; the padded MOs (energy =
// alpha) are skipped by construction because we never index past n_atoms.
// ===========================================================================
MoleculeResult analyze_molecule(const std::vector<double>& eval, int n_atoms) {
    MoleculeResult r;
    r.n_atoms = n_atoms;

    const int n_elec = tb_num_pi_electrons(n_atoms);   // = n_atoms (neutral)
    double energy = 0.0;
    int remaining = n_elec;
    int homo_idx = -1, lumo_idx = -1;

    // Walk MOs from the bottom, depositing up to 2 electrons each.
    for (int k = 0; k < n_atoms && remaining > 0; ++k) {
        const int occ = (remaining >= 2) ? 2 : remaining;   // 2, or 1 for a radical
        energy += occ * eval[k];
        remaining -= occ;
        homo_idx = k;                                       // last MO we put e- in
    }
    // LUMO is the first MO above the HOMO (if any unoccupied MO exists).
    if (homo_idx + 1 < n_atoms) lumo_idx = homo_idx + 1;

    r.total_pi_energy = energy;
    r.homo_energy = (homo_idx >= 0) ? eval[homo_idx] : 0.0;
    r.lumo_energy = (lumo_idx >= 0) ? eval[lumo_idx] : r.homo_energy;
    r.homo_lumo_gap = r.lumo_energy - r.homo_energy;
    return r;
}
