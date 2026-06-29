// ===========================================================================
// src/amoeba.h  --  Shared (host + device) AMOEBA induced-dipole physics + CG
// ---------------------------------------------------------------------------
// Project 1.21 : Polarizable / AMOEBA Force Field MD
//
// WHAT THIS PROJECT COMPUTES (the teaching core)
//   The single feature that makes the AMOEBA force field "polarizable" -- and the
//   single thing that makes it ~10x more expensive than a fixed-charge field like
//   AMBER -- is the SELF-CONSISTENT INDUCED-DIPOLE problem. Every atom carries an
//   isotropic polarizability alpha_i; placed in an electric field it acquires an
//   INDUCED dipole mu_i = alpha_i * E_total(i). But each induced dipole is itself
//   a tiny field source, so the field at atom i depends on the dipoles of all the
//   other atoms, which depend on the field at i, ... -> a coupled linear system
//   that must be solved to self-consistency at EVERY molecular-dynamics step.
//
//   Mutual-polarization equation (one 3-vector per atom):
//       mu_i = alpha_i * ( E_i^perm  +  sum_{j != i} T_ij . mu_j )
//   where
//       E_i^perm  = permanent electric field at atom i (from fixed charges),
//       T_ij      = 3x3 dipole-dipole interaction tensor between atoms i and j,
//                   T_ij = (3 r_ij r_ij^T - r^2 I) / r^5   (r_ij = x_j - x_i).
//
//   Rearranged into a linear system  A mu = b  with 3N unknowns:
//       (1/alpha_i) mu_i  -  sum_{j != i} T_ij . mu_j  =  E_i^perm
//   The operator A is symmetric (T_ij = T_ji^T and T is symmetric) and, for
//   physical polarizabilities and non-overlapping atoms, positive definite. That
//   is EXACTLY the class of system the CONJUGATE GRADIENT method was built for --
//   which is why Tinker-HP / OpenMM solve the AMOEBA induced dipoles with CG.
//
// WHY A SHARED HOST+DEVICE HEADER
//   The per-system physics -- the matrix-free operator `apply_A` and the whole CG
//   loop `solve_induced_dipoles` -- live here as `__host__ __device__` inline
//   functions. The CPU reference (reference_cpu.cpp) and the GPU kernel
//   (kernels.cu) therefore run BYTE-FOR-BYTE-IDENTICAL math, so verification is
//   essentially exact (see ../THEORY.md "How we verify correctness"). AMOEBA_HD
//   expands to `__host__ __device__` under nvcc and to nothing under the host
//   compiler. Keep CUDA-only constructs (no `__global__`, no `<<<>>>`) out of
//   this header so the plain C++ compiler can include it too.
//
// READ THIS AFTER: util/cuda_check.cuh; READ BEFORE: kernels.cuh, reference_cpu.h.
// ===========================================================================
#pragma once

#include <cmath>   // std::sqrt -- used in both the host reference and the device kernel

#ifdef __CUDACC__
#define AMOEBA_HD __host__ __device__
#else
#define AMOEBA_HD
#endif

// Hard cap on atoms per polarization system. Each thread integrates ONE system
// entirely in registers / local memory, so we need a compile-time bound to size
// the small fixed work arrays (dipoles, residual, search direction). 32 atoms is
// plenty for a teaching cluster (e.g. a small solute + a few waters) and keeps
// the per-thread footprint tiny. Real Tinker-HP solves 3N for N up to millions
// using a NEIGHBOR LIST + PME, not the dense O(N^2) loop we use here for clarity.
#define AMOEBA_MAX_ATOMS 32

// ---------------------------------------------------------------------------
// AtomSystem  --  one independent polarization problem (a tiny "molecule").
//   We store positions, the permanent field, and per-atom polarizability for up
//   to AMOEBA_MAX_ATOMS atoms. All systems in the ensemble share this layout so
//   one GPU thread can own one AtomSystem. Units are atomic-style "reduced"
//   units (Angstrom for length, atomic polarizability volume in Angstrom^3, the
//   permanent field in field units) -- this is a TEACHING model, not a
//   parameter-accurate AMOEBA implementation, so the constants are illustrative.
// ---------------------------------------------------------------------------
struct AtomSystem {
    int    n;                              // number of atoms in this system (<= MAX)
    double pos  [AMOEBA_MAX_ATOMS][3];     // atom positions [Angstrom]
    double Eperm[AMOEBA_MAX_ATOMS][3];     // permanent electric field at each atom
    double alpha[AMOEBA_MAX_ATOMS];        // atomic polarizability [Angstrom^3]
};

// ---------------------------------------------------------------------------
// PerSystemResult  --  the deterministic summary we report + verify per system.
//   We keep scalar, order-independent quantities so CPU and GPU agree exactly:
//     * iters       : CG iterations taken to reach the residual tolerance.
//     * upol         : the polarization energy  U = -1/2 * sum_i mu_i . E_i^perm
//                      (the energy stored in inducing the dipoles).
//     * mu_total[3]  : the net induced dipole moment (vector sum of mu_i).
//     * max_mu       : the largest single induced-dipole magnitude (a stability
//                      diagnostic: a runaway dipole signals "polarization
//                      catastrophe", the classic failure mode of mutual models).
// ---------------------------------------------------------------------------
struct PerSystemResult {
    int    iters;
    double upol;
    double mu_total[3];
    double max_mu;
};

// ---------------------------------------------------------------------------
// dipole_field_contrib: field at atom i produced by a unit-strength dipole `mu`
//   sitting on atom j, i.e. the action of the 3x3 interaction tensor T_ij on mu.
//   This is the kernel of the whole method -- it is the off-diagonal coupling in
//   the linear operator A.
//
//   Math:  contribution = (3 (r_hat . mu) r_hat - mu) / r^3
//          which equals T_ij . mu with T_ij = (3 r r^T - r^2 I)/r^5, r = xi - xj.
//   Here we pass r = (xi - xj) so r points FROM j TO i.
//
//   Params:
//     rx,ry,rz : components of r_ij = pos_i - pos_j           [Angstrom]
//     mx,my,mz : the dipole vector on atom j                  [dipole units]
//     out[3]   : (write) the field contribution at atom i     [field units]
//   No allocation, no side effects beyond `out`. O(1).
// ---------------------------------------------------------------------------
AMOEBA_HD inline void dipole_field_contrib(double rx, double ry, double rz,
                                           double mx, double my, double mz,
                                           double out[3]) {
    const double r2  = rx*rx + ry*ry + rz*rz;   // |r|^2
    const double r   = ::sqrt(r2);              // |r|
    const double r3  = r2 * r;                  // |r|^3
    const double r5  = r3 * r2;                 // |r|^5
    // dot = r . mu  (numerator of the 3 (r_hat.mu) r_hat term, pre-scaling)
    const double dot = rx*mx + ry*my + rz*mz;
    // T_ij . mu = (3 (r.mu) r - r^2 mu) / r^5. Written so it is exactly the same
    // sequence of FLOPs on host and device (no library calls) -> bitwise parity.
    out[0] = (3.0 * dot * rx - r2 * mx) / r5;
    out[1] = (3.0 * dot * ry - r2 * my) / r5;
    out[2] = (3.0 * dot * rz - r2 * mz) / r5;
    (void)r3;   // r3 kept for readability of the derivation; not needed directly
}

// ---------------------------------------------------------------------------
// apply_A: the matrix-free linear operator y = A x for the induced-dipole system.
//   A x has, per atom i:
//       (A x)_i = (1/alpha_i) x_i  -  sum_{j != i} T_ij . x_j
//   We never build the 3N x 3N matrix A; we just know how to MULTIPLY by it. That
//   is the essence of a matrix-free Krylov solver and is why CG fits the GPU so
//   well: each matvec is an O(N^2) all-pairs loop with no global storage.
//
//   Params:
//     s     : the AtomSystem (positions, polarizabilities)   [in]
//     x     : input vector, x[i][k] = component k of atom i   [in]
//     y     : output vector y = A x, same layout              [out]
//   Complexity: O(n^2) per call (the double loop over atom pairs).
// ---------------------------------------------------------------------------
AMOEBA_HD inline void apply_A(const AtomSystem& s,
                              const double x[AMOEBA_MAX_ATOMS][3],
                              double       y[AMOEBA_MAX_ATOMS][3]) {
    for (int i = 0; i < s.n; ++i) {
        // Diagonal block: (1/alpha_i) * x_i. The "self" term of A.
        double yi0 = x[i][0] / s.alpha[i];
        double yi1 = x[i][1] / s.alpha[i];
        double yi2 = x[i][2] / s.alpha[i];
        // Off-diagonal: subtract the field that every OTHER dipole x_j induces at
        // atom i. This is the coupling that makes the dipoles mutually dependent.
        for (int j = 0; j < s.n; ++j) {
            if (j == i) continue;
            const double rx = s.pos[i][0] - s.pos[j][0];
            const double ry = s.pos[i][1] - s.pos[j][1];
            const double rz = s.pos[i][2] - s.pos[j][2];
            double f[3];
            dipole_field_contrib(rx, ry, rz, x[j][0], x[j][1], x[j][2], f);
            yi0 -= f[0];
            yi1 -= f[1];
            yi2 -= f[2];
        }
        y[i][0] = yi0;
        y[i][1] = yi1;
        y[i][2] = yi2;
    }
}

// ---------------------------------------------------------------------------
// dot3N: inner product of two 3N-vectors (flattened atom-major). The CG method
//   needs two dot products per iteration (for the step length alpha and the
//   conjugacy coefficient beta). We sum in a FIXED order (atom 0..n-1, component
//   0..2) so the host and device produce the same rounding -> deterministic.
// ---------------------------------------------------------------------------
AMOEBA_HD inline double dot3N(int n,
                              const double a[AMOEBA_MAX_ATOMS][3],
                              const double b[AMOEBA_MAX_ATOMS][3]) {
    double acc = 0.0;
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < 3; ++k)
            acc += a[i][k] * b[i][k];
    return acc;
}

// ---------------------------------------------------------------------------
// solve_induced_dipoles: the heart of the project -- a matrix-free CONJUGATE
//   GRADIENT solver for  A mu = b,  with  b_i = E_i^perm  and  A as in apply_A.
//
//   CG is the canonical Krylov method for a symmetric positive-definite system.
//   It builds a sequence of search directions that are A-orthogonal (conjugate),
//   so it reduces the error in the optimal way given only matrix-vector products.
//   For an n-atom system the system is 3n-dimensional, so CG converges in at most
//   3n exact-arithmetic steps -- usually far fewer, because the dipole couplings
//   are weak (the spectrum of A is tightly clustered around the 1/alpha diagonal).
//
//   We seed with the DIRECT (uncoupled) guess mu_i = alpha_i E_i^perm -- the
//   field a fixed-charge model would give -- and let CG add the mutual coupling.
//   That is also exactly how production codes warm-start the SCF iteration.
//
//   Params:
//     s        : the polarization system                                  [in]
//     tol      : convergence threshold on the relative residual norm      [in]
//     max_iter : safety cap on iterations (<= 3*MAX_ATOMS)                 [in]
//     mu       : (out) the converged induced dipoles, mu[i][k]            [out]
//   Returns: PerSystemResult (iters + energy + net/peak dipole diagnostics).
//
//   Memory: everything is on-stack fixed arrays of size MAX_ATOMS -> the whole
//   solve lives in registers/local memory of ONE thread. No global traffic, no
//   atomics, no shared memory: a textbook "thread per independent job" mapping.
// ---------------------------------------------------------------------------
AMOEBA_HD inline PerSystemResult solve_induced_dipoles(const AtomSystem& s,
                                                       double tol, int max_iter,
                                                       double mu[AMOEBA_MAX_ATOMS][3]) {
    const int n = s.n;

    // r = b - A mu   (residual);   p = r  (initial search direction);
    // Ap = A p (recomputed each iter). b is the permanent field E^perm.
    double r [AMOEBA_MAX_ATOMS][3];
    double p [AMOEBA_MAX_ATOMS][3];
    double Ap[AMOEBA_MAX_ATOMS][3];

    // --- Initial guess: the direct (non-mutual) dipoles mu_i = alpha_i E_i. ---
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < 3; ++k)
            mu[i][k] = s.alpha[i] * s.Eperm[i][k];

    // --- r0 = b - A mu0,  p0 = r0 ---
    apply_A(s, mu, Ap);                  // Ap temporarily holds A*mu0
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < 3; ++k) {
            r[i][k] = s.Eperm[i][k] - Ap[i][k];
            p[i][k] = r[i][k];
        }

    double rs_old = dot3N(n, r, r);      // <r,r>; the squared residual norm
    // Squared tolerance relative to the right-hand side, so `tol` is a relative
    // residual: we stop when |r| <= tol * |b|. Comparing squares avoids a sqrt
    // in the hot loop and keeps host/device arithmetic identical. The RHS b is
    // the permanent field E^perm, so |b|^2 is just its sum of squared components.
    double bnorm2 = 0.0;
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < 3; ++k)
            bnorm2 += s.Eperm[i][k] * s.Eperm[i][k];
    const double stop2 = tol * tol * (bnorm2 > 0.0 ? bnorm2 : 1.0);

    int it = 0;
    for (; it < max_iter; ++it) {
        if (rs_old <= stop2) break;      // converged: residual small enough

        apply_A(s, p, Ap);               // Ap = A p   (the one matvec per iter)
        const double pAp = dot3N(n, p, Ap);
        // alpha = <r,r> / <p, A p>  -- the exact line-search step length that
        // minimizes the A-norm of the error along direction p.
        const double alpha = rs_old / pAp;

        // mu += alpha p ;  r -= alpha Ap
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < 3; ++k) {
                mu[i][k] += alpha * p[i][k];
                r [i][k] -= alpha * Ap[i][k];
            }

        const double rs_new = dot3N(n, r, r);
        // beta = <r_new,r_new> / <r_old,r_old> -- makes the next direction
        // A-conjugate to all previous ones (Fletcher-Reeves form).
        const double beta = rs_new / rs_old;
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < 3; ++k)
                p[i][k] = r[i][k] + beta * p[i][k];

        rs_old = rs_new;
    }

    // --- Assemble the deterministic per-system summary ----------------------
    PerSystemResult out;
    out.iters = it;
    out.mu_total[0] = out.mu_total[1] = out.mu_total[2] = 0.0;
    double maxmu = 0.0;
    double energy = 0.0;
    for (int i = 0; i < n; ++i) {
        // Polarization energy U = -1/2 sum_i mu_i . E_i^perm. The 1/2 is because
        // half the interaction energy goes into the work of polarizing the atom.
        energy -= 0.5 * (mu[i][0]*s.Eperm[i][0]
                       + mu[i][1]*s.Eperm[i][1]
                       + mu[i][2]*s.Eperm[i][2]);
        out.mu_total[0] += mu[i][0];
        out.mu_total[1] += mu[i][1];
        out.mu_total[2] += mu[i][2];
        const double m2 = mu[i][0]*mu[i][0] + mu[i][1]*mu[i][1] + mu[i][2]*mu[i][2];
        const double m  = ::sqrt(m2);
        if (m > maxmu) maxmu = m;
    }
    out.upol   = energy;
    out.max_mu = maxmu;
    return out;
}
