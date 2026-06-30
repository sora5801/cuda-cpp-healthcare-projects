// ===========================================================================
// src/polar.h  --  Shared (host + device) polarizable-water electrostatics
// ---------------------------------------------------------------------------
// Project 2.27 : Polarizable Water Model GPU Dynamics
//
// WHAT THIS PROJECT COMPUTES
//   The defining, expensive step of every polarizable water model (SWM4-NDP,
//   AMOEBA, the Drude oscillator, and -- at the high end -- MB-pol's induction
//   term) is the SELF-CONSISTENT INDUCED-DIPOLE problem:
//
//     Every polarizable site i carries a fixed (permanent) point charge q_i AND
//     an INDUCIBLE point dipole mu_i. The dipole is induced by the local
//     electric field E_i at that site:
//
//                       mu_i = alpha_i * E_i                          (1)
//
//     where alpha_i is the atomic polarizability (a volume, units Angstrom^3 in
//     Gaussian-style units). The catch: E_i is produced by the permanent
//     charges of all OTHER sites PLUS the induced dipoles of all other sites --
//     so mu_i depends on every mu_j, which depends on mu_i. The dipoles must be
//     solved SELF-CONSISTENTLY. This is the "mutual polarization" SCF loop that
//     dominates the cost of polarizable MD.
//
//   We split the field into two pieces:
//     E_i = E_i^perm  (from the static charges q_j, FIXED -- computed once)
//         + E_i^dip   (from the induced dipoles mu_j, CHANGES every iteration)
//
//   and solve (1) by JACOBI ITERATION (a.k.a. fixed-point / Picard iteration):
//
//     mu_i^(k+1) = alpha_i * ( E_i^perm + sum_{j != i} T_ij . mu_j^(k) )   (2)
//
//   where T_ij is the dipole field tensor (below). We sweep until the dipoles
//   stop changing (max |mu^(k+1) - mu^(k)| < tol). Production codes use a
//   preconditioned conjugate gradient to converge in fewer iterations; Jacoby is
//   the transparent teaching version and is what we implement (THEORY.md §"Real
//   world" explains the CG upgrade the catalog mentions).
//
//   The "physics that must match bit-for-bit on CPU and GPU" lives HERE as
//   POLAR_HD (= __host__ __device__ under nvcc, nothing under the host compiler)
//   inline functions, so the CPU reference (reference_cpu.cpp) and the GPU kernel
//   (kernels.cu) evaluate IDENTICAL arithmetic and their dipoles agree to
//   round-off. This is the HD-macro idiom from docs/PATTERNS.md §2.
//
//   UNITS (a self-consistent Gaussian-flavored teaching system; THEORY.md §Math):
//     length        : Angstrom (A)
//     charge         : elementary charge (e)
//     polarizability : A^3
//     field          : e / A^2
//     dipole         : e * A
//     energy         : e^2 / A  (multiply by 332.0637 to get kcal/mol; we report
//                                in this internal unit and also in kcal/mol)
//   In these units Coulomb's constant is 1, which keeps the formulas clean and
//   the teaching focus on the algorithm rather than unit bookkeeping.
//
// READ THIS AFTER: util/cuda_check.cuh.  READ BEFORE: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#pragma once

#include <cmath>   // std::sqrt, std::fabs (host); device intrinsics under nvcc

// POLAR_HD expands to __host__ __device__ when compiled by nvcc (so the same
// inline function is emitted for BOTH the CPU reference and the GPU kernel), and
// to nothing under the plain host compiler (which does not know those keywords).
#ifdef __CUDACC__
#define POLAR_HD __host__ __device__
#else
#define POLAR_HD
#endif

// ---------------------------------------------------------------------------
// A 3-vector with just the arithmetic the field math needs. Kept tiny and
// trivially-copyable so it lives happily in registers on the device and in a
// std::vector on the host. We avoid CUDA's float3/double3 here so the host
// compiler (which never sees <vector_types.h>) can include this header too.
// ---------------------------------------------------------------------------
struct Vec3 {
    double x, y, z;
};

POLAR_HD inline Vec3 vadd(const Vec3& a, const Vec3& b) { return Vec3{a.x + b.x, a.y + b.y, a.z + b.z}; }
POLAR_HD inline Vec3 vsub(const Vec3& a, const Vec3& b) { return Vec3{a.x - b.x, a.y - b.y, a.z - b.z}; }
POLAR_HD inline Vec3 vscale(const Vec3& a, double s)    { return Vec3{a.x * s,   a.y * s,   a.z * s};   }
POLAR_HD inline double vdot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// ---------------------------------------------------------------------------
// One polarizable site: a permanent point charge q AND an inducible dipole.
//   pos   : Cartesian position (Angstrom).
//   q     : permanent charge (e). In SWM4-NDP a water has charges on H, the
//           M-site, and the polarizable Drude/oxygen carrier; here we keep the
//           per-site charge explicit so any fixed-charge layout can be loaded.
//   alpha : isotropic polarizability (A^3). Sites with alpha == 0 are pure
//           fixed charges (e.g. the bare hydrogens) and never carry a dipole.
// The induced dipole itself is NOT stored here -- it lives in a separate array
// that the Jacobi solver ping-pongs, so the geometry stays read-only.
// ---------------------------------------------------------------------------
struct Site {
    Vec3   pos;
    double q;
    double alpha;
};

// ---------------------------------------------------------------------------
// thole_lambda3 / thole_lambda5 : Thole short-range DAMPING factors.
//
//   Two point dipoles at very short range produce a divergent ("polarization
//   catastrophe") interaction -- the 1/r^3 tensor blows up and the SCF diverges.
//   Thole's fix is to smear each dipole over a short distance, which multiplies
//   the bare tensor's two terms by damping factors lambda3, lambda5 that -> 1 at
//   long range and smoothly soften the r->0 singularity. We use the common
//   exponential ("Thole-exp") form with screening parameter a:
//
//       u  = r / (alpha_i * alpha_j)^(1/6)      (dimensionless screened distance)
//       lambda3 = 1 - exp(-a u^3)
//       lambda5 = 1 - (1 + a u^3) exp(-a u^3)
//
//   These are the exact factors used in AMOEBA-style polarizable force fields.
//   Returning both from one helper keeps the dipole-tensor code below readable.
// ---------------------------------------------------------------------------
POLAR_HD inline void thole_lambdas(double r, double alpha_i, double alpha_j,
                                   double a_thole, double& lambda3, double& lambda5) {
    // Screened distance u. If either polarizability is ~0 the pair cannot be a
    // dipole-dipole pair anyway (the caller guards that), but we keep the
    // denominator safe regardless.
    const double aa = alpha_i * alpha_j;
    if (aa <= 0.0) { lambda3 = 1.0; lambda5 = 1.0; return; }   // no damping needed
    const double s = std::pow(aa, 1.0 / 6.0);                  // (alpha_i alpha_j)^(1/6)
    const double u = r / s;
    const double au3 = a_thole * u * u * u;                    // a * u^3
    const double e = std::exp(-au3);                           // exp(-a u^3)
    lambda3 = 1.0 - e;
    lambda5 = 1.0 - (1.0 + au3) * e;
}

// ---------------------------------------------------------------------------
// field_perm_pair : electric field at site i due to the PERMANENT charge of a
//   single other site j. This is the static (k-independent) part of E_i^perm and
//   is computed ONCE per configuration (it never changes during the SCF loop).
//
//     E_i += q_j * r_ij / |r_ij|^3        (r_ij = pos_i - pos_j)
//
//   Coulomb's constant is 1 in our unit system. `rij` points FROM j TO i so the
//   field of a positive charge pushes a test charge away -- the standard sign.
// ---------------------------------------------------------------------------
POLAR_HD inline Vec3 field_perm_pair(const Vec3& pos_i, const Vec3& pos_j, double q_j) {
    const Vec3 rij = vsub(pos_i, pos_j);          // i <- j
    const double r2 = vdot(rij, rij);
    const double r = std::sqrt(r2);
    const double inv_r3 = 1.0 / (r2 * r);         // 1 / r^3
    return vscale(rij, q_j * inv_r3);
}

// ---------------------------------------------------------------------------
// field_dip_pair : electric field at site i due to the induced DIPOLE mu_j of
//   one other site j, via the (Thole-damped) dipole field tensor:
//
//     E_i = T_ij . mu_j ,   T_ij = ( 3 (r r^T) lambda5 - I lambda3 ) / r^3
//
//   Expanded for a vector mu_j this is:
//
//     E_i = [ 3 lambda5 (rhat . mu_j) rhat - lambda3 mu_j ] / r^3
//
//   This is the term that couples the dipoles together and forces the
//   self-consistent (iterative) solve. It is recomputed every Jacobi sweep
//   because mu_j changes each sweep.
// ---------------------------------------------------------------------------
POLAR_HD inline Vec3 field_dip_pair(const Vec3& pos_i, const Vec3& pos_j,
                                    const Vec3& mu_j, double alpha_i, double alpha_j,
                                    double a_thole) {
    const Vec3 rij = vsub(pos_i, pos_j);          // i <- j
    const double r2 = vdot(rij, rij);
    const double r = std::sqrt(r2);
    const double inv_r3 = 1.0 / (r2 * r);         // 1 / r^3

    double lambda3, lambda5;
    thole_lambdas(r, alpha_i, alpha_j, a_thole, lambda3, lambda5);

    const double rdotmu = vdot(rij, mu_j);        // (r . mu_j)  [note: r not unit yet]
    // 3 lambda5 (r.mu)/r^2 * r  - lambda3 mu , all over r^3. Using r2 in the
    // first term's denominator turns (rhat.mu)rhat into (r.mu)r/r^2.
    const Vec3 term1 = vscale(rij, 3.0 * lambda5 * rdotmu / r2);
    const Vec3 term2 = vscale(mu_j, lambda3);
    return vscale(vsub(term1, term2), inv_r3);
}

// ---------------------------------------------------------------------------
// polarization_energy_site : the polarization (induction) energy contributed by
//   one converged dipole. The standard result for linear induced dipoles is
//
//       U_pol = -1/2 * sum_i  mu_i . E_i^perm                       (3)
//
//   i.e. minus one half the dot of each induced dipole with the PERMANENT field
//   that created it (the 1/2 is the work of charging up the dipole against its
//   own polarizability). This single number is our headline scalar result; it is
//   the induction energy that polarizable models add on top of fixed-charge
//   electrostatics, and it is what GPU acceleration of MB-pol/AMOEBA is chasing.
//   We sum it per site so the reduction is explicit.
// ---------------------------------------------------------------------------
POLAR_HD inline double polarization_energy_site(const Vec3& mu_i, const Vec3& Eperm_i) {
    return -0.5 * vdot(mu_i, Eperm_i);
}

// Conversion from our internal energy unit (e^2 / Angstrom) to kcal/mol. This is
// the familiar electrostatic prefactor 1/(4 pi eps0) expressed in these units;
// we report both the raw internal energy (used for the exact CPU==GPU check) and
// the kcal/mol value (the chemically meaningful number).
POLAR_HD inline double energy_to_kcal_per_mol(double e_internal) {
    return e_internal * 332.0637137; // e^2/A -> kcal/mol
}
