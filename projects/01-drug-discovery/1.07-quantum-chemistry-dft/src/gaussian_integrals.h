// ===========================================================================
// src/gaussian_integrals.h  --  The ONE TRUE per-integral physics (CPU == GPU)
// ---------------------------------------------------------------------------
// Project 1.7 : Quantum Chemistry / DFT  (reduced-scope teaching version: RHF
//               self-consistent field on a minimal Gaussian basis -- see THEORY.md)
//
// WHY THIS HEADER EXISTS  (PATTERNS.md  §2  "the shared __host__ __device__ core")
//   Density Functional Theory and its simpler cousin Hartree-Fock both reduce to
//   the SAME computational skeleton: build a matrix of one- and two-electron
//   INTEGRALS over basis functions, then solve a generalized eigenproblem
//   self-consistently. The dominant cost -- the one the catalog calls out as the
//   O(N^4) bottleneck -- is the TWO-ELECTRON repulsion integrals (ERIs). We put
//   the closed-form formula for one primitive-Gaussian integral HERE, in a single
//   header, decorated `__host__ __device__`, so that:
//
//     * reference_cpu.cpp  (compiled by the host C++ compiler) and
//     * kernels.cu         (compiled by nvcc for the GPU)
//
//   call the EXACT SAME inline functions. The CPU reference and the GPU kernel
//   therefore run byte-for-byte identical arithmetic, which makes verification
//   EXACT rather than approximate (THEORY.md "How we verify correctness").
//
// WHAT'S A "PRIMITIVE GAUSSIAN"?
//   Real atomic orbitals look like exp(-r): a cusp at the nucleus, a long tail.
//   That shape makes integrals hard. The trick (Boys, 1950) that made quantum
//   chemistry tractable is to APPROXIMATE each orbital by a fixed sum of GAUSSIANS
//   exp(-alpha r^2). Gaussians have a magic property: the product of two Gaussians
//   centered at A and B is itself ONE Gaussian centered between them (the
//   "Gaussian product theorem"). That collapses every 1- and 2-electron integral
//   to a closed form -- no numerical quadrature needed. We implement only s-type
//   (spherically symmetric) Gaussians here; that is enough for H and He and keeps
//   every formula short and readable. Higher angular momenta (p, d, ...) add
//   polynomial prefactors and are described in THEORY.md "real world".
//
//   A CONTRACTED basis function is a fixed linear combination of primitives:
//       phi(r) = sum_k  c_k * N(alpha_k) * exp(-alpha_k |r - center|^2)
//   The contraction coefficients c_k and exponents alpha_k come from a standard
//   basis set (we use STO-3G: "Slater orbital approximated by 3 Gaussians").
//
// UNITS: ATOMIC UNITS throughout (Hartree atomic units). Distances are in BOHR
//   (1 Bohr = 0.529 Angstrom), energies in HARTREE (1 Ha = 27.211 eV). In these
//   units hbar = m_e = e = 4*pi*eps0 = 1, which is exactly why the formulas below
//   have no physical constants cluttering them.
//
// READ THIS BEFORE: reference_cpu.h (the molecule + basis structs that feed these
//   formulas) and kernels.cuh (the GPU ERI kernel that calls eri_primitive()).
// ===========================================================================
#pragma once

#include <cmath>   // std::exp, std::sqrt, std::erf, std::tgamma

// ---------------------------------------------------------------------------
// HD: the "host-device" decorator macro (PATTERNS.md §2 idiom).
//   When this header is compiled by nvcc (__CUDACC__ is defined), every function
//   below is tagged `__host__ __device__` so it can run on BOTH the CPU and inside
//   a GPU kernel. When compiled by the plain host compiler (reference_cpu.cpp),
//   those CUDA keywords do not exist, so the macro expands to nothing. One source
//   of truth, two compilers, identical math.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// Pi to double precision. We avoid M_PI because it is not guaranteed by the C++
// standard on every compiler (it lives behind _USE_MATH_DEFINES on MSVC).
#ifndef QC_PI
#define QC_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// boys_f0(t)  --  the Boys function F0(t), the special function at the heart of
//   every Coulomb (nuclear-attraction and electron-repulsion) integral.
//
//   F0(t) = integral_0^1 exp(-t u^2) du.  It has the closed form
//       F0(t) = (1/2) * sqrt(pi/t) * erf(sqrt(t))      for t > 0,
//       F0(0) = 1                                       (the limit as t -> 0).
//
//   WHY IT APPEARS: the Coulomb operator 1/r12 is not Gaussian, so it cannot be
//   integrated by the product theorem directly. Boys' insight was to write
//   1/r = (2/sqrt(pi)) * integral_0^inf exp(-r^2 u^2) du, turning the troublesome
//   1/r into a Gaussian in u that DOES combine with the orbital Gaussians. After
//   the orbital integrals are done analytically, what remains is the u-integral
//   F0(t). For s-functions only F0 is needed; p,d,... need F1,F2,... (THEORY.md).
//
//   NUMERICS: near t=0, sqrt(pi/t)*erf(sqrt t) is 0/0 in floating point, so we
//   switch to the Taylor series F0(t) = 1 - t/3 + t^2/10 - ... for small t. This
//   keeps the function smooth and accurate to ~1e-12 across the whole range.
// ---------------------------------------------------------------------------
HD inline double boys_f0(double t) {
    if (t < 1.0e-12) {
        // Taylor series about t = 0 (first three terms are ample at this scale).
        return 1.0 - t / 3.0 + t * t / 10.0;
    }
    const double st = std::sqrt(t);
    // 0.5 * sqrt(pi)/sqrt(t) * erf(sqrt t). erf is in <cmath> (C++11).
    return 0.5 * std::sqrt(QC_PI) / st * std::erf(st);
}

// ---------------------------------------------------------------------------
// gauss_norm(alpha)  --  normalization constant N of a single s-type primitive
//   exp(-alpha r^2), i.e. the factor making integral |N exp(-alpha r^2)|^2 = 1.
//   For a 3-D s-Gaussian this is (2 alpha / pi)^(3/4). We fold this into every
//   primitive so the contracted basis functions come out (very nearly) normalized.
// ---------------------------------------------------------------------------
HD inline double gauss_norm(double alpha) {
    return std::pow(2.0 * alpha / QC_PI, 0.75);
}

// ---------------------------------------------------------------------------
// Below: the four kinds of integral over a PAIR (or quartet) of s-PRIMITIVES.
// Each takes raw exponents `a,b,...`, centers `(Ax,Ay,Az),...`, and returns the
// integral for UNNORMALIZED exp(-a r^2) primitives; the caller multiplies in the
// contraction coefficients and normalization. Keeping these primitive-level and
// branch-free is exactly what lets one GPU thread evaluate one quartet (kernels.cu).
//
// Notation used in the formulas (Gaussian product theorem for centers A,B):
//     p   = a + b                         (combined exponent)
//     P   = (a*A + b*B) / p               (combined center, between A and B)
//     AB2 = |A - B|^2                     (squared center separation)
//     K   = exp(-a*b/p * AB2)             (the overlap "pre-exponential")
// ---------------------------------------------------------------------------

// dist2: squared distance between two 3-D points. Tiny helper, inlined.
HD inline double dist2(double ax, double ay, double az,
                       double bx, double by, double bz) {
    const double dx = ax - bx, dy = ay - by, dz = az - bz;
    return dx * dx + dy * dy + dz * dz;
}

// --- (1) OVERLAP  S_ab = integral exp(-a|r-A|^2) exp(-b|r-B|^2) d^3r ----------
//   Product theorem => one Gaussian of exponent p; the space integral of a 3-D
//   Gaussian is (pi/p)^(3/2). The whole thing is (pi/p)^1.5 * exp(-ab/p * AB2).
//   Physically: how much two basis functions "overlap" -- the metric S in FC=SCe.
HD inline double overlap_primitive(double a, double ax, double ay, double az,
                                   double b, double bx, double by, double bz) {
    const double p   = a + b;
    const double ab2 = dist2(ax, ay, az, bx, by, bz);
    const double K   = std::exp(-a * b / p * ab2);
    return std::pow(QC_PI / p, 1.5) * K;
}

// --- (2) KINETIC  T_ab = integral exp(-a..) * (-1/2 nabla^2) exp(-b..) d^3r ----
//   The electron's kinetic-energy operator is -1/2 Laplacian. For s-Gaussians the
//   result has the compact closed form below (a standard textbook identity):
//       T = a*b/p * (3 - 2*a*b/p*AB2) * S_ab
//   It reuses the overlap S_ab, so kinetic energy costs essentially nothing extra.
HD inline double kinetic_primitive(double a, double ax, double ay, double az,
                                   double b, double bx, double by, double bz) {
    const double p   = a + b;
    const double ab2 = dist2(ax, ay, az, bx, by, bz);
    const double S   = overlap_primitive(a, ax, ay, az, b, bx, by, bz);
    const double xi  = a * b / p;            // reduced exponent
    return xi * (3.0 - 2.0 * xi * ab2) * S;
}

// --- (3) NUCLEAR ATTRACTION  V_ab^C = -Z_C integral [phi_a phi_b / |r - C|] -----
//   Attraction of the electron density phi_a*phi_b to a nucleus of charge Z at C.
//   The 1/|r-C| Coulomb operator brings in the Boys function:
//       V = -Z * (2*pi/p) * exp(-ab/p AB2) * F0( p * |P - C|^2 )
//   where P is the combined center. The (2*pi/p) replaces the overlap's (pi/p)^1.5
//   because one spatial integral has been "used up" by the Boys transform.
HD inline double nuclear_primitive(double a, double ax, double ay, double az,
                                   double b, double bx, double by, double bz,
                                   double Z, double cx, double cy, double cz) {
    const double p   = a + b;
    const double ab2 = dist2(ax, ay, az, bx, by, bz);
    // Combined center P = (a*A + b*B)/p (Gaussian product theorem).
    const double px = (a * ax + b * bx) / p;
    const double py = (a * ay + b * by) / p;
    const double pz = (a * az + b * bz) / p;
    const double pc2 = dist2(px, py, pz, cx, cy, cz);  // |P - C|^2
    const double K   = std::exp(-a * b / p * ab2);
    return -Z * (2.0 * QC_PI / p) * K * boys_f0(p * pc2);
}

// --- (4) TWO-ELECTRON REPULSION  (ab|cd) -- THE O(N^4) BOTTLENECK -------------
//   The repulsion between charge cloud (phi_a phi_b) of electron 1 and cloud
//   (phi_c phi_d) of electron 2:
//       (ab|cd) = integral integral [phi_a phi_b](r1) (1/r12) [phi_c phi_d](r2).
//   With the product theorem applied to BOTH pairs (giving centers P,Q and
//   exponents p,q) and Boys handling 1/r12, the closed form for s-primitives is:
//
//       (ab|cd) = 2*pi^2.5 / (p*q*sqrt(p+q))
//                 * exp(-a*b/p AB2) * exp(-c*d/q CD2)
//                 * F0( p*q/(p+q) * |P - Q|^2 )
//
//   This single function, evaluated for every quartet (a,b,c,d) of basis
//   functions, is the heart of the whole calculation -- there are O(N^4) of them,
//   and on the GPU we give EACH ONE ITS OWN THREAD (kernels.cu). Because it is the
//   identical inline used by reference_cpu.cpp, the CPU and GPU ERIs are bitwise
//   equal, so the SCF energies match to ~1e-12 (verification is essentially exact).
HD inline double eri_primitive(double a, double ax, double ay, double az,
                               double b, double bx, double by, double bz,
                               double c, double cx, double cy, double cz,
                               double d, double dx, double dy, double dz) {
    const double p   = a + b;                    // exponent of pair (a,b)
    const double q   = c + d;                    // exponent of pair (c,d)
    const double ab2 = dist2(ax, ay, az, bx, by, bz);
    const double cd2 = dist2(cx, cy, cz, dx, dy, dz);
    // Combined centers P (from a,b) and Q (from c,d).
    const double px = (a * ax + b * bx) / p, py = (a * ay + b * by) / p, pz = (a * az + b * bz) / p;
    const double qx = (c * cx + d * dx) / q, qy = (c * cy + d * dy) / q, qz = (c * cz + d * dz) / q;
    const double pq2 = dist2(px, py, pz, qx, qy, qz);   // |P - Q|^2
    const double alpha = p * q / (p + q);               // Boys argument prefactor
    const double Kab = std::exp(-a * b / p * ab2);       // pre-exponential of pair ab
    const double Kcd = std::exp(-c * d / q * cd2);       // pre-exponential of pair cd
    const double pref = 2.0 * std::pow(QC_PI, 2.5) / (p * q * std::sqrt(p + q));
    return pref * Kab * Kcd * boys_f0(alpha * pq2);
}
