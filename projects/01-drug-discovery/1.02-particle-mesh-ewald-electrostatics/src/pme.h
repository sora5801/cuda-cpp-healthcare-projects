// ===========================================================================
// src/pme.h  --  Shared (host + device) Particle-Mesh Ewald primitives
// ---------------------------------------------------------------------------
// Project 1.2 : Particle-Mesh Ewald Electrostatics  (see ../THEORY.md)
//
// WHAT THIS PROJECT COMPUTES
//   The electrostatic (Coulomb) energy of N point charges in a PERIODIC box --
//   the dominant long-range interaction in any molecular-dynamics simulation of
//   a solvated protein. A naive periodic Coulomb sum does not even converge
//   absolutely; Ewald summation fixes that by splitting 1/r into two pieces:
//
//       E_total = E_real  +  E_recip  -  E_self
//
//     * E_real  : a SHORT-range part, sum of q_i q_j * erfc(beta r)/r over pairs
//                 within a cutoff. Decays fast -> evaluate directly (here on the
//                 host, with the minimum-image convention).
//     * E_recip : a smooth LONG-range part, a sum over reciprocal-space vectors.
//                 Smooth PME (SPME) evaluates it on a 3D charge GRID via an FFT
//                 in O(N log N). THIS is the GPU-accelerated heart of the method
//                 and what kernels.cu computes.
//     * E_self  : a constant correction removing each charge's interaction with
//                 its own smeared Gaussian (added in the real+recip split).
//
//   The Ewald SPLITTING PARAMETER beta (units 1/length) trades work between the
//   real and reciprocal sums; the TOTAL energy is INVARIANT to beta. We exploit
//   that as a physics check (main.cu, THEORY "How we verify").
//
// WHY A GPU  (the catalog "Deep dive")
//   E_recip dominates wall-time for large biological systems. Two steps are
//   data-parallel over atoms -- CHARGE SPREADING (particle -> mesh) and force/
//   energy interpolation (mesh -> particle) -- while the 3D FFT is handled by
//   cuFFT. PME scales O(N log N). This project teaches the spreading kernel
//   (an atomic SCATTER made deterministic with fixed-point, like project 11.09)
//   and USING cuFFT WITHOUT IT BEING A BLACK BOX (like project 8.03).
//
// WHY THIS HEADER EXISTS  (the shared __host__ __device__ core, PATTERNS.md §2)
//   The per-atom physics -- the cardinal B-spline interpolation weights and the
//   fixed-point quantization -- is written ONCE here as PME_HD inline functions
//   so the CPU reference (reference_cpu.cpp) and the GPU kernel (kernels.cu) run
//   BYTE-FOR-BYTE IDENTICAL math. That is what lets us verify GPU==CPU to a tiny
//   tolerance instead of hand-waving "close enough".
//
//   Keep CUDA-only constructs (no __global__, no cufft types) OUT of this header
//   so the plain host compiler can include it too.
//
// READ THIS BEFORE: reference_cpu.h, kernels.cuh.
// ===========================================================================
#pragma once

#include <cstdint>
#include <cmath>

// PME_HD expands to the CUDA decorators under nvcc, and to nothing under the
// host compiler -- so the SAME function body compiles for CPU and GPU.
#ifdef __CUDACC__
#define PME_HD __host__ __device__
#else
#define PME_HD
#endif

// ---------------------------------------------------------------------------
// Physical constants and method parameters (one place, used by both paths).
// ---------------------------------------------------------------------------
// We work in a self-consistent "reduced" unit system where Coulomb's constant
// 1/(4*pi*eps0) == 1. Charges are in elementary units (e), lengths in the same
// arbitrary unit as the box, and energies come out in those reduced units. This
// keeps the teaching code free of unit-conversion clutter; THEORY "real world"
// notes the kcal/mol factor a production MD code would multiply in.
//
// B-spline interpolation ORDER. Cardinal B-splines of order p spread each charge
// onto a p x p x p block of grid points (p contiguous points per axis). Order 4
// (cubic) is the classic SPME choice: smooth enough for good accuracy, cheap
// enough to teach. THEORY explains why higher order (6) buys more accuracy.
static const int PME_ORDER = 4;

// ---------------------------------------------------------------------------
// FIXED-POINT scatter accumulation (determinism trick, same idea as 11.09/5.01)
// ---------------------------------------------------------------------------
// The spreading step has MANY atoms adding fractional charge to the SAME grid
// point -> a scatter-reduction via atomicAdd. Float atomicAdd is NOT associative
// (the sum depends on the nondeterministic order threads arrive), so a float
// grid would be irreproducible AND would not match the CPU. We instead add
// charge as fixed-point INTEGERS (atomicAdd on unsigned long long), which commute
// -> the GPU grid is bit-identical every run and equals the CPU grid exactly.
//
// PME_QSCALE is the number of fixed-point units per 1.0 of charge. 2^40 ~ 1.1e12
// gives ~12 significant digits of charge resolution. Worst case a single grid
// point receives ~ N * |q_max| of charge; with N ~ thousands and |q| ~ O(1),
// the accumulated integer stays far below 2^63, so unsigned long long never
// overflows. THEORY "Numerical considerations" works the bound.
static const long long PME_QSCALE = 1ll << 40;

// Quantize a (possibly negative) charge contribution to signed fixed-point.
// We accumulate into unsigned long long and interpret the bit pattern as a
// two's-complement signed integer on readout (pme_fixed_to_double): unsigned
// atomicAdd with wraparound implements signed modular addition exactly, so
// negative charges (which are ubiquitous) are handled with no extra machinery.
PME_HD inline unsigned long long pme_to_fixed(double contribution) {
    // llround -> nearest integer, deterministic and symmetric about zero.
    const long long q = static_cast<long long>(std::llround(contribution * static_cast<double>(PME_QSCALE)));
    return static_cast<unsigned long long>(q);   // reinterpret signed bits as unsigned
}

// Convert an accumulated fixed-point grid cell back to a real charge density.
PME_HD inline double pme_fixed_to_double(unsigned long long acc) {
    // Reinterpret the unsigned accumulator as signed two's complement, then
    // divide out the scale. This recovers the exact integer sum the atomics built.
    const long long s = static_cast<long long>(acc);
    return static_cast<double>(s) / static_cast<double>(PME_QSCALE);
}

// ---------------------------------------------------------------------------
// Cardinal B-spline weights M_p(u) for SPME charge interpolation.
// ---------------------------------------------------------------------------
// Given a charge whose scaled grid coordinate along one axis is `g` (a real
// number in [0, K) where K is the grid size on that axis), we place it on the
// p nearest grid points. Let g0 = floor(g) be the lower grid index; then the
// p weights apply to grid points g0, g0+1, ..., g0+(p-1) (wrapped periodically).
// `frac` = g - g0 in [0,1) is the offset into the leftmost interval, and w[i]
// is the weight of grid point (g0 + i).
//
// We compute the order-p cardinal B-spline values by the standard recursion,
// starting from the order-2 (linear) hat and lifting to order p. This is the
// textbook Essmann et al. (1995) "fill_bspline" construction (the same one in
// AMBER/GROMACS). Validated against the known cubic node values M4 = {1/6, 2/3,
// 1/6} -- see the unit check in THEORY "How we verify".
//
// Writing it once here (PME_HD) guarantees the CPU reference and the GPU
// spreading kernel use the SAME weights -> identical grids -> exact verification.
PME_HD inline void pme_bspline_weights(double frac, double* w /*[PME_ORDER]*/) {
    const int p = PME_ORDER;

    // Order 2 (linear interpolation) is the base case of the recursion.
    //   w[0] = 1-frac is the weight of the LEFT node g0, w[1] = frac the right.
    w[p - 1] = 0.0;
    w[1]     = frac;
    w[0]     = 1.0 - frac;
    for (int j = 2; j < p - 1; ++j) w[j] = 0.0;

    // Lift from order (k-1) to order k using the Cox-de Boor / Essmann recursion.
    // After the k-loop, w[0..k-1] hold the order-k cardinal B-spline values at the
    // current fractional offset. `div = 1/(k-1)` is the recursion's normalizer.
    for (int k = 3; k <= p; ++k) {
        const double div = 1.0 / (k - 1);
        // Highest index term first (depends only on the old top weight w[k-2]).
        w[k - 1] = div * frac * w[k - 2];
        // Middle terms blend two neighbouring old weights.
        for (int j = 1; j < k - 1; ++j) {
            w[k - 1 - j] = div * ((frac + j) * w[k - 2 - j]
                                  + (k - j - frac) * w[k - 1 - j]);
        }
        // Lowest index term.
        w[0] = div * (1.0 - frac) * w[0];
    }
}

// ---------------------------------------------------------------------------
// Euler exponential-spline B-factor |b(m)|^2 for one axis (the SPME prefactor).
// ---------------------------------------------------------------------------
// SPME approximates the structure factor by interpolating with B-splines, which
// introduces a per-reciprocal-vector correction factor B(m) = |b1|^2 |b2|^2 |b3|^2.
// For order p and grid size K, the 1-D factor at integer wavevector index m is
//       b(m) = exp(2*pi*i*(p-1)*m/K) / sum_{j=0}^{p-2} M_p(j+1) * exp(2*pi*i*m*j/K)
// and we need |b(m)|^2. We pass in `Mp`, the integer B-spline node values as
// produced by pme_bspline_weights(0.0, Mp): Mp[k] == M_p(k+1) for k=0..p-2 (and
// Mp[p-1] == 0). Returns |b(m)|^2.
//
// This lives here so the host (building the influence array) and any device code
// that might rebuild it use identical math. In this project the host builds the
// influence function once and uploads it; the function is HD for symmetry/reuse.
PME_HD inline double pme_bsp_modulus2(int m, int K, const double* Mp /*[PME_ORDER]*/) {
    const int p = PME_ORDER;
    const double two_pi = 6.283185307179586476925286766559;
    double sr = 0.0, si = 0.0;   // real/imag parts of the denominator sum
    for (int j = 0; j <= p - 2; ++j) {
        const double ang = two_pi * static_cast<double>(m) * static_cast<double>(j) / static_cast<double>(K);
        sr += Mp[j] * std::cos(ang);   // Mp[j] = M_p(j+1), the integer B-spline value
        si += Mp[j] * std::sin(ang);
    }
    const double denom2 = sr * sr + si * si;     // |sum|^2 ; numerator |exp(...)|^2 == 1
    // Guard the rare exact zero (denom2 -> 0 only for degenerate small grids).
    return denom2 > 0.0 ? 1.0 / denom2 : 0.0;
}
