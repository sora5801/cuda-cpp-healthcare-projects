// ===========================================================================
// src/saxs_core.h  --  The ONE TRUE per-element SAXS physics (CPU == GPU)
// ---------------------------------------------------------------------------
// Project 2.24 : SAXS / SANS Data-Driven Structure Modeling
//
// WHY THIS HEADER EXISTS  (PATTERNS.md §2: the shared __host__ __device__ core)
//   The single most useful idiom in this repo. The per-pair scattering physics
//   below is written ONCE as `__host__ __device__` inline functions, then
//   #included by BOTH:
//     * reference_cpu.cpp (compiled by the host C++ compiler), and
//     * kernels.cu        (compiled by nvcc for the GPU).
//   Because both paths run the *identical* arithmetic, the CPU reference and the
//   GPU kernel produce byte-for-byte-comparable intensities -> verification is
//   exact-to-rounding instead of "approximately right". This file therefore
//   contains NO CUDA-only types (no __global__, no <<<>>>) so the plain C++
//   compiler can include it happily.
//
// THE SCIENCE IN ONE PARAGRAPH
//   A protein in solution tumbles randomly, so a small-angle scattering (SAXS)
//   experiment measures the *orientationally averaged* X-ray intensity as a
//   function of the momentum transfer q = 4*pi*sin(theta)/lambda (units 1/Å).
//   For a set of point scatterers (atoms) at positions r_i with scattering
//   strengths f_i, that average has the famous closed form -- the DEBYE FORMULA:
//
//        I(q) = sum_i sum_j  f_i * f_j * sinc(q * r_ij),   r_ij = |r_i - r_j|
//
//   where sinc(x) = sin(x)/x and sinc(0)=1. The double sum over all atom PAIRS
//   is O(N^2): that is the bottleneck this project parallelizes on the GPU.
//   (We use a point-atom approximation -- one constant electron count per atom
//   instead of a q-dependent atomic form factor and an explicit hydration shell.
//   THEORY.md §"Where this sits in the real world" explains what CRYSOL/FOXS add.)
//
// READ THIS FIRST in the code tour, then reference_cpu.h -> kernels.cuh.
// ===========================================================================
#pragma once

#include <cmath>   // std::sin, std::sqrt (host side); nvcc maps these to device intrinsics

// ---------------------------------------------------------------------------
// HD: the host/device decorator macro (PATTERNS.md §2 idiom).
//   * Under nvcc (__CUDACC__ defined) every function below is compiled for BOTH
//     the host and the device, so the SAME symbol is callable from a kernel and
//     from plain C++.
//   * Under a plain C++ compiler the decorators do not exist, so HD expands to
//     nothing and the functions are ordinary inline host functions.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ---------------------------------------------------------------------------
// debye_sinc: the spherically-averaged scattering kernel sinc(x) = sin(x)/x.
//   This is the heart of the Debye formula -- it is what you get when you
//   average exp(i*q·r_ij) over all orientations of the vector r_ij on the unit
//   sphere. We compute everything in DOUBLE precision: the running intensity sum
//   spans many orders of magnitude (the i==j self terms dominate I(0)), so float
//   would lose the small high-q structure we actually want to see.
//
//   x : the dimensionless product q * r_ij  (q in 1/Å, r_ij in Å -> x unitless)
//   returns sinc(x), with the removable singularity at x=0 handled explicitly.
// ---------------------------------------------------------------------------
HD inline double debye_sinc(double x) {
    // Near x=0, sin(x)/x is 0/0 numerically. The analytic limit is 1, and the
    // Taylor series sinc(x) ≈ 1 - x^2/6 is already accurate to ~1e-12 for the
    // tiny |x| we hit. Branching on a small threshold keeps BOTH the CPU and GPU
    // paths identical (same constant, same compare) so results stay in lockstep.
    if (x > -1.0e-8 && x < 1.0e-8) {
        return 1.0 - x * x / 6.0;   // leading Taylor term; avoids 0/0
    }
    return std::sin(x) / x;
}

// ---------------------------------------------------------------------------
// debye_intensity_at_q: I(q) for ONE momentum-transfer value q, summed over all
//   atom pairs. This is the per-output-element work item -- on the GPU each
//   thread owns one q and runs exactly this loop; the CPU reference runs it in a
//   serial loop over all q. Both call THIS function, guaranteeing parity.
//
//   q     : the momentum transfer (1/Å) this intensity is evaluated at
//   x,y,z : flat arrays of the n atom coordinates (Å), one component per array
//   f     : per-atom scattering strength (electron count, arbitrary units)
//   n     : number of atoms
//   returns I(q) in (electron-count)^2 units. I(0) = (sum_i f_i)^2.
//
//   Complexity: O(n^2) distance/sinc evaluations. We exploit the symmetry
//   I(q) = sum_i f_i^2 + 2 * sum_{i<j} f_i f_j sinc(q r_ij): the diagonal (self)
//   terms are sinc(0)=1, and each off-diagonal pair is counted once and doubled.
//   That halves the work versus the naive full double loop -- a standard and
//   worthwhile teaching optimization (THEORY.md §Algorithm).
// ---------------------------------------------------------------------------
HD inline double debye_intensity_at_q(double q,
                                      const double* x, const double* y, const double* z,
                                      const double* f, int n) {
    // 1) Diagonal (i == j) contribution: sinc(0)=1, so each atom contributes
    //    f_i^2. Accumulated separately for clarity and numerical bookkeeping.
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        acc += f[i] * f[i];     // self term: f_i * f_i * sinc(0)
    }

    // 2) Off-diagonal pairs i<j, each counted ONCE then doubled (symmetry above).
    for (int i = 0; i < n; ++i) {
        const double xi = x[i], yi = y[i], zi = z[i], fi = f[i];
        for (int j = i + 1; j < n; ++j) {
            // Euclidean atom-atom distance r_ij (Å).
            const double dx = xi - x[j];
            const double dy = yi - y[j];
            const double dz = zi - z[j];
            const double rij = std::sqrt(dx * dx + dy * dy + dz * dz);
            // Pair contribution f_i f_j sinc(q r_ij); the factor 2 folds in both
            // (i,j) and (j,i) orderings of the original double sum.
            acc += 2.0 * fi * f[j] * debye_sinc(q * rij);
        }
    }
    return acc;
}

#undef HD   // keep the macro local to this header (it is re-defined per includer)
