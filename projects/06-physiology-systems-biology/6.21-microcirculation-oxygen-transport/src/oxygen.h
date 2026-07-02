// ===========================================================================
// src/oxygen.h  --  Shared (host + device) oxygen-transport physics
// ---------------------------------------------------------------------------
// Project 6.21 : Microcirculation & Oxygen Transport
//
// WHAT THIS PROJECT COMPUTES
//   The steady-state partial pressure of oxygen (PO2) in a small block of tissue
//   fed by a handful of capillaries. This is the classic problem of the
//   MICROCIRCULATION: red blood cells carry O2 down capillaries; O2 diffuses out
//   through the vessel wall into the surrounding tissue; tissue cells consume it.
//   The question a physiologist asks is "is every corner of the tissue getting
//   enough oxygen, or are there hypoxic pockets far from any vessel?"
//
//   We solve it with the GREEN'S FUNCTION METHOD of Secomb & Hsu (a standard tool
//   in quantitative microvascular physiology). The idea:
//
//     * Each capillary is chopped into short SEGMENTS. Each segment releases O2
//       into the tissue at some rate q_j  (units: cm^3 O2 / s per unit length,
//       here lumped into a per-segment source strength). It behaves like a
//       point/line SOURCE of oxygen.
//
//     * A single point source of strength q sitting in an infinite medium of
//       O2 diffusivity D creates a PO2 field that falls off like 1/r:
//                     G(r) = 1 / (4*pi*D*alpha*r)                 (the "Green's function")
//       where alpha is the O2 solubility (Henry coefficient) converting between
//       concentration and partial pressure, and r is the distance from the
//       source. This is EXACTLY the same math as an electrostatic point charge
//       (Coulomb 1/r) or a point heat source -- diffusion, electrostatics and
//       steady heat conduction all obey the same Laplace/Poisson equation. That
//       is why the deep-dive lists APBS (an electrostatics solver) as reusable
//       here.
//
//     * By LINEARITY of the diffusion equation, the PO2 at any tissue point i is
//       the SUM of the contributions of every source j, plus a background term
//       for the O2 the tissue itself consumes:
//                 PO2_i = PO2_inflow + sum_j q_j * G(|x_i - x_j|)  -  consumption
//       Computing this sum for every tissue grid point against every source is an
//       O(N_grid * N_src) all-pairs operation -- embarrassingly parallel, and the
//       exact GPU pattern this project teaches (one thread per tissue point).
//
//   The biology enters through three physiological laws, all implemented below as
//   __host__ __device__ inline functions so the CPU reference (reference_cpu.cpp)
//   and the GPU kernel (kernels.cu) evaluate BYTE-FOR-BYTE identical math -> their
//   results agree to round-off (see ../THEORY.md "How we verify correctness"):
//
//     1. GREEN'S FUNCTION with a finite-radius core (green_function): the raw 1/r
//        blows up at r=0 (right on top of a source). Physically a capillary has a
//        radius R_cap, so we cap the kernel at r = R_cap. This is the standard
//        regularization and keeps the sum finite and well-defined.
//
//     2. HILL HEMOGLOBIN SATURATION (hill_saturation): how much O2 the blood
//        actually carries at a given PO2 -- the sigmoid oxyhemoglobin dissociation
//        curve S = P^n / (P50^n + P^n). Used to set each segment's source
//        strength from the local blood PO2 (more saturated blood -> more O2 to
//        give up).
//
//     3. MICHAELIS-MENTEN CONSUMPTION (mm_consumption): tissue cells consume O2 at
//        a rate that saturates at high PO2 and falls off as PO2 -> 0:
//        M(P) = M0 * P / (P + Km). This is the demand side of the balance.
//
//   OXY_HD expands to "__host__ __device__" under nvcc and to nothing under the
//   plain host compiler (the HD-macro idiom, PATTERNS.md section 2). Keep this
//   header free of __global__ / CUDA-only types so reference_cpu.cpp can include
//   it too.
//
// READ THIS AFTER: nothing -- start here, it defines the model. Then read
//   reference_cpu.h (grid + source containers), kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <cmath>   // std::sqrt, std::pow  (host); nvcc maps these to device intrinsics

// The HD-macro idiom: one definition, two compilers.
#ifdef __CUDACC__
#define OXY_HD __host__ __device__
#else
#define OXY_HD
#endif

// ---------------------------------------------------------------------------
// Physical / model constants (SI-ish, kept simple and documented). These are
// order-of-magnitude physiological values for teaching, NOT tuned to any tissue.
// Lengths are in micrometres (um); PO2 in mmHg; time implicitly folded into the
// lumped source strengths so the arithmetic stays in convenient units.
// ---------------------------------------------------------------------------

// Oxygen diffusivity * solubility in tissue, lumped into one "conductivity"
// constant K = D*alpha that appears in the Green's function denominator. Using a
// single lumped constant keeps the teaching model transparent; the real Secomb
// code carries D and alpha separately. Units chosen so PO2 comes out in mmHg for
// the source strengths used in the sample (see data/README.md).
#define OXY_K_DIFF 1.0

// Capillary radius (um): the Green's function core size. Inside this radius we
// clamp r so the 1/r kernel does not diverge (a real capillary is not a point).
#define OXY_R_CAP 3.0

// ---------------------------------------------------------------------------
// green_function: the steady-state diffusion Green's function G(r) = 1/(4*pi*K*r),
//   REGULARIZED so it stays finite at the vessel core.
//
//   Parameters:
//     r         -- distance (um) from the source to the field point (>= 0).
//   Returns:
//     G(r)      -- PO2 (mmHg) produced per unit source strength at distance r.
//
//   Why the clamp: exactly at a source r=0 and 1/r = infinity, which is
//   unphysical. A capillary has radius R_CAP, so for r < R_CAP we evaluate the
//   kernel at R_CAP instead (the field is ~flat inside the vessel core). This is
//   the standard finite-core regularization used in Green's-function O2 solvers.
//   Complexity: O(1). Called N_grid * N_src times -> it must be cheap.
// ---------------------------------------------------------------------------
OXY_HD inline double green_function(double r) {
    const double pi = 3.14159265358979323846;
    // Clamp the distance to the capillary radius so we never divide by ~0.
    const double r_eff = (r < OXY_R_CAP) ? OXY_R_CAP : r;
    return 1.0 / (4.0 * pi * OXY_K_DIFF * r_eff);
}

// ---------------------------------------------------------------------------
// dist3: Euclidean distance between two 3-D points (um). One sqrt; O(1).
//   Pulled out so the CPU and GPU compute the geometry identically.
// ---------------------------------------------------------------------------
OXY_HD inline double dist3(double ax, double ay, double az,
                           double bx, double by, double bz) {
    const double dx = ax - bx;
    const double dy = ay - by;
    const double dz = az - bz;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// ---------------------------------------------------------------------------
// hill_saturation: fraction of hemoglobin bound with O2 at partial pressure P.
//   S(P) = P^n / (P50^n + P^n)   -- the sigmoidal oxyhemoglobin dissociation
//   curve (Hill equation). P50 is the PO2 at which hemoglobin is half-saturated
//   (~26 mmHg in human blood); n is the Hill cooperativity coefficient (~2.7).
//
//   Parameters:
//     p    -- blood PO2 (mmHg), >= 0.
//     p50  -- half-saturation PO2 (mmHg).
//     n    -- Hill coefficient (dimensionless).
//   Returns: saturation in [0,1].
//
//   Used to convert a capillary segment's blood PO2 into how much O2 it can
//   deliver -- higher saturation blood is a stronger tissue O2 source. O(1) but
//   uses pow(), the most expensive op here; called once per source, not per pair.
// ---------------------------------------------------------------------------
OXY_HD inline double hill_saturation(double p, double p50, double n) {
    if (p <= 0.0) return 0.0;                 // no negative PO2; guard pow(0,n)
    const double pn   = std::pow(p, n);
    const double p50n = std::pow(p50, n);
    return pn / (p50n + pn);
}

// ---------------------------------------------------------------------------
// mm_consumption: Michaelis-Menten tissue O2 consumption rate at partial
//   pressure P.  M(P) = M0 * P / (P + Km).
//     * At high PO2 (P >> Km) it saturates at M0 (demand-limited).
//     * As PO2 -> 0 it falls linearly (supply-limited) -> avoids the unphysical
//       "cells keep consuming O2 that isn't there".
//
//   Parameters:
//     p   -- local tissue PO2 (mmHg), >= 0.
//     m0  -- maximal consumption rate (mmHg-equivalent sink strength).
//     km  -- Michaelis constant (mmHg): PO2 at half-maximal consumption.
//   Returns: consumption rate (same units as m0).
//
//   In this teaching model the consumption is applied as a fixed background sink
//   evaluated at the inflow PO2 (a first-order, non-iterative approximation) so
//   the whole solve stays a single linear superposition. THEORY.md section
//   "Where this sits in the real world" explains the fully-coupled nonlinear
//   version (Secomb solves for the q_j self-consistently).
// ---------------------------------------------------------------------------
OXY_HD inline double mm_consumption(double p, double m0, double km) {
    if (p <= 0.0) return 0.0;
    return m0 * p / (p + km);
}

// ---------------------------------------------------------------------------
// clamp_po2: PO2 cannot be negative (no such thing as negative oxygen). After
//   summing sources and subtracting consumption we clamp to >= 0. Doing this in
//   ONE shared function means CPU and GPU clamp identically -> exact agreement.
// ---------------------------------------------------------------------------
OXY_HD inline double clamp_po2(double p) {
    return (p < 0.0) ? 0.0 : p;
}
