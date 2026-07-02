// ===========================================================================
// src/coronary.h  --  The ONE shared "per-vessel physics" core (CPU == GPU)
// ---------------------------------------------------------------------------
// Project 6.20 : Coronary Autoregulation & Microvascular Perfusion
//
// WHY THIS FILE EXISTS (the single most important idea in the project)
//   The CPU reference (reference_cpu.cpp, compiled by cl.exe/g++) and the GPU
//   kernels (kernels.cu, compiled by nvcc) must compute BYTE-FOR-BYTE identical
//   arithmetic so that "GPU == CPU within tolerance" is a meaningful test and
//   not a fudge. The way we guarantee that (PATTERNS.md §2, the "HD-macro
//   idiom") is to put every per-vessel formula in ONE header, marked
//   __host__ __device__, and #include it from BOTH sides. Neither side gets to
//   have its own private copy of the physics that could silently drift.
//
//   Keep this header free of CUDA-only constructs (no __global__, no <<<>>>,
//   no thrust). It must compile under a plain C++ host compiler too.
//
// WHAT'S MODELED HERE (the science, in one breath)
//   A coronary microvascular network is a tree/graph of cylindrical vessel
//   SEGMENTS meeting at NODES (junctions). Blood is (to first order) an
//   incompressible viscous fluid in slow laminar flow, so each segment obeys
//   POISEUILLE'S LAW:  Q = G * (p_a - p_b),  where the conductance
//       G = pi * r^4 / (8 * mu * L)
//   depends on radius r, length L, and viscosity mu. Conservation of flow at
//   every interior node (sum of Q in = sum of Q out) turns the whole network
//   into a SPARSE, SYMMETRIC, POSITIVE-DEFINITE linear system  L p = b  for the
//   nodal pressures p (a "resistor network" / graph-Laplacian problem).
//
//   AUTOREGULATION: coronary arterioles actively change their radius to keep
//   perfusion roughly constant as demand/pressure vary (metabolic + myogenic
//   feedback). We model this as a slow radius update that pushes each segment's
//   flow toward a target flow. Re-solving after the update shows perfusion
//   being regulated -- the headline phenomenon of this project.
//
// READ THIS AFTER: main.cu (the 5-step flow). READ BEFORE: kernels.cu,
//   reference_cpu.cpp (both call the inline functions below).
// ===========================================================================
#pragma once

// ---------------------------------------------------------------------------
// The HD (host+device) decorator switch. Under nvcc, __CUDACC__ is defined and
// we tag each function __host__ __device__ so it compiles for BOTH the CPU and
// the GPU. Under a plain host compiler the tokens don't exist, so we #define
// them away. This is the crux of the CPU/GPU-parity idiom (PATTERNS.md §2).
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define CORONARY_HD __host__ __device__
#else
#define CORONARY_HD
#endif

#include <cmath>     // std::pow, std::sqrt (host); nvcc maps these to device math

// ---------------------------------------------------------------------------
// Physical + numerical constants. Units are documented next to every value so
// the arithmetic is auditable. We work in a consistent CGS-ish micro-scale set:
//     length  : micrometres (um)
//     pressure: mmHg
//     viscosity: mmHg * s        (so that G has units um^3 / (mmHg*s) and
//     flow    : um^3 / s          Q = G * dP comes out in um^3/s)
// The absolute unit system does not affect the LINEAR-ALGEBRA correctness we
// verify; it only sets the scale of the printed numbers. We pick values that
// keep pressures in the physiological 0..100 mmHg range so output is legible.
// ---------------------------------------------------------------------------
namespace coronary {

// pi to double precision. We spell it out rather than rely on M_PI, which is
// not guaranteed by the C++ standard headers on every compiler (MSVC hides it
// behind _USE_MATH_DEFINES).
constexpr double PI = 3.14159265358979323846;

// Plasma viscosity baseline (mmHg*s). Real plasma is ~1.2 mPa*s; here it is a
// scaled teaching constant. The Fahraeus-Lindqvist correction below multiplies
// this by a hematocrit/radius-dependent factor.
constexpr double MU_PLASMA = 3.0e-9;   // mmHg*s (scaled)

// ---------------------------------------------------------------------------
// fahraeus_lindqvist_factor(radius_um, hct)
//   The Fahraeus-Lindqvist effect: in vessels narrower than ~300 um, red cells
//   migrate to the axial core leaving a cell-free plasma layer at the wall, so
//   the EFFECTIVE viscosity DROPS as the vessel narrows (down to ~10 um), then
//   rises again in capillaries. We use a smooth, monotone-in-the-teaching-range
//   approximation: viscosity relative to plasma grows with radius toward an
//   asymptote set by hematocrit. This is a simplified stand-in for the classic
//   in-vitro fit (Pries et al.); the point for the learner is that viscosity is
//   NOT constant across scales, which changes each segment's conductance.
//
//   Parameters:
//     radius_um : vessel radius in micrometres (> 0)
//     hct       : hematocrit as a fraction in [0,1] (e.g. 0.45)
//   Returns: dimensionless multiplier applied to MU_PLASMA (>= 1).
//   Complexity: O(1), a handful of flops. Called once per segment per solve.
// ---------------------------------------------------------------------------
CORONARY_HD inline double fahraeus_lindqvist_factor(double radius_um, double hct) {
    // A bounded, smooth ramp: at large r the relative apparent viscosity
    // approaches (1 + k*hct); at small r it approaches 1 (nearly plasma). The
    // 0.5*diameter half-saturation constant (~ 25 um radius) puts the transition
    // in the arteriolar range where autoregulation actually happens.
    const double d = 2.0 * radius_um;                 // diameter (um)
    const double hi = 1.0 + 2.5 * hct;                // large-vessel asymptote
    const double sat = d / (d + 50.0);                // 0 at d->0, ->1 at d>>50um
    return 1.0 + (hi - 1.0) * sat;                    // in [1, hi)
}

// ---------------------------------------------------------------------------
// segment_conductance(radius_um, length_um, hct)
//   Poiseuille conductance G = pi r^4 / (8 mu_eff L), where mu_eff includes the
//   Fahraeus-Lindqvist correction. G is the "1/resistance" of the segment: the
//   flow through it is Q = G * (p_a - p_b). The r^4 dependence is the reason a
//   small radius change (autoregulation!) has an OUTSIZED effect on flow -- the
//   single most important intuition in this whole model.
//
//   Returns G in um^3/(mmHg*s). Complexity O(1).
// ---------------------------------------------------------------------------
CORONARY_HD inline double segment_conductance(double radius_um, double length_um, double hct) {
    const double mu_eff = MU_PLASMA * fahraeus_lindqvist_factor(radius_um, hct);
    const double r2 = radius_um * radius_um;
    const double r4 = r2 * r2;                          // r^4 (the key nonlinearity)
    return (PI * r4) / (8.0 * mu_eff * length_um);
}

// ---------------------------------------------------------------------------
// autoregulate_radius(radius_um, flow, target_flow, gain, rmin, rmax)
//   One step of the autoregulatory radius update. Coronary arterioles dilate
//   when perfusion is below the metabolic demand (too little flow -> vasodilate
//   -> larger r -> much larger G because of r^4) and constrict when perfusion
//   is above demand. We model this as proportional feedback on the RELATIVE
//   flow error, clamped to a physiological radius band [rmin, rmax]:
//
//       err   = (target_flow - |Q|) / target_flow      (dimensionless)
//       r_new = clamp( r * (1 + gain * err), rmin, rmax )
//
//   This is a deliberately simple, deterministic surrogate for the coupled
//   metabolic (adenosine/O2) + myogenic ODEs named in the catalog; THEORY.md
//   §"real world" explains the fuller model. It is monotone and bounded, so the
//   outer autoregulation loop converges.
//
//   Parameters:
//     radius_um   : current radius (um)
//     flow        : current signed flow through the segment (um^3/s)
//     target_flow : desired flow magnitude (um^3/s), the metabolic set-point
//     gain        : feedback gain (dimensionless, small, e.g. 0.2)
//     rmin, rmax  : radius clamp band (um)
//   Returns: the updated radius (um). O(1).
// ---------------------------------------------------------------------------
CORONARY_HD inline double autoregulate_radius(double radius_um, double flow,
                                              double target_flow, double gain,
                                              double rmin, double rmax) {
    const double q = flow < 0.0 ? -flow : flow;         // |Q| (branch, not fabs, so
                                                        //   host & device agree exactly)
    const double err = (target_flow - q) / target_flow; // >0 => under-perfused => dilate
    double r_new = radius_um * (1.0 + gain * err);
    if (r_new < rmin) r_new = rmin;                     // clamp low (myogenic floor)
    if (r_new > rmax) r_new = rmax;                     // clamp high (max dilation)
    return r_new;
}

}  // namespace coronary
