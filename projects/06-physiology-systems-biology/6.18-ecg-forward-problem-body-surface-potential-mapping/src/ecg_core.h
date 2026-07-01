// ===========================================================================
// src/ecg_core.h  --  The ONE TRUE per-element ECG physics, shared CPU + GPU
// ---------------------------------------------------------------------------
// Project 6.18 : ECG Forward Problem & Body-Surface Potential Mapping
//   (see ../THEORY.md and the catalog deep-dive for the full "why")
//
// WHY THIS HEADER EXISTS  (PATTERNS.md §2, the "__host__ __device__ core" idiom)
//   The ECG forward problem has ONE scalar formula that everything else is built
//   from: the electric potential produced at a body-surface electrode by a single
//   current DIPOLE inside the torso. If we write that formula twice -- once in the
//   CPU reference (reference_cpu.cpp, compiled by cl.exe) and once in the GPU
//   kernel (kernels.cu, compiled by nvcc) -- the two will inevitably drift, and
//   our "GPU == CPU" verification becomes fuzzy. So we write it EXACTLY ONCE,
//   here, as an inline function tagged `__host__ __device__`, and call it from
//   both sides. Neither side gets to "improve" the formula on its own; the
//   verification in main.cu can then be near machine-exact.
//
//   Keep this header free of CUDA-only constructs (no __global__, no <<<>>>).
//   The ECG_HD macro evaporates to nothing under cl.exe and becomes
//   `__host__ __device__` under nvcc -- that is the whole trick.
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cu  (both include this file).
// ===========================================================================
#pragma once

#include <cmath>     // std::sqrt
#include <cstddef>   // std::size_t

// ---------------------------------------------------------------------------
// ECG_HD : the host/device portability shim.
//   * Under nvcc (__CUDACC__ defined) a function tagged ECG_HD is compiled for
//     BOTH the CPU (host) and the GPU (device), so the same code object runs in
//     main.cu's host code AND inside a __global__ kernel.
//   * Under a plain host compiler the decorators do not exist, so ECG_HD must
//     expand to nothing -- the function is then just an ordinary inline.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define ECG_HD __host__ __device__
#else
#define ECG_HD
#endif

namespace ecg {

// ---------------------------------------------------------------------------
// A point in 3-D space (units: metres). Used for electrode positions on the
// torso surface AND for the fixed anchor point of each cardiac dipole source.
// Plain-old-data so it can live in a device array and be memcpy'd with no fuss.
// ---------------------------------------------------------------------------
struct Vec3 {
    double x;   // metres
    double y;   // metres
    double z;   // metres
};

// The vacuum/tissue permittivity is irrelevant here: the ECG forward problem is
// QUASI-STATIC (frequencies are ~1 Hz-1 kHz, wavelengths are hundreds of km, so
// magnetic induction and displacement currents are negligible). Only the ohmic
// conductivity sigma matters. We use a single homogeneous torso conductivity.
//   Typical mean thoracic conductivity ~ 0.2 S/m (siemens per metre). The exact
//   value only rescales ALL potentials by a constant, so it never changes which
//   electrode is largest; we keep it explicit for physical honesty.
static constexpr double TORSO_SIGMA = 0.2;         // S/m (homogeneous model)
static constexpr double FOUR_PI     = 12.566370614359172; // 4*pi, folded once

// ---------------------------------------------------------------------------
// dipole_potential: the ONE TRUE forward formula (an entry of the lead field).
//
//   PHYSICS.  In an infinite homogeneous ohmic medium of conductivity sigma, a
//   current dipole of moment p (units: A*m) located at r0 produces, at a field
//   point r, the electric potential
//
//                     1        p . (r - r0)
//        phi(r)  =  ------- * ---------------- .
//                   4 pi s        |r - r0|^3
//
//   This is the quasi-static Green's function of the Poisson equation
//   div(sigma grad phi) = -div(J_source) for a point-dipole source. It is the
//   physically exact forward map for an UNBOUNDED homogeneous conductor and is
//   the didactic heart of every ECG forward model (a real torso model replaces
//   this analytic kernel with a boundary/finite-element solve -- see THEORY.md
//   §"real world"). We factor the dipole moment as (strength s) * (unit
//   direction d), so this routine returns the potential PER UNIT SOURCE STRENGTH
//   -- i.e. exactly one entry A[electrode][source] of the lead-field matrix.
//
//   PARAMETERS
//     r     : electrode position on the torso surface        (Vec3, metres)
//     r0    : the dipole's fixed anchor position in the heart (Vec3, metres)
//     d     : the dipole's unit direction (should be normalized by the caller;
//             a stable activation direction for that source)  (Vec3, unitless)
//   RETURNS
//     potential at r per unit dipole strength                 (volts per A*m,
//     times 1/sigma) -- one lead-field entry. Multiplying by the time-varying
//     source strength s(t) gives that source's contribution to the electrode.
//
//   NUMERICS.  The 1/dist^3 falls off fast, so distant sources contribute little
//   and near sources dominate -- exactly the spatial sensitivity ("lead field")
//   an electrode has. We add a tiny softening EPS to the distance so a source
//   that happens to sit on an electrode cannot divide by zero; with our geometry
//   (heart strictly inside the torso) this branch never actually triggers, but a
//   seatbelt keeps the result finite and reproducible on both CPU and GPU.
// ---------------------------------------------------------------------------
ECG_HD inline double dipole_potential(const Vec3& r, const Vec3& r0, const Vec3& d) {
    double dx = r.x - r0.x;                 // displacement electrode <- source
    double dy = r.y - r0.y;
    double dz = r.z - r0.z;
    double dist2 = dx * dx + dy * dy + dz * dz;   // |r - r0|^2
    double dist  = std::sqrt(dist2);              // |r - r0|
    const double EPS = 1.0e-9;                     // softening (metres); seatbelt
    if (dist < EPS) dist = EPS;
    double dist3 = dist * dist * dist;             // |r - r0|^3
    double dot   = dx * d.x + dy * d.y + dz * d.z; // p_hat . (r - r0)
    // 1/(4 pi sigma) * (d . (r-r0)) / |r-r0|^3  -- potential per unit strength.
    return dot / (FOUR_PI * TORSO_SIGMA * dist3);
}

// ---------------------------------------------------------------------------
// normalize: return a unit-length copy of v (its direction).
//   Dipole DIRECTIONS must be unit vectors so that the "strength" time series
//   carries the magnitude. A zero vector is returned unchanged (guarded) so a
//   degenerate source cannot produce NaNs.
// ---------------------------------------------------------------------------
ECG_HD inline Vec3 normalize(const Vec3& v) {
    double n = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (n < 1.0e-300) return v;                    // guard the zero vector
    double inv = 1.0 / n;
    return Vec3{ v.x * inv, v.y * inv, v.z * inv };
}

}  // namespace ecg
