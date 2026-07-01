// ===========================================================================
// src/pct_physics.h  --  Shared __host__ __device__ proton-CT physics core
// ---------------------------------------------------------------------------
// Project 5.15 : Proton CT & Ion Imaging Reconstruction
//
// WHY THIS FILE EXISTS (read this first)
//   The single most important idiom in this repo (docs/PATTERNS.md section 2):
//   put the per-element PHYSICS in ONE header marked `__host__ __device__` so
//   that the CPU reference (reference_cpu.cpp, compiled by cl.exe) and the GPU
//   kernels (kernels.cu, compiled by nvcc) run BYTE-FOR-BYTE IDENTICAL math.
//   Verification then becomes exact instead of "close enough": every voxel of
//   the GPU reconstruction must equal the CPU reconstruction.
//
//   Everything here is plain arithmetic on POD types (no std::vector, no
//   __global__, no CUDA-only types) so BOTH compilers accept it.
//
// WHAT PROTON CT IS (the science, in one paragraph)
//   X-ray CT measures how much X-rays are attenuated and must then CONVERT
//   Hounsfield units to relative stopping power (RSP) to plan a proton therapy
//   treatment -- a conversion with ~3% range uncertainty. Proton CT skips that:
//   it sends protons THROUGH the patient and measures each proton's residual
//   energy, which gives its water-equivalent path length (WEPL) -- the integral
//   of RSP along the proton's path. Reconstruct RSP from many WEPL measurements
//   and you have exactly the quantity treatment planning needs, no conversion.
//
//   The twist that makes pCT its own algorithm (not just X-ray CT): a proton
//   does NOT travel in a straight line. Multiple Coulomb scattering off nuclei
//   bends it. The best estimate of where the proton actually went, given its
//   measured entry AND exit position+direction, is the MOST-LIKELY PATH (MLP) --
//   a curved trajectory. We must integrate/backproject along that CURVE.
//
// WHAT THIS HEADER PROVIDES
//   * PctGeom        : the reconstruction grid (square, world coords).
//   * Proton         : one measured history (entry/exit pose + WEPL).
//   * tanf_hd()      : host/device single-precision tangent (parity helper).
//   * mlp_point()    : the most-likely-path position at depth fraction t in [0,1].
//                      (Reduces to a straight chord when both angles -> 0.)
//
//   These are consumed by:
//     - reference_cpu.cpp : loops protons on the host (the trusted baseline).
//     - kernels.cu        : one GPU thread per proton, same calls.
//   READ THIS BEFORE: reference_cpu.h, kernels.cuh.
// ===========================================================================
#pragma once

#include <cmath>   // std::sqrt, std::floor, std::tan (host side)

// ---------------------------------------------------------------------------
// HD macro (docs/PATTERNS.md section 2). When compiled by nvcc (__CUDACC__ is
// defined) these functions become callable from BOTH host and device. When
// compiled by the plain host C++ compiler the decorators simply vanish, so the
// same source is ordinary C++.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define PCT_HD __host__ __device__
#else
#define PCT_HD
#endif

// ---------------------------------------------------------------------------
// PctGeom : the reconstruction volume. We do the 2-D teaching case (a single
// slice); the RSP image is `n x n` voxels covering the square world region
// [-half, half] x [-half, half]. Real pCT reconstructs a 3-D volume, but the
// geometry, the MLP, and the SART update are all identical per slice -- 2-D
// keeps the arithmetic legible (THEORY.md "Where this sits in the real world").
//   world units are centimetres (cm); RSP is dimensionless (RSP of water = 1).
// ---------------------------------------------------------------------------
struct PctGeom {
    int   n     = 0;      // image side length in voxels (image is n*n)
    float half  = 0.0f;   // image spans [-half, half] in x and y (cm)
    // voxel_size() : world cm per voxel. n-1 spacings span 2*half.
    PCT_HD float voxel_size() const { return (n > 1) ? (2.0f * half / (n - 1)) : 0.0f; }
};

// ---------------------------------------------------------------------------
// Proton : ONE detected proton history, i.e. one row of list-mode pCT data.
//   A tracker plane in front of the patient records entry position + direction;
//   a tracker plane behind records exit position + direction; a residual-range
//   detector (calorimeter) records the energy loss, converted to WEPL.
//
//   We parameterise the proton's traversal by a straight "beam axis": the chord
//   from entry point to exit point. `t in [0,1]` walks that chord; the MLP gives
//   the true (curved) lateral offset from the chord at each t.
//
//   Angles are the trajectory direction measured RELATIVE to the entry->exit
//   chord, in radians (small-angle: a few milliradians to tens of mrad). A
//   perfectly straight proton has entry_angle == exit_angle == 0.
// ---------------------------------------------------------------------------
struct Proton {
    float x0, y0;         // entry point (cm), on the front tracker plane
    float x1, y1;         // exit  point (cm), on the back  tracker plane
    float entry_angle;    // entry direction rel. to chord (rad)
    float exit_angle;     // exit  direction rel. to chord (rad)
    float wepl;           // measured water-equivalent path length (cm) = integral of RSP
};

// ---------------------------------------------------------------------------
// tanf_hd : a tiny host/device single-precision tangent wrapper, declared and
//   defined BEFORE mlp_point() (which calls it) so ordinary C++ name-lookup is
//   satisfied. On the device nvcc's `tanf` is used; on the host we compute in
//   double then narrow. Both yield the SAME float for the small angles we use
//   (|a| < ~0.1 rad), so the MLP -- and therefore the reconstruction -- matches
//   bit-for-bit between CPU and GPU. Without this, a std::tan vs tanf mismatch
//   would drift the path by ULPs and break exact verification.
// ---------------------------------------------------------------------------
PCT_HD inline float tanf_hd(float a) {
#ifdef __CUDA_ARCH__
    // Device path: single-precision device intrinsic tangent.
    return tanf(a);
#else
    // Host path: double tangent narrowed to float (matches device tanf here).
    return static_cast<float>(std::tan(static_cast<double>(a)));
#endif
}

// ---------------------------------------------------------------------------
// mlp_point : the most-likely-path (MLP) position of the proton at chord
//   fraction t in [0,1]. Returns world (x,y) through out-params.
//
// THE MATH (simplified, teaching form)
//   The rigorous MLP (Schulte et al. 2008) solves for the trajectory that
//   maximises the likelihood of the measured entry/exit given Gaussian multiple
//   Coulomb scattering; it is a scattering-covariance-weighted combination of
//   two boundary conditions and works out to a CUBIC in the depth coordinate.
//   For a slab of uniform-ish material at small angles, that cubic reduces to
//   the unique cubic HERMITE curve that matches BOTH endpoints' POSITION and
//   SLOPE. We use that Hermite form: it is the correct small-angle limit,
//   captures the S-shaped bend real MLPs show, and is cheap and closed-form (no
//   per-proton matrix solve). THEORY.md "The math" derives the full covariance
//   MLP this approximates and states the tolerance it costs us.
//
//   Construction:
//     * u = along-chord unit vector (entry->exit); v = left-normal unit vector.
//     * The proton starts on the chord (lateral offset 0) with slope set by the
//       entry angle, and ends on the chord (offset 0) with slope set by the exit
//       angle. Lateral offset h(t) is the cubic Hermite with h(0)=h(1)=0 and
//       prescribed endpoint tangents m0,m1 (units: cm per unit-t):
//         h(t) = (t^3 - 2t^2 + t) * m0   [tangent-at-0 basis]
//              + (t^3 -   t^2)     * m1   [tangent-at-1 basis]
//       with m0 = tan(entry_angle)*L, m1 = tan(exit_angle)*L, L = chord length.
//     * Position = chord_point(t) + h(t) * v.
//
// Parameters:
//   p           : the proton history.
//   t           : chord fraction in [0,1] (0 = entry, 1 = exit).
//   out_x,out_y : world position on the MLP at fraction t (cm).
//
// Called O(samples) times per proton by BOTH the CPU loop and the GPU thread --
// identical code, so the sampled line integrals match to the bit.
// ---------------------------------------------------------------------------
PCT_HD inline void mlp_point(const Proton& p, float t, float* out_x, float* out_y) {
    // Chord from entry to exit.
    const float dx = p.x1 - p.x0;
    const float dy = p.y1 - p.y0;
    const float L  = std::sqrt(dx * dx + dy * dy);       // chord length (cm)

    // Point on the straight chord at fraction t (the "beam axis" position).
    const float cx = p.x0 + t * dx;
    const float cy = p.y0 + t * dy;

    // Guard degenerate (zero-length) chords: no lateral direction is defined,
    // so just return the chord point. (Never happens for real geometry.)
    if (L <= 0.0f) { *out_x = cx; *out_y = cy; return; }

    // Left-normal unit vector v: rotate along-chord unit u=(dx,dy)/L by +90deg
    // -> v=(-dy,dx)/L. Lateral offsets h(t) are measured along v.
    const float vx = -dy / L;
    const float vy =  dx / L;

    // Endpoint lateral SLOPES d(offset)/d(length). slope*L makes the Hermite
    // tangent (per unit-t) since the chord parameter t spans length L.
    const float m0 = tanf_hd(p.entry_angle) * L;         // Hermite tangent at t=0
    const float m1 = tanf_hd(p.exit_angle)  * L;         // Hermite tangent at t=1

    // Cubic Hermite with zero endpoint offsets (proton is ON the chord at both
    // tracker planes) and prescribed endpoint tangents -> the S-shaped MLP bend.
    const float t2 = t * t;
    const float t3 = t2 * t;
    const float h  = (t3 - 2.0f * t2 + t) * m0    // tangent-at-0 basis
                   + (t3 -        t2)     * m1;    // tangent-at-1 basis

    // World position = chord point + lateral offset along the normal.
    *out_x = cx + h * vx;
    *out_y = cy + h * vy;
}
