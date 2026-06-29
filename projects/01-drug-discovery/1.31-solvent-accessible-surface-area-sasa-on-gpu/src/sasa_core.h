// ===========================================================================
// src/sasa_core.h  --  The ONE TRUE per-atom SASA math, shared by CPU and GPU
// ---------------------------------------------------------------------------
// Project 1.31 : Solvent-Accessible Surface Area (SASA) on GPU
//
// WHY THIS HEADER EXISTS  (the HD-macro idiom, PATTERNS.md sec 2)
//   The whole project rests on one numerical claim: "the GPU computes the same
//   SASA as the trusted CPU reference." The cleanest way to GUARANTEE that is to
//   make both sides call the *exact same functions*. So the per-atom physics --
//   how we lay out test points on a sphere, and how we decide a point is buried
//   -- lives here ONCE, decorated `__host__ __device__`, and is compiled into:
//       * reference_cpu.cpp  (by the host C++ compiler, where SASA_HD vanishes), and
//       * kernels.cu         (by nvcc, where SASA_HD becomes __host__ __device__).
//   Result: byte-for-byte identical arithmetic => exact verification, not "close".
//
//   This header is pure C++ + <cmath> (NO __global__, NO CUDA types), so the
//   host compiler can include it happily. CUDA-only code stays in kernels.cu.
//
// READ THIS FIRST (it defines the science). Then reference_cpu.h, then kernels.cuh.
// The derivation and notation are in ../THEORY.md.
// ===========================================================================
#pragma once

#include <cmath>     // std::sqrt, std::cos, std::sin, M_PI (we define PI ourselves)
#include <cstdint>   // fixed-width ints for the deterministic counting

// ---------------------------------------------------------------------------
// The HD ("host+device") decorator switch.
//   * When nvcc compiles this file (it defines __CUDACC__), SASA_HD expands to
//     `__host__ __device__`, so each inline function is emitted for BOTH the CPU
//     and the GPU. The kernel can then call them from a device thread.
//   * When the plain host compiler compiles reference_cpu.cpp, __CUDACC__ is not
//     defined, so SASA_HD expands to nothing and the functions are ordinary host
//     inlines. Same source text, same math, two targets.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define SASA_HD __host__ __device__
#else
#define SASA_HD
#endif

// Pi to double precision. We define our own constant instead of relying on the
// non-standard M_PI so the code is portable across compilers (MSVC does not
// define M_PI without _USE_MATH_DEFINES) and identical on host and device.
#ifndef SASA_PI
#define SASA_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// Probe radius and the Shrake-Rupley test-point count.
//   PROBE_RADIUS = 1.4 Angstrom is the textbook radius of a water molecule; the
//     "solvent-accessible surface" is the locus traced by this probe's CENTER as
//     it rolls over the atoms, which is why every atom is INFLATED by 1.4 A.
//   N_SPHERE_POINTS = 96: the number of test points spread over each atom's
//     sphere. More points = a finer (more accurate, slower) surface integral.
//     The classic Shrake-Rupley paper used 92 (a geodesic sphere); we use 96 on
//     a Fibonacci lattice (see fib_point) because it is trivial to generate on
//     both CPU and GPU with no lookup tables. It is a COMPILE-TIME constant so
//     the inner loop bound is known and the same value is baked into both sides.
// ---------------------------------------------------------------------------
constexpr double PROBE_RADIUS    = 1.4;   // Angstrom, water probe
constexpr int    N_SPHERE_POINTS = 96;    // Shrake-Rupley test points per atom

// An atom as the SASA computation sees it: a center (x,y,z) in Angstrom and a
// van der Waals radius (Angstrom). Element identity has already been mapped to a
// radius by the loader; SASA only needs the geometry. POD so it copies trivially
// to the device with a single cudaMemcpy.
struct Atom {
    double x, y, z;   // center coordinates, Angstrom
    double r;         // van der Waals radius, Angstrom
};

// ---------------------------------------------------------------------------
// fib_point: the k-th of N points spread (nearly) uniformly on the UNIT sphere
//            via the Fibonacci-spiral lattice.
//   Idea: walk up the sphere in equal z-steps while spinning by the golden angle
//   each step. This gives a low-discrepancy, deterministic point set with no
//   trig tables and no clustering at the poles -- ideal for a surface integral
//   that must match bit-for-bit on CPU and GPU.
//   Inputs : k in [0, n), n = number of points (= N_SPHERE_POINTS).
//   Outputs: (ux,uy,uz) a unit vector (||u|| = 1) -- the direction of test point k.
//   Determinism: uses only +,-,*,/,sqrt,cos,sin on doubles -> identical on both
//   sides (IEEE-754). See THEORY "Numerical considerations".
// ---------------------------------------------------------------------------
SASA_HD inline void fib_point(int k, int n, double& ux, double& uy, double& uz) {
    // z marches linearly from near +1 down to near -1 across the n points. The
    // (2k+1)/n offset centers the samples in their bands (avoids the exact poles).
    const double z = 1.0 - (2.0 * k + 1.0) / static_cast<double>(n);
    const double r = std::sqrt(1.0 - z * z);          // radius of the latitude circle
    // Golden angle = pi * (3 - sqrt(5)) ~= 2.399963 rad. Multiplying by k spins
    // each successive point by this irrational fraction of a turn -> no two points
    // share a longitude, so they interleave instead of lining up.
    const double golden = SASA_PI * (3.0 - std::sqrt(5.0));
    const double theta  = golden * k;
    ux = std::cos(theta) * r;
    uy = std::sin(theta) * r;
    uz = z;
}

// ---------------------------------------------------------------------------
// point_is_buried: is test point P (lying on atom `self`'s inflated surface)
//                  hidden inside ANY other atom's inflated sphere?
//   A test point counts as solvent-ACCESSIBLE only if it is outside every other
//   atom's probe-inflated sphere. As soon as one neighbor covers it, it is
//   buried and contributes no surface area.
//   Inputs:
//     px,py,pz : the test point, in absolute Angstrom coordinates.
//     atoms    : the full atom array (so we can test against neighbors).
//     n        : number of atoms.
//     self     : index of the atom that OWNS this test point (skip it).
//     probe    : probe radius added to every neighbor's vdW radius.
//   Returns true if buried (covered by some neighbor), false if exposed.
//   Complexity: O(n) here (all-pairs). THEORY explains the neighbor-list speedup.
//   We compare SQUARED distances to avoid a sqrt per neighbor -- and, crucially,
//   the same squared-distance test runs on CPU and GPU so the boundary decision
//   is identical (no sqrt rounding to disagree on). See THEORY "verify".
// ---------------------------------------------------------------------------
SASA_HD inline bool point_is_buried(double px, double py, double pz,
                                    const Atom* atoms, int n, int self,
                                    double probe) {
    for (int j = 0; j < n; ++j) {
        if (j == self) continue;                 // a point is never buried by its own atom
        const double rj = atoms[j].r + probe;    // neighbor j's inflated radius
        const double dx = px - atoms[j].x;
        const double dy = py - atoms[j].y;
        const double dz = pz - atoms[j].z;
        const double d2 = dx * dx + dy * dy + dz * dz;
        // Strictly-inside test (`<`, not `<=`): a point grazing the surface is
        // still considered exposed. Using the same operator on both sides keeps
        // the rare boundary case deterministic.
        if (d2 < rj * rj) return true;           // covered by neighbor j -> buried
    }
    return false;                                // survived every neighbor -> exposed
}

// ---------------------------------------------------------------------------
// count_exposed_points: how many of this atom's N_SPHERE_POINTS test points are
//                       solvent-accessible. THIS IS THE DETERMINISTIC CORE.
//   We return an INTEGER count (0..N_SPHERE_POINTS). Integers are exact and
//   order-independent, so the CPU and GPU agree on this number EXACTLY -- there
//   is no floating-point sum to disagree on. The continuous SASA in Angstrom^2
//   is then a single multiply by the per-point area (see atom_sasa), which is
//   also identical on both sides. (PATTERNS.md sec 3/4: count in integers, derive
//   the float once.)
//   Inputs:
//     i     : index of the atom we are scoring.
//     atoms : the atom array; n : its length; probe : probe radius.
//   Returns the number of exposed test points for atom i.
// ---------------------------------------------------------------------------
SASA_HD inline int count_exposed_points(int i, const Atom* atoms, int n, double probe) {
    const Atom a = atoms[i];
    const double surf_r = a.r + probe;     // radius of atom i's solvent-accessible sphere
    int exposed = 0;
    for (int k = 0; k < N_SPHERE_POINTS; ++k) {
        // Place test point k on atom i's inflated surface: center + surf_r * unit.
        double ux, uy, uz;
        fib_point(k, N_SPHERE_POINTS, ux, uy, uz);
        const double px = a.x + surf_r * ux;
        const double py = a.y + surf_r * uy;
        const double pz = a.z + surf_r * uz;
        if (!point_is_buried(px, py, pz, atoms, n, i, probe)) ++exposed;
    }
    return exposed;
}

// ---------------------------------------------------------------------------
// atom_sasa: convert an exposed-point COUNT into an area in Angstrom^2.
//   The full inflated sphere has area 4*pi*surf_r^2, sampled by N_SPHERE_POINTS
//   equal-weight points, so each exposed point represents (4*pi*surf_r^2)/N of
//   accessible area. SASA_i = exposed * that per-point area.
//   Separated from the counting so the integer (exact) and float (derived) parts
//   are visibly distinct -- and so reference_cpu and the kernel share this too.
// ---------------------------------------------------------------------------
SASA_HD inline double atom_sasa(int exposed, double surf_r) {
    const double full_area      = 4.0 * SASA_PI * surf_r * surf_r;       // whole sphere
    const double area_per_point = full_area / static_cast<double>(N_SPHERE_POINTS);
    return static_cast<double>(exposed) * area_per_point;
}
