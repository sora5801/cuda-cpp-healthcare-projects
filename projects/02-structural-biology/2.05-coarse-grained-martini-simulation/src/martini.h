// ===========================================================================
// src/martini.h  --  Shared (host + device) coarse-grained MARTINI physics
// ---------------------------------------------------------------------------
// Project 2.5 : Coarse-Grained / MARTINI Simulation
//
// WHAT THIS PROJECT COMPUTES
//   A tiny coarse-grained (CG) molecular-dynamics simulation in the spirit of
//   the MARTINI force field. In MARTINI, ~4 heavy atoms are lumped into ONE
//   "bead" (a single interaction site), so a whole lipid is ~12 beads instead
//   of ~50 atoms. That ~4x reduction in particle count -- plus the loss of the
//   fast hydrogen vibrations -- is exactly what buys MARTINI its ~100x reach in
//   simulated time (microseconds-to-milliseconds) over all-atom MD.
//
//   Here we simulate a small box of CG beads of two MARTINI-like types:
//     - type 0 = "C" (apolar / lipid-tail-like) beads, and
//     - type 1 = "P" (polar / water-like)        beads.
//   Beads interact through a pairwise LENNARD-JONES (LJ) potential whose well
//   depth depends on the type pair (the heart of the MARTINI interaction
//   matrix: like-likes-like, oil and water demix). We integrate Newton's
//   equations with velocity-Verlet inside a cubic box with the minimum-image
//   convention (periodic boundaries), and report the system's energy and the
//   demixing of C from P beads.
//
// WHY A GPU
//   The dominant cost of MD is the NON-BONDED PAIR force: every bead feels a
//   force from every other nearby bead. For N beads that is O(N^2) pair work
//   per step (or O(N) with a neighbour list at production scale). Each bead's
//   total force is an INDEPENDENT sum over its partners, so the natural GPU
//   mapping is ONE THREAD PER BEAD: thread i loops over all j, accumulates the
//   force on bead i, and writes it out. No two threads write the same output,
//   so there are no atomics and no races (see THEORY "GPU mapping").
//
//   The per-pair force law and the velocity-Verlet update live HERE as
//   __host__ __device__ inline functions, so the CPU reference and the GPU
//   kernels run BYTE-FOR-BYTE identical math. Because each thread sums its
//   pair contributions in the SAME index order as the CPU loop, the two paths
//   produce the same floating-point result -> exact verification (THEORY §6).
//   MD_HD expands to __host__ __device__ under nvcc, and to nothing under the
//   plain host compiler that builds reference_cpu.cpp.
// ===========================================================================
#pragma once

#include <cmath>

#ifdef __CUDACC__
#define MD_HD __host__ __device__
#else
#define MD_HD            // host compiler: the CUDA decorators do not exist
#endif

// Number of distinct MARTINI-like bead types in this teaching model.
// (Real MARTINI 3 has ~800 typed interaction levels; we use two so the demo's
// physics -- oil/water demixing -- is legible. See THEORY "real world".)
#define MD_NTYPES 2

// --- Minimal 3-vector with just the operations the MD integrator needs. ---
// We hand-roll this (rather than use float3/double3) so the SAME struct compiles
// under both nvcc and the host compiler, keeping the host/device math identical.
struct Vec3 { double x, y, z; };

MD_HD inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
MD_HD inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
MD_HD inline Vec3 operator*(Vec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
MD_HD inline double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// All the constants that define one simulation. Passed by value to the kernels
// (it is tiny and read-only), which keeps the launch self-contained.
struct MdParams {
    int    n;          // number of CG beads
    double box;        // cubic box edge length (nm); periodic in x,y,z
    double dt;         // timestep (reduced units)
    int    steps;      // number of velocity-Verlet steps
    double rcut;       // non-bonded cutoff radius (nm): pairs beyond rcut ignored
    double mass;       // bead mass (all beads share one mass in this model)
    // MARTINI-style interaction matrix, flattened [MD_NTYPES*MD_NTYPES]:
    //   eps[a*NT+b] = LJ well depth for a type-a / type-b pair (kJ/mol-ish).
    //   sigma       = LJ contact distance (nm), shared by all pairs here.
    double eps[MD_NTYPES * MD_NTYPES];
    double sigma;
};

// ---------------------------------------------------------------------------
// min_image: nearest-image displacement under periodic boundaries.
//   Returns the component d shifted into [-box/2, +box/2). This is the
//   "minimum-image convention": a bead interacts with the CLOSEST copy of its
//   partner across the periodic walls, not the one in the home box. Using
//   round() (nearest integer) makes this branch-free and identical on host/GPU.
// ---------------------------------------------------------------------------
MD_HD inline double min_image(double d, double box) {
    // Subtract the box length times the nearest number of whole boxes.
    return d - box * std::nearbyint(d / box);
}

// ---------------------------------------------------------------------------
// lj_pair_force: the Lennard-Jones force ON bead i FROM one partner j.
//   The LJ potential is  U(r) = 4*eps*[ (sigma/r)^12 - (sigma/r)^6 ],
//   a steep repulsive wall (r^-12) plus a softer attractive well (r^-6).
//   The force is f = -dU/dr along the unit vector (r_i - r_j)/r:
//     f(r) = 24*eps/r^2 * [ 2*(sigma/r)^12 - (sigma/r)^6 ] * (r_i - r_j)
//   We return the force VECTOR added to bead i; bead j gets the negative
//   (Newton's third law) -- but in the per-thread kernel each bead recomputes
//   its own sum, so we simply return i's contribution here.
//
//   Beyond `rcut` the pair contributes nothing (a hard cutoff). Production
//   MARTINI uses a smoothly SHIFTED potential so energy is continuous at rcut;
//   we keep the plain cutoff for clarity and discuss the shift in THEORY §5.
//
//   PARAMETERS
//     rij   : displacement r_i - r_j, already minimum-imaged (nm)
//     r2    : squared length of rij (nm^2), passed in to avoid recomputing
//     eps   : LJ well depth for this type pair
//     sigma : LJ contact distance (nm)
//     rcut2 : squared cutoff (nm^2)
//   RETURNS the force vector on bead i due to bead j (zero if beyond cutoff).
// ---------------------------------------------------------------------------
MD_HD inline Vec3 lj_pair_force(Vec3 rij, double r2, double eps,
                                double sigma, double rcut2) {
    if (r2 >= rcut2 || r2 < 1e-12) return {0.0, 0.0, 0.0};  // out of range / self
    const double inv_r2  = 1.0 / r2;                 // 1/r^2
    const double s2      = sigma * sigma * inv_r2;   // (sigma/r)^2
    const double s6      = s2 * s2 * s2;             // (sigma/r)^6
    const double s12     = s6 * s6;                  // (sigma/r)^12
    // Scalar force magnitude / r, ready to multiply the displacement vector.
    const double fmag = 24.0 * eps * inv_r2 * (2.0 * s12 - s6);
    return rij * fmag;
}

// ---------------------------------------------------------------------------
// lj_pair_energy: the Lennard-Jones POTENTIAL energy of one pair (for tallying
//   the system energy -- a teaching diagnostic, not used in the force loop).
//   Same cutoff as the force. Returns 0 beyond rcut.
// ---------------------------------------------------------------------------
MD_HD inline double lj_pair_energy(double r2, double eps, double sigma, double rcut2) {
    if (r2 >= rcut2 || r2 < 1e-12) return 0.0;
    const double s2  = sigma * sigma / r2;
    const double s6  = s2 * s2 * s2;
    const double s12 = s6 * s6;
    return 4.0 * eps * (s12 - s6);
}

// ---------------------------------------------------------------------------
// compute_force_on: the WHOLE force on bead i -- the sum over every other bead.
//   This is the per-thread inner loop, shared verbatim by the CPU reference and
//   the GPU kernel so both sum in the SAME order (j = 0,1,...,n-1, skipping i).
//   That shared order is what makes the GPU result bit-identical to the CPU's.
//
//   PARAMETERS
//     i    : index of the bead whose force we want
//     pos  : array of all bead positions [n]
//     type : array of all bead types     [n]  (0 = C, 1 = P)
//     P    : simulation parameters (box, eps matrix, sigma, rcut)
//   RETURNS the net force vector on bead i.
// ---------------------------------------------------------------------------
MD_HD inline Vec3 compute_force_on(int i, const Vec3* pos, const int* type,
                                   const MdParams& P) {
    const Vec3   ri    = pos[i];
    const int    ti    = type[i];
    const double rcut2 = P.rcut * P.rcut;
    Vec3 f = {0.0, 0.0, 0.0};
    // Loop over ALL beads in fixed index order; skip self. The deterministic
    // order matters: floating-point addition is not associative, so summing in
    // a different order would give a slightly different result on the GPU.
    for (int j = 0; j < P.n; ++j) {
        if (j == i) continue;
        // Minimum-image displacement r_i - r_j across periodic walls.
        Vec3 d;
        d.x = min_image(ri.x - pos[j].x, P.box);
        d.y = min_image(ri.y - pos[j].y, P.box);
        d.z = min_image(ri.z - pos[j].z, P.box);
        const double r2 = dot(d, d);
        // Well depth for this specific (ti, tj) MARTINI type pair.
        const double eps = P.eps[ti * MD_NTYPES + type[j]];
        f = f + lj_pair_force(d, r2, eps, P.sigma, rcut2);
    }
    return f;
}

// ---------------------------------------------------------------------------
// verlet_kick_drift: the first half of one velocity-Verlet step for bead i.
//   Velocity-Verlet (the standard MD integrator) splits each step into:
//     (A) half-kick:   v += 0.5 * (f/m) * dt          [uses OLD force]
//     (B) drift:       x += v * dt                    [move at new v]
//   then forces are recomputed at the new positions, and
//     (C) half-kick:   v += 0.5 * (f_new/m) * dt      [uses NEW force]
//   This is time-reversible and conserves energy far better than naive Euler --
//   the reason essentially all MD codes use it. This function does (A)+(B);
//   verlet_kick (below) does (C). Positions are wrapped back into the box so
//   coordinates stay bounded (purely cosmetic; min-image handles the physics).
//
//   PARAMETERS (all in/out via references for x and v)
//     x   : position of bead i (updated in place)
//     v   : velocity of bead i (updated in place)
//     f   : force on bead i from the PREVIOUS force evaluation
//     P   : simulation parameters (dt, mass, box)
// ---------------------------------------------------------------------------
MD_HD inline void verlet_kick_drift(Vec3& x, Vec3& v, Vec3 f, const MdParams& P) {
    const double half_dt_over_m = 0.5 * P.dt / P.mass;
    v = v + f * half_dt_over_m;     // (A) half-kick with the old force
    x = x + v * P.dt;               // (B) drift to the new position
    // Wrap each coordinate into [0, box) so positions do not run off to infinity.
    x.x -= P.box * std::floor(x.x / P.box);
    x.y -= P.box * std::floor(x.y / P.box);
    x.z -= P.box * std::floor(x.z / P.box);
}

// ---------------------------------------------------------------------------
// verlet_kick: the second half-kick (C) using the freshly recomputed force.
//   Splitting the kicks around the drift is what makes Verlet symplectic.
// ---------------------------------------------------------------------------
MD_HD inline void verlet_kick(Vec3& v, Vec3 f, const MdParams& P) {
    const double half_dt_over_m = 0.5 * P.dt / P.mass;
    v = v + f * half_dt_over_m;     // (C) half-kick with the new force
}
