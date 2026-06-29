// ===========================================================================
// src/md.h  --  Shared (host + device) molecular-dynamics physics core
// ---------------------------------------------------------------------------
// Project 1.1 : Molecular Dynamics Engine  (reduced-scope teaching version)
//
// WHAT THIS PROJECT COMPUTES
//   Classical molecular dynamics (MD) follows Newton's second law, F = m*a, for
//   a system of N atoms. We integrate the equations of motion in tiny timesteps,
//   so the atoms move under the forces they exert on one another and we watch the
//   system evolve. This is the beating heart of every production MD engine
//   (GROMACS, OpenMM, NAMD, AMBER) -- they add bonded terms, electrostatics,
//   thermostats and barostats, but the inner loop is always the same:
//       compute forces  ->  advance positions & velocities  ->  repeat.
//
//   This teaching version models the SIMPLEST physically meaningful force field:
//   a single pairwise non-bonded term, the LENNARD-JONES (LJ) 12-6 potential.
//   Every pair of atoms (i, j) attracts at long range (the -1/r^6 dispersion
//   term) and repulses hard at short range (the +1/r^12 Pauli term):
//
//       U(r) = 4*eps * [ (sigma/r)^12 - (sigma/r)^6 ]
//
//   where r = |r_i - r_j| is the inter-atom distance, `eps` sets the depth of the
//   attractive well, and `sigma` sets the distance at which U crosses zero. This
//   is the canonical model of a noble-gas fluid (e.g. liquid argon) and the
//   textbook entry point to MD. Real biomolecular force fields (CHARMM36m,
//   AMBER ff19SB -- see the catalog) add bonds/angles/dihedrals and Coulomb +
//   Particle-Mesh-Ewald electrostatics on top of exactly this LJ term.
//
//   The per-pair physics AND the velocity-Verlet integrator step live here as
//   `__host__ __device__` inline functions, so the CPU reference (reference_cpu)
//   and the GPU kernel (kernels.cu) run BYTE-FOR-BYTE-IDENTICAL math. That makes
//   verification exact rather than approximate (PATTERNS.md §2: the HD-core idiom).
//   We work in DOUBLE precision precisely so the all-pairs force sum -- the same
//   additions, in the same order, on both sides -- agrees to round-off.
//
//   UNITS: we use "LJ reduced units" (the MD convention): set the particle mass
//   m = 1, eps = 1, sigma = 1. Then length is in units of sigma, energy in units
//   of eps, and time in units of sigma*sqrt(m/eps). Temperatures, pressures, etc.
//   all become pure numbers. This removes unit bookkeeping so the learner can
//   focus on the algorithm; a real engine carries kcal/mol, angstrom, fs.
//
// READ THIS AFTER: nothing -- start here, it defines the physics. Then read
//   reference_cpu.h (the serial driver), kernels.cuh, kernels.cu. See ../THEORY.md.
// ===========================================================================
#pragma once

// HD-macro idiom (PATTERNS.md §2): under nvcc (__CUDACC__ defined) the functions
// below are compiled for BOTH the host and the device; under the plain host
// compiler the decorators simply vanish, so reference_cpu.cpp can include this
// header too. Keep CUDA-only constructs (no __global__, no <<<>>>) out of here so
// the host compiler is happy.
#ifdef __CUDACC__
#define MD_HD __host__ __device__
#else
#define MD_HD
#endif

#include <cmath>   // std::sqrt (host); device gets the CUDA intrinsic via overload

// ---------------------------------------------------------------------------
// Vec3: a tiny 3-component double vector for positions, velocities, forces.
//   We roll our own (rather than CUDA's double3) so the SAME type compiles on
//   the host and the device, keeping the HD core self-contained. Plain data, no
//   methods that could differ between compilers -> identical layout everywhere.
// ---------------------------------------------------------------------------
struct Vec3 {
    double x, y, z;
};

// Convenience: zero vector. Used to reset force accumulators each step.
MD_HD inline Vec3 vec3_zero() { return Vec3{0.0, 0.0, 0.0}; }

// ---------------------------------------------------------------------------
// SimParams: everything that defines one simulation run. Passed by value into
//   the kernel (so it lives in each thread's registers / constant bank) and used
//   by the CPU driver. Reduced LJ units throughout (mass=eps=sigma=1).
// ---------------------------------------------------------------------------
struct SimParams {
    int    n        = 0;      // number of atoms
    double box      = 0.0;    // cubic periodic box edge length L (reduced units)
    double dt       = 0.0;    // integration timestep (reduced time units)
    int    steps    = 0;      // number of velocity-Verlet steps to run
    double eps      = 1.0;    // LJ well depth   (reduced: 1)
    double sigma    = 1.0;    // LJ zero-crossing distance (reduced: 1)
    double rcut     = 0.0;    // force/energy cutoff radius (pairs beyond -> 0)
    double mass     = 1.0;    // particle mass m (reduced: 1)
};

// ---------------------------------------------------------------------------
// minimum_image: apply the periodic-boundary "minimum image convention".
//   In a periodic box of edge L, atom j has infinitely many images shifted by
//   multiples of L. The physically correct interaction uses the NEAREST image,
//   so each coordinate difference d is wrapped into the half-open interval
//   (-L/2, +L/2]. The branch-free trick: d -= L * round(d / L).
//     * round() picks the nearest integer number of box lengths to subtract.
//   We isolate this in one function so the CPU and GPU wrap identically (any
//   discrepancy here would desynchronize the two trajectories instantly).
// ---------------------------------------------------------------------------
MD_HD inline double minimum_image(double d, double box) {
    // nearbyint/round: subtract the whole number of boxes that brings |d| < L/2.
    // We use a manual round-half-to-even-free formula (floor(x+0.5)) only if
    // needed; the standard round() is deterministic for our magnitudes, and it
    // is available as a __host__ __device__ overload under nvcc.
    return d - box * round(d / box);
}

// ---------------------------------------------------------------------------
// lj_pair_force: the core two-body interaction. Given the displacement vector
//   rij = r_i - r_j (already minimum-imaged), return the force ON ATOM i due to
//   atom j, and ADD this pair's potential energy into *u_accum.
//
//   The LJ potential and its force (F = -dU/dr, projected onto the rij axis):
//       U(r)      = 4*eps [ (s/r)^12 - (s/r)^6 ]
//       F_vec(r)  = ( 24*eps/r^2 ) [ 2 (s/r)^12 - (s/r)^6 ] * rij
//   We compute everything from r2 = |rij|^2 to avoid a square root in the force:
//       sr2 = (sigma^2)/r2,  sr6 = sr2^3,  sr12 = sr6^2.
//       fscale = 24*eps * (2*sr12 - sr6) / r2     (so F_vec = fscale * rij)
//       u      = 4*eps * (sr12 - sr6)
//
//   CUTOFF: real MD truncates the pair loop at r = rcut (the LJ tail is tiny far
//   out, and a cutoff turns the O(N^2) all-pairs cost into O(N) with a neighbor
//   list -- see THEORY). Beyond rcut we return zero force and zero energy. We use
//   a SIMPLE truncation (no shift) here for clarity; THEORY notes the small energy
//   discontinuity this introduces and how production codes shift/​switch it away.
//
//   PARAMETERS
//     rij      : displacement r_i - r_j after minimum-image wrapping (reduced len)
//     p        : sim parameters (eps, sigma, rcut)
//     u_accum  : in/out; this pair's potential energy is ADDED here (counted once
//                per pair by the caller). Pass nullptr to skip the energy tally.
//   RETURNS    : the force vector on atom i from this single pair.
//
//   This function is the one place the physics lives; both the CPU reference and
//   the GPU kernel call it, guaranteeing identical numbers (PATTERNS.md §2).
// ---------------------------------------------------------------------------
MD_HD inline Vec3 lj_pair_force(Vec3 rij, const SimParams& p, double* u_accum) {
    const double r2 = rij.x*rij.x + rij.y*rij.y + rij.z*rij.z;  // squared distance

    // Skip self-interaction (r2==0) and anything beyond the cutoff sphere. Using
    // rcut^2 avoids a sqrt just to compare against rcut.
    const double rcut2 = p.rcut * p.rcut;
    if (r2 == 0.0 || r2 > rcut2) {
        return vec3_zero();
    }

    // sigma^2 / r^2 and its powers. With sigma=1 (reduced units) sigma2==1, but
    // we keep it general so non-reduced parameters also work.
    const double sigma2 = p.sigma * p.sigma;
    const double sr2  = sigma2 / r2;     // (sigma/r)^2
    const double sr6  = sr2 * sr2 * sr2; // (sigma/r)^6
    const double sr12 = sr6 * sr6;       // (sigma/r)^12

    // Scalar force factor such that F_vec = fscale * rij (points along the axis).
    // Positive fscale => repulsion (pushes i away from j); negative => attraction.
    const double fscale = 24.0 * p.eps * (2.0 * sr12 - sr6) / r2;

    if (u_accum) {
        // 4*eps[(s/r)^12 - (s/r)^6]; the caller adds this ONCE per unordered pair.
        *u_accum += 4.0 * p.eps * (sr12 - sr6);
    }

    return Vec3{ fscale * rij.x, fscale * rij.y, fscale * rij.z };
}

// ---------------------------------------------------------------------------
// kinetic_energy_one: KE contribution of a single atom, 0.5 * m * v^2.
//   Summed over all atoms it gives the system kinetic energy; with the potential
//   energy it forms the TOTAL energy, our headline conserved quantity (a correct
//   integrator keeps it nearly constant -- the key teaching diagnostic).
// ---------------------------------------------------------------------------
MD_HD inline double kinetic_energy_one(Vec3 v, double mass) {
    return 0.5 * mass * (v.x*v.x + v.y*v.y + v.z*v.z);
}

// ---------------------------------------------------------------------------
// wrap_into_box: keep a coordinate inside [0, L) under periodic boundaries.
//   After a position update an atom may drift out of the primary box; we fold it
//   back so coordinates stay bounded (purely cosmetic for the physics because the
//   minimum-image convention already handles interactions, but it keeps positions
//   well-conditioned over long runs and makes the checksum stable).
// ---------------------------------------------------------------------------
MD_HD inline double wrap_into_box(double x, double box) {
    // fmod can return a negative remainder for negative x, so add one box and
    // fmod again to land in [0, L). Deterministic on host and device.
    double y = fmod(x, box);
    if (y < 0.0) y += box;
    return y;
}
