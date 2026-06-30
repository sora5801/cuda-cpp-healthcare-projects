// ===========================================================================
// src/membrane.h  --  Shared (host + device) coarse-grained membrane physics
// ---------------------------------------------------------------------------
// Project 2.19 : Membrane Protein Simulation   (REDUCED-SCOPE teaching version)
//
// WHAT THIS PROJECT COMPUTES
//   A tiny COARSE-GRAINED (CG) molecular-dynamics (MD) patch of a lipid
//   bilayer with a few embedded PROTEIN beads -- the MARTINI-style first stage
//   of the real pipeline in the catalog ("GPU-accelerated CG-MARTINI
//   pre-equilibration ... followed by backmapping to all-atom"). Each lipid is
//   reduced to THREE beads -- one HEAD bead and two TAIL beads -- linked by
//   harmonic bonds; protein beads are heavier inclusions. The beads interact
//   through a truncated Lennard-Jones (LJ) potential whose well depth encodes
//   the hydrophobic effect: tails attract tails, heads stay near the water
//   slabs, so a flat double layer self-stabilises and stays intact while the
//   protein sits in it. We advance the system with velocity-Verlet integration
//   and a Langevin thermostat (friction + random kicks) that holds temperature
//   -- exactly the NVT ensemble a membrane equilibration uses.
//
//   The headline observable we report is the BILAYER THICKNESS (peak-to-peak
//   head-group separation along z) and the membrane's area/energy -- the same
//   quantities you watch to know a membrane has equilibrated.
//
// WHY A GPU
//   The cost of MD is the per-step FORCE evaluation: every bead feels every
//   other bead within a cutoff. With N beads that is O(N^2) pair work per step
//   (production codes cut it to O(N) with neighbour lists + PME; THEORY.md).
//   Crucially, the force on bead i is INDEPENDENT of the force on bead j during
//   a step -- so we give each bead its own GPU thread, which loops over all the
//   others and sums its force. That is the "independent per-item job" pattern
//   (PATTERNS.md section 1); the integration is a second independent per-bead
//   pass. Thousands of beads -> thousands of threads.
//
//   The per-pair force, the per-bead Verlet update, and the deterministic
//   random kick all live HERE as __host__ __device__ functions, so the CPU
//   reference (reference_cpu.cpp) and the GPU kernels (kernels.cu) run
//   BYTE-FOR-BYTE identical math -> verification is near-exact (PATTERNS.md
//   section 2). MEM_HD = __host__ __device__ under nvcc, nothing under the host
//   compiler, so this header is includable by BOTH cl.exe and nvcc.
//
// READ THIS AFTER: main.cu (the 5-step shape), then reference_cpu.h, kernels.cuh.
// ===========================================================================
#pragma once

#include <cmath>      // sqrt, sin, cos, floor
#include <cstdint>    // uint32_t, uint64_t

// MEM_HD expands to the CUDA decorators only when this header is seen by nvcc
// (which defines __CUDACC__). The host compiler (cl.exe / g++) sees nothing,
// so the very same source compiles in BOTH worlds -- the key to CPU/GPU parity.
#ifdef __CUDACC__
#define MEM_HD __host__ __device__
#else
#define MEM_HD
#endif

// ---------------------------------------------------------------------------
// Bead "species". A coarse-grained model lumps several atoms into one bead and
// gives each bead a TYPE that sets how strongly it attracts/repels others. We
// keep three didactic types; their pairwise well depths live in the params.
//   HEAD  : the polar lipid head group (likes the water region, mild attraction)
//   TAIL  : the hydrophobic lipid tail bead (tails strongly attract each other,
//           which is what drives the two leaflets to stack into a bilayer)
//   PROT  : a protein bead (a heavier inclusion embedded in the membrane)
// We store the type as a small int; the index into the 3x3 epsilon matrix.
// ---------------------------------------------------------------------------
enum BeadType { BEAD_HEAD = 0, BEAD_TAIL = 1, BEAD_PROT = 2, BEAD_NTYPES = 3 };

// --- Minimal 3-vector with just the operations MD needs (host + device) -----
// We use double throughout: MD force sums are catastrophically sensitive to
// precision (small forces between many beads), and double keeps CPU and GPU in
// lock-step. (Production GPU MD uses mixed FP32/FP64; see THEORY "Numerics".)
struct Vec3 { double x, y, z; };
MEM_HD inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
MEM_HD inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
MEM_HD inline Vec3 operator*(Vec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
MEM_HD inline double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
MEM_HD inline double length(Vec3 a) { return sqrt(dot(a, a)); }

// ---------------------------------------------------------------------------
// SimParams: everything that defines the run, read once from the data file and
// then passed BY VALUE into every kernel (it is small and read-only, so copying
// it into each thread's registers is cheaper than chasing a pointer).
//
// Units are "reduced MD units" (a common teaching choice): lengths in sigma,
// energies in epsilon, masses in m, time in tau = sqrt(m sigma^2 / epsilon).
// Reduced units keep the numbers O(1) and the physics dimensionless -- exactly
// how MARTINI-style CG models are often taught. THEORY.md maps these to nm/kJ.
// ---------------------------------------------------------------------------
struct SimParams {
    int    n_lipids;     // number of lipid molecules (3 beads each)
    int    n_prot;       // number of protein beads
    int    n_beads;      // = 3*n_lipids + n_prot   (total particles; convenience)

    double box_x, box_y; // periodic box size in x,y (the membrane plane). z is
                         // NON-periodic (a free slab) -- a 2D-periodic "slab"
                         // geometry, the same simplification real membrane MD
                         // makes for the bilayer normal (THEORY "real world").

    double sigma;        // LJ size parameter (bead diameter), = 1 in reduced units
    double rcut;         // LJ cutoff radius: beyond this we ignore the pair
                         // (truncation -> O(N*neighbours) instead of all pairs;
                         // here N is tiny so we still loop all pairs but skip
                         // anything past rcut, which is what a neighbour list does)
    double eps[BEAD_NTYPES][BEAD_NTYPES];  // LJ well depth per type pair (symmetric)

    double k_bond;       // harmonic bond stiffness (head-tail, tail-tail springs)
    double r_bond;       // bond rest length

    double dt;           // integration timestep (reduced units)
    int    steps;        // number of MD steps to run

    double temperature;  // target temperature kT (Langevin thermostat setpoint)
    double gamma;        // Langevin friction coefficient (1/time)
    uint64_t seed;       // master RNG seed -> deterministic random forces
};

// ---------------------------------------------------------------------------
// DETERMINISTIC random numbers (counter-based PRNG)
//   A Langevin thermostat adds a RANDOM force to every bead every step. If the
//   CPU and GPU drew DIFFERENT randoms, their trajectories would diverge and we
//   could not verify GPU==CPU. The fix (PATTERNS.md section 3): a STATELESS,
//   counter-based PRNG. Instead of a sequence with hidden state, we HASH the
//   tuple (seed, step, bead, component) into a number. Same inputs -> same
//   output, on CPU and GPU alike, in any order, with no shared state. This is
//   the idea behind cuRAND's Philox; we use a small public-domain integer hash
//   (SplitMix64) so the header stays dependency-free and host-includable.
// ---------------------------------------------------------------------------

// SplitMix64: a well-tested 64-bit mixing function. Given any 64-bit input it
// returns a well-scrambled 64-bit output. Deterministic and branch-free.
MEM_HD inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;                       // golden-ratio increment
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Map a 64-bit hash to a double uniform in [0,1). We take the top 53 bits (the
// mantissa width of a double) so every representable value is reachable.
MEM_HD inline double u64_to_unit(uint64_t h) {
    return (h >> 11) * (1.0 / 9007199254740992.0);    // / 2^53
}

// One standard normal N(0,1) from two uniforms via the Box-Muller transform.
// We derive the two uniforms from two DIFFERENT hashed counters so they are
// independent. (Box-Muller is exact and reproducible; cuRAND offers it too.)
//   key encodes (step, bead, component) so each (step,bead,axis) gets its own,
//   reproducible normal draw on both CPU and GPU.
MEM_HD inline double normal01(uint64_t seed, uint64_t key) {
    // Two independent hashes from the same key via different salts.
    const uint64_t h1 = splitmix64(seed ^ (key * 2u + 1u));
    const uint64_t h2 = splitmix64(seed ^ (key * 2u + 2u));
    double u1 = u64_to_unit(h1);
    double u2 = u64_to_unit(h2);
    // Guard the log against u1 == 0 (would be -inf); nudge to the smallest unit.
    if (u1 < 1e-300) u1 = 1e-300;
    const double r = sqrt(-2.0 * log(u1));
    const double theta = 6.283185307179586 * u2;       // 2*pi*u2
    return r * cos(theta);                              // one normal; cheap & enough
}

// Pack (step, bead, axis) into a single 64-bit key for the PRNG. Distinct
// tuples must give distinct keys; we shift each field into its own bit-field.
//   axis in {0,1,2}; bead < n_beads (<< 2^32 here); step < 2^29 for our runs.
MEM_HD inline uint64_t rng_key(int step, int bead, int axis) {
    return (static_cast<uint64_t>(step) << 34)
         ^ (static_cast<uint64_t>(bead) << 2)
         ^ static_cast<uint64_t>(axis);
}

// ---------------------------------------------------------------------------
// MINIMUM-IMAGE convention in x,y (periodic membrane plane)
//   The membrane plane is periodic: a bead near the right edge is a neighbour
//   of one near the left edge. The "minimum image" of a displacement wraps it
//   into [-L/2, +L/2) so we always use the SHORTEST copy. z is NOT periodic.
// ---------------------------------------------------------------------------
MEM_HD inline double min_image(double d, double L) {
    // Subtract the nearest whole number of box lengths. round() ties-to-even is
    // identical on host and device for our magnitudes, keeping CPU==GPU.
    return d - L * floor(d / L + 0.5);
}

// Displacement r_i - r_j under the minimum-image convention (x,y periodic).
MEM_HD inline Vec3 min_image_delta(Vec3 ri, Vec3 rj, double Lx, double Ly) {
    Vec3 d = ri - rj;
    d.x = min_image(d.x, Lx);
    d.y = min_image(d.y, Ly);
    // z is a free slab: no wrapping.
    return d;
}

// ---------------------------------------------------------------------------
// LENNARD-JONES pair force (the heart of the model)
//   U_LJ(r) = 4 eps [ (sigma/r)^12 - (sigma/r)^6 ]
//   The r^-12 term is steep repulsion (beads cannot overlap); the r^-6 term is
//   the attractive van der Waals well of depth eps at r = 2^(1/6) sigma. By
//   making TAIL-TAIL eps large and HEAD-TAIL eps small we encode the
//   hydrophobic effect: tails clump, heads face outward -> a bilayer.
//
//   The FORCE is -dU/dr along the unit vector r_hat. Working it out:
//     F(r) = 24 eps/r * [ 2 (sigma/r)^12 - (sigma/r)^6 ] * r_hat
//   We return the force VECTOR on bead i from bead j (Newton's 3rd law gives
//   the opposite on j). Returns zero past the cutoff. `eps` is looked up from
//   the type pair by the caller.
//
//   Returns the force on i; also writes the pair POTENTIAL energy via `*u_out`
//   (shifted so U(rcut)=0, the standard truncation, so energy is continuous).
// ---------------------------------------------------------------------------
MEM_HD inline Vec3 lj_force(Vec3 dij, double eps, double sigma, double rcut, double* u_out) {
    const double r2 = dot(dij, dij);
    const double rc2 = rcut * rcut;
    if (r2 >= rc2 || r2 < 1e-12) { *u_out = 0.0; return {0.0, 0.0, 0.0}; }

    const double inv_r2  = 1.0 / r2;
    const double s2      = sigma * sigma;
    const double sr2     = s2 * inv_r2;        // (sigma/r)^2
    const double sr6     = sr2 * sr2 * sr2;    // (sigma/r)^6
    const double sr12    = sr6 * sr6;          // (sigma/r)^12

    // Magnitude/r of the force so that F_vec = (fmag_over_r) * dij gives the
    // correctly-directed vector without a sqrt (dij already carries direction).
    const double fmag_over_r = 24.0 * eps * inv_r2 * (2.0 * sr12 - sr6);

    // Energy shift so U(rcut)=0: subtract the value at the cutoff. This removes
    // the tiny discontinuity at rcut and makes total energy a smooth diagnostic.
    const double src2  = s2 / rc2;
    const double src6  = src2 * src2 * src2;
    const double src12 = src6 * src6;
    const double u_full  = 4.0 * eps * (sr12 - sr6);
    const double u_shift = 4.0 * eps * (src12 - src6);
    *u_out = u_full - u_shift;

    return dij * fmag_over_r;                   // force on i (points along +dij)
}

// ---------------------------------------------------------------------------
// HARMONIC BOND force
//   Lipid beads within a molecule are wired head-tail and tail-tail by springs:
//     U_bond(r) = 1/2 k (r - r0)^2,   F = -k (r - r0) * r_hat
//   This keeps a lipid from flying apart and gives it its rod-like shape.
//   Returns the force on i from its bonded partner j; writes the energy too.
// ---------------------------------------------------------------------------
MEM_HD inline Vec3 bond_force(Vec3 dij, double k, double r0, double* u_out) {
    const double r = length(dij);
    if (r < 1e-12) { *u_out = 0.0; return {0.0, 0.0, 0.0}; }
    const double dr = r - r0;                   // extension (signed)
    *u_out = 0.5 * k * dr * dr;
    // F = -k (r - r0) * (dij / r). Negative sign pulls back toward rest length.
    const double fmag_over_r = -k * dr / r;
    return dij * fmag_over_r;
}

// ---------------------------------------------------------------------------
// VELOCITY-VERLET integration split into the two standard half-steps.
//   Verlet is the workhorse MD integrator: symplectic (conserves energy over
//   long runs), time-reversible, and only needs one force evaluation per step.
//   Split form, per step:
//     (A) half-kick + drift:  v += (f/m)*(dt/2);  x += v*dt
//     (B) recompute forces f(x)
//     (C) half-kick:          v += (f/m)*(dt/2)
//   We expose (A) and (C) as functions so CPU and GPU share them exactly.
// ---------------------------------------------------------------------------

// Half-kick + drift (step A). Updates v by half a kick, then moves x by v*dt.
MEM_HD inline void verlet_kick_drift(Vec3& x, Vec3& v, Vec3 f, double inv_mass, double dt) {
    v = v + f * (inv_mass * 0.5 * dt);          // half-kick with current force
    x = x + v * dt;                             // drift with the new velocity
}

// Final half-kick (step C). Applies the second half-kick with the NEW force.
MEM_HD inline void verlet_kick(Vec3& v, Vec3 f, double inv_mass, double dt) {
    v = v + f * (inv_mass * 0.5 * dt);
}

// ---------------------------------------------------------------------------
// LANGEVIN thermostat force on one bead.
//   To hold temperature (the NVT ensemble of a membrane equilibration), we add
//   to each bead a FRICTION force -gamma*m*v that bleeds energy, plus a RANDOM
//   force whose variance is tuned (fluctuation-dissipation theorem) so the
//   system relaxes to the target kT:
//       f_lang = -gamma m v + sqrt(2 gamma m kT / dt) * xi,   xi ~ N(0,1)
//   The random kick uses our deterministic normal01(), so CPU and GPU draw the
//   SAME xi for the same (step,bead,axis) -> identical trajectories.
//   mass is passed (not inv_mass) because both terms scale with m.
// ---------------------------------------------------------------------------
MEM_HD inline Vec3 langevin_force(Vec3 v, double mass, double gamma, double kT,
                                  double dt, uint64_t seed, int step, int bead) {
    const double coeff = sqrt(2.0 * gamma * mass * kT / dt);  // noise amplitude
    Vec3 f;
    f.x = -gamma * mass * v.x + coeff * normal01(seed, rng_key(step, bead, 0));
    f.y = -gamma * mass * v.y + coeff * normal01(seed, rng_key(step, bead, 1));
    f.z = -gamma * mass * v.z + coeff * normal01(seed, rng_key(step, bead, 2));
    return f;
}

// ---------------------------------------------------------------------------
// Convenience: the LJ well depth for a pair of bead types, read from the matrix.
// (A tiny wrapper so kernels.cu and reference_cpu.cpp index epsilon identically.)
// ---------------------------------------------------------------------------
MEM_HD inline double eps_of(const SimParams& P, int ti, int tj) {
    return P.eps[ti][tj];
}
