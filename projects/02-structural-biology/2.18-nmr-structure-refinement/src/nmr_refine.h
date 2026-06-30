// ===========================================================================
// src/nmr_refine.h  --  Shared (host + device) NMR restraint physics + SA core
// ---------------------------------------------------------------------------
// Project 2.18 : NMR Structure Refinement   (reduced-scope teaching version)
//
// WHAT THIS PROJECT COMPUTES (the science, in one breath)
//   Solution-NMR does not hand you a structure. It hands you a list of
//   RESTRAINTS: pairs of atoms that NOE (Nuclear Overhauser Effect) cross-peaks
//   say are "close" (an upper bound on their distance, typically < 5-6 A), plus
//   the covalent geometry that any protein chain must obey (bonded neighbours sit
//   at a fixed spacing). The job of structure REFINEMENT is to find 3-D
//   coordinates that satisfy as many of those restraints as possible. The
//   standard tool for that search is restrained SIMULATED ANNEALING (SA): start
//   hot and random, make random moves, accept good ones always and bad ones with
//   a temperature-dependent probability, and cool slowly so the structure settles
//   into a low-"energy" (low-violation) conformation.
//
//   Because the NOE data is sparse and noisy, no single SA run is trusted. Real
//   NMR pipelines (XPLOR-NIH, CYANA, ARIA -- see README "Prior art") run an
//   ENSEMBLE of hundreds of independent SA trajectories from different random
//   seeds and keep the lowest-energy members. THAT ensemble is the published
//   "NMR structure". The ensemble is embarrassingly parallel: each trajectory is
//   independent, so we give each one its own GPU thread. This is the same
//   "ensemble of independent integrators, one thread per member" pattern as the
//   9.02 SEIR and 13.02 PBPK flagships (PATTERNS.md section 1, row "same ODE for
//   many parameter sets") -- here the per-thread loop is a Monte-Carlo annealer
//   instead of an RK4 integrator.
//
// WHY THIS HEADER IS SHARED (the __host__ __device__ idiom; PATTERNS.md section 2)
//   The ONLY way to verify a GPU result is to run the SAME computation on the CPU
//   and check they agree. That only works if the per-replica annealer -- the RNG,
//   the energy function, every accept/reject decision -- is byte-for-byte the same
//   code on both sides. So all of it lives here, in ONE header, behind the NMR_HD
//   macro that expands to `__host__ __device__` under nvcc and to nothing under
//   the plain host compiler. reference_cpu.cpp includes it via cl.exe; kernels.cu
//   and main.cu include it via nvcc. Same source -> identical trajectories.
//
//   Keep CUDA-only constructs (no `__global__`, no `<<<>>>`) out of this header so
//   the host compiler can include it. (Production GPU MC uses cuRAND; we use a
//   shared, reproducible counter-based RNG specifically so CPU and GPU histories
//   are bit-identical and the demo is deterministic -- see THEORY.md section 5.)
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh.
// READ THIS BEFORE: reference_cpu.h, kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <cstdint>   // uint64_t, fixed-width integers for the RNG
#include <cmath>     // sqrt, exp, log (device math intrinsics under nvcc)

// NMR_HD: present the same inline functions to both compilers (see header note).
#ifdef __CUDACC__
#define NMR_HD __host__ __device__
#else
#define NMR_HD
#endif

// Hard caps so a replica's working state fits in registers/local memory with no
// dynamic allocation inside the kernel. The committed teaching sample stays well
// under these; THEORY.md section 7 discusses how production codes lift the caps.
static const int   NMR_MAX_BEADS     = 64;   // backbone beads (Calpha atoms) per chain
static const int   NMR_MAX_RESTRAINTS = 256; // NOE distance restraints we score

// ---------------------------------------------------------------------------
// RNG: a splitmix64 counter-based stream, identical on host and device.
//   We need a random-number generator that produces the EXACT same bit stream on
//   the CPU and the GPU from the same seed -- otherwise the two annealers would
//   make different moves and could never be compared. splitmix64 is a tiny,
//   high-quality integer hash with no lookup tables and no platform-specific
//   floating point, so it satisfies that requirement. (cuRAND would be faster and
//   higher quality but is NOT bit-identical to a host RNG, which would defeat the
//   verification -- a deliberate teaching trade-off; THEORY.md section 5.)
// ---------------------------------------------------------------------------
struct Rng { uint64_t state; };   // the entire RNG state is one 64-bit word

// One splitmix64 step: advance `x` in place and return a well-mixed 64-bit value.
NMR_HD inline uint64_t splitmix64(uint64_t& x) {
    x += 0x9E3779B97F4A7C15ULL;                       // golden-ratio odd increment
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;      // avalanche the high bits...
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;      // ...then the next ones
    return z ^ (z >> 31);
}

// Seed an independent stream for replica `replica` from a global `base` seed, so
// every trajectory is uncorrelated yet exactly reproducible from (base, replica).
NMR_HD inline Rng rng_seed(uint64_t base, uint64_t replica) {
    Rng r;
    r.state = base ^ (replica * 0x9E3779B97F4A7C15ULL + 0xD1B54A32D192ED03ULL);
    splitmix64(r.state);     // warm up so low-index replicas are not correlated
    return r;
}

// Uniform double in [0,1) built from 53 random bits (identical math host/device).
NMR_HD inline double rng_uniform(Rng& r) {
    uint64_t z = splitmix64(r.state);
    return (z >> 11) * (1.0 / 9007199254740992.0);    // multiply by 2^-53
}

// A standard-normal sample via the Box-Muller transform. Used to perturb a bead
// by a Gaussian random displacement -- the natural, isotropic SA trial move.
// (We deliberately recompute both sin and cos per call rather than cache the
// second deviate; caching would make the host and device call counts differ and
// break bit-exact reproducibility. Simplicity beats the micro-optimisation here.)
NMR_HD inline double rng_normal(Rng& r) {
    const double u1 = 1.0 - rng_uniform(r);           // in (0,1], so log is finite
    const double u2 = rng_uniform(r);
    const double TWO_PI = 6.283185307179586476925286766559;
    return sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2);   // one N(0,1) deviate
}

// ---------------------------------------------------------------------------
// The problem definition: a chain of beads, a list of NOE restraints, and the
// annealing schedule. This is plain data; both the loader and the kernel use it.
// ---------------------------------------------------------------------------

// One NOE distance restraint: atoms i and j should be no farther apart than
// `upper` Angstrom (an UPPER bound -- NOE intensity falls off as 1/r^6, so a
// cross-peak means "these protons are close", not "exactly this far"). We model
// it as a FLAT-BOTTOM penalty: zero cost while r <= upper, harmonic beyond it.
struct Restraint {
    int    i;        // first bead index   (0 <= i < n_beads)
    int    j;        // second bead index  (0 <= j < n_beads)
    double upper;    // upper-bound distance in Angstrom
};

// The full refinement job. Coordinates are flat xyz triples (target/answer is NOT
// stored here -- SA must discover a satisfying structure from the restraints).
struct RefineConfig {
    int    n_beads;                 // number of backbone beads in the chain
    int    n_restraints;            // number of NOE restraints
    double bond_len;                // ideal spacing between bonded neighbours (A)
    double k_bond;                  // bond-restraint force constant (energy/A^2)
    double k_noe;                   // NOE-restraint force constant (energy/A^2)
    int    n_replicas;              // independent SA trajectories in the ensemble
    int    n_steps;                 // Monte-Carlo steps per trajectory
    double T_hot;                   // starting (high) annealing temperature
    double T_cold;                  // final (low) annealing temperature
    double step_sigma;              // std-dev of a trial Gaussian bead move (A)
    uint64_t base_seed;             // global RNG seed (replica r derives from it)
    Restraint restr[NMR_MAX_RESTRAINTS];  // the restraint list (value, not pointer)
};

// What each replica reports back. Kept small and POD so it copies trivially from
// device to host. The DISCRETE field (n_satisfied) is what we verify exactly; the
// continuous energy is verified to a small documented tolerance (THEORY.md s.6).
struct ReplicaResult {
    double final_energy;   // restraint pseudo-energy of the replica's best structure
    int    n_satisfied;    // how many of the n_restraints are satisfied (<= upper+tol)
    int    accepted;       // number of accepted Monte-Carlo moves (a mixing diagnostic)
};

// ---------------------------------------------------------------------------
// Geometry + energy: the per-element physics shared by CPU and GPU.
// ---------------------------------------------------------------------------

// Euclidean distance between beads a and b in the flat coordinate array x[3*nb].
NMR_HD inline double bead_distance(const double* x, int a, int b) {
    const double dx = x[3*a + 0] - x[3*b + 0];
    const double dy = x[3*a + 1] - x[3*b + 1];
    const double dz = x[3*a + 2] - x[3*b + 2];
    return sqrt(dx*dx + dy*dy + dz*dz);
}

// Flat-bottom NOE penalty for one restraint: zero while satisfied, harmonic past
// the upper bound. This is the textbook NOE restraint energy (XPLOR's NOE term):
//     E = 0                       if r <= upper
//     E = 0.5 * k * (r - upper)^2 if r >  upper
NMR_HD inline double noe_energy(double r, double upper, double k) {
    if (r <= upper) return 0.0;
    const double d = r - upper;
    return 0.5 * k * d * d;
}

// Harmonic bond-restraint penalty keeping bonded neighbours near bond_len. NMR
// refinement always layers the known covalent geometry on top of the NOE data so
// the chain stays physically connected: E = 0.5 * k * (r - bond_len)^2.
NMR_HD inline double bond_energy(double r, double bond_len, double k) {
    const double d = r - bond_len;
    return 0.5 * k * d * d;
}

// Total restraint pseudo-energy of a whole structure: every bonded neighbour pair
// (i, i+1) contributes a bond term; every NOE restraint contributes a flat-bottom
// term. This is the function SA minimises. It is the "potential energy surface"
// the annealer walks; THEORY.md section 2 writes it out formally.
NMR_HD inline double total_energy(const double* x, const RefineConfig& c) {
    double E = 0.0;
    // Covalent backbone: consecutive beads are bonded.
    for (int b = 0; b + 1 < c.n_beads; ++b) {
        const double r = bead_distance(x, b, b + 1);
        E += bond_energy(r, c.bond_len, c.k_bond);
    }
    // Experimental NOE restraints.
    for (int q = 0; q < c.n_restraints; ++q) {
        const Restraint& R = c.restr[q];
        const double r = bead_distance(x, R.i, R.j);
        E += noe_energy(r, R.upper, c.k_noe);
    }
    return E;
}

// Count how many NOE restraints a structure SATISFIES (distance within the upper
// bound plus a small validation tolerance, mirroring how NMR software reports
// "restraint violations"). This is an INTEGER metric: it is robust to the last-
// bit floating-point differences between host and device (a 1e-9 A wobble cannot
// flip an integer count when the slack is 0.02 A), so we can verify it EXACTLY.
NMR_HD inline int count_satisfied(const double* x, const RefineConfig& c) {
    const double slack = 0.02;   // 0.02 A validation tolerance (well above round-off)
    int n = 0;
    for (int q = 0; q < c.n_restraints; ++q) {
        const Restraint& R = c.restr[q];
        if (bead_distance(x, R.i, R.j) <= R.upper + slack) ++n;
    }
    return n;
}

// ---------------------------------------------------------------------------
// The annealer itself: run ONE simulated-annealing trajectory to completion.
//   This is the heart of the project. It is shared by the CPU reference and the
//   GPU kernel so a given replica produces the IDENTICAL trajectory on both.
//
//   Algorithm (restrained SA via Metropolis Monte Carlo):
//     1. Build a random initial chain (a self-avoiding-ish random walk seeded by
//        the replica's RNG).
//     2. For each of n_steps:
//          a. Geometric cooling: T = T_hot * (T_cold/T_hot)^(step/n_steps).
//          b. Pick one random bead, propose a Gaussian displacement.
//          c. Compute the energy change dE of moving just that bead (we recompute
//             the full energy here for clarity; THEORY.md section 4 explains the
//             O(1) local-update optimisation we deliberately skip for teaching).
//          d. Metropolis rule: accept if dE <= 0, else accept with probability
//             exp(-dE/T). Reject -> restore the bead.
//          e. Track the lowest-energy structure seen (the "best" the replica found).
//     3. Report the best structure's energy and satisfied-restraint count.
//
//   `x` and `xbest` are caller-provided scratch arrays of length 3*n_beads (kept
//   off the stack so big chains don't blow the per-thread frame). Returns the
//   ReplicaResult for this trajectory.
// ---------------------------------------------------------------------------
NMR_HD inline ReplicaResult anneal_one(const RefineConfig& c, uint64_t replica,
                                       double* x, double* xbest) {
    Rng rng = rng_seed(c.base_seed, replica);

    // ---- 1. Random initial chain: a random walk with steps near bond_len. ----
    // Starting from the origin, each bead is placed a bond-length-ish hop in a
    // random direction from the previous one. Different replicas (different seeds)
    // explore different basins -- the whole reason we run an ensemble.
    x[0] = 0.0; x[1] = 0.0; x[2] = 0.0;
    for (int b = 1; b < c.n_beads; ++b) {
        const double ux = rng_normal(rng), uy = rng_normal(rng), uz = rng_normal(rng);
        const double nrm = sqrt(ux*ux + uy*uy + uz*uz) + 1e-12;  // avoid /0
        x[3*b + 0] = x[3*(b-1) + 0] + c.bond_len * ux / nrm;
        x[3*b + 1] = x[3*(b-1) + 1] + c.bond_len * uy / nrm;
        x[3*b + 2] = x[3*(b-1) + 2] + c.bond_len * uz / nrm;
    }

    double E = total_energy(x, c);          // current energy
    double Ebest = E;                       // best (lowest) energy seen
    for (int k = 0; k < 3 * c.n_beads; ++k) xbest[k] = x[k];   // snapshot the best
    int accepted = 0;

    // ---- 2. Annealing loop ------------------------------------------------
    const double ratio = c.T_cold / c.T_hot;   // geometric-cooling base (< 1)
    for (int step = 0; step < c.n_steps; ++step) {
        // (a) temperature for this step: smooth geometric descent hot -> cold.
        const double frac = (c.n_steps > 1) ? (double)step / (double)(c.n_steps - 1) : 1.0;
        const double T = c.T_hot * pow(ratio, frac);

        // (b) propose: move one random bead by a Gaussian displacement.
        const int b = (int)(rng_uniform(rng) * c.n_beads);   // bead in [0, n_beads)
        const double ox = x[3*b + 0], oy = x[3*b + 1], oz = x[3*b + 2];  // save old
        x[3*b + 0] = ox + c.step_sigma * rng_normal(rng);
        x[3*b + 1] = oy + c.step_sigma * rng_normal(rng);
        x[3*b + 2] = oz + c.step_sigma * rng_normal(rng);

        // (c) energy of the trial structure and the change dE.
        const double Enew = total_energy(x, c);
        const double dE = Enew - E;

        // (d) Metropolis accept/reject. The RNG draw happens UNCONDITIONALLY (for
        //     dE <= 0 we still consume one uniform) so the host and device draw
        //     the exact same number of randoms per step -> identical streams.
        const double xi = rng_uniform(rng);
        bool accept = (dE <= 0.0) || (xi < exp(-dE / T));
        if (accept) {
            E = Enew;
            ++accepted;
            if (E < Ebest) {     // (e) remember the best structure we have seen
                Ebest = E;
                for (int k = 0; k < 3 * c.n_beads; ++k) xbest[k] = x[k];
            }
        } else {
            x[3*b + 0] = ox; x[3*b + 1] = oy; x[3*b + 2] = oz;   // undo the move
        }
    }

    // ---- 3. Report on the best structure found ----------------------------
    ReplicaResult out;
    out.final_energy = Ebest;
    out.n_satisfied  = count_satisfied(xbest, c);
    out.accepted     = accepted;
    return out;
}
