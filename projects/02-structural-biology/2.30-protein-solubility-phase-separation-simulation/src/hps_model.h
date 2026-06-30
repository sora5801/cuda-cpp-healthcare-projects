// ===========================================================================
// src/hps_model.h  --  Shared (host + device) HPS coarse-grained force field
// ---------------------------------------------------------------------------
// Project 2.30 : Protein Solubility & Phase Separation Simulation
//
// WHY THIS HEADER IS SHARED  (the single most important idiom in this project)
//   The CPU reference (reference_cpu.cpp) and the GPU kernel (kernels.cu) must
//   integrate the SAME equations of motion so their trajectories agree to
//   (near) machine precision. The only way to guarantee that is to put the
//   per-pair PHYSICS in ONE place and compile it for both worlds. So every
//   force/energy formula lives here, in `__host__ __device__` inline functions,
//   and is #included by:
//       * reference_cpu.cpp  (plain host C++ compiler: cl.exe / g++)  and
//       * kernels.cu / main.cu  (nvcc, the CUDA compiler).
//   The HPS_HD macro expands to `__host__ __device__` under nvcc and to nothing
//   under the host compiler, so the identical source compiles in both. (See
//   docs/PATTERNS.md §2 "the shared __host__ __device__ core".)
//
//   Keep this header free of CUDA-only constructs (no __global__, no <<<>>>,
//   no cuda*() calls) so the host compiler can include it unchanged.
//
// THE SCIENCE IN ONE PARAGRAPH  (THEORY.md has the full story)
//   Intrinsically disordered proteins (IDPs) such as the low-complexity domains
//   of FUS, TDP-43 and hnRNPA1 can demix from solution into protein-rich liquid
//   droplets -- "biomolecular condensates" (stress granules, P-bodies, nucleolus).
//   This is liquid-liquid phase separation (LLPS). Atomistic MD cannot reach the
//   needed micron / millisecond scales, so the field uses RESIDUE-LEVEL
//   coarse-grained (CG) models: one spherical bead per amino acid. The HPS
//   ("hydrophobicity scale", Dignon-Mittal 2018) / CALVADOS family assigns each
//   residue a "stickiness" lambda in [0,1] and lets beads interact through an
//   Ashbaugh-Hatch modified Lennard-Jones potential, with consecutive residues
//   tied by harmonic bonds. Sticky chains aggregate into a dense droplet; the
//   rest stay dilute -- exactly the two coexisting phases of LLPS.
//
// THE MODEL WE IMPLEMENT (a deliberately REDUCED teaching version; see THEORY)
//   * Beads of equal mass m, in a cubic box of side L with periodic boundaries
//     (minimum-image convention) so a small box mimics bulk solution.
//   * Bonded term: harmonic spring of stiffness k_bond, rest length r0, between
//     consecutive beads of the same chain.
//   * Non-bonded term: the Ashbaugh-Hatch (HPS) potential between every
//     non-bonded pair, parameterized by the pair stickiness lambda_ij and the
//     bead diameter sigma. lambda near 1 => fully attractive LJ (sticky);
//     lambda near 0 => purely repulsive (excluded volume only).
//   * Integrator: velocity-Verlet in the NVE ensemble. We deliberately use a
//     DETERMINISTIC integrator (no random thermostat force) so that, given the
//     same fixed pair-summation order, CPU and GPU produce identical numbers --
//     making verification a near-exact check, not a statistical one. (Production
//     LLPS uses a Langevin thermostat; we note where the noise would enter.)
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cu.  Tour starts in main.cu.
// ===========================================================================
#pragma once

#include <cmath>     // sqrt, pow, fabs  (host <cmath>; nvcc maps to device intrinsics)
#include <cstdint>

// HPS_HD: decorate a function so it compiles for BOTH host and device.
//   Under nvcc, __CUDACC__ is defined and we want __host__ __device__.
//   Under the plain host compiler those keywords do not exist, so expand to "".
#ifdef __CUDACC__
#define HPS_HD __host__ __device__
#else
#define HPS_HD
#endif

// ---------------------------------------------------------------------------
// Simulation parameters. All in reduced (Lennard-Jones) units so the numbers
// stay O(1) and the physics is transparent; THEORY.md maps these to SI. Stored
// as double: the integrator accumulates millions of additions and FP64 keeps
// CPU and GPU agreeing far longer than FP32 would.
// ---------------------------------------------------------------------------
struct SimParams {
    int    n_beads;       // total number of CG beads (residues) in the box
    int    n_chains;      // number of IDP chains (n_beads must be divisible by it)
    int    chain_len;     // beads per chain (= n_beads / n_chains)
    double box;           // cubic box side L (reduced units); periodic in x,y,z
    double sigma;         // bead diameter sigma (LJ length scale)
    double epsilon;       // LJ well depth epsilon (energy scale)
    double r_cut;         // non-bonded cutoff radius (beyond it, force = 0)
    double k_bond;        // harmonic bond stiffness (energy / length^2)
    double r0;            // harmonic bond rest length
    double mass;          // bead mass m (same for every bead)
    double dt;            // integration time step (reduced time units)
    int    n_steps;       // number of velocity-Verlet steps to run
};

// ---------------------------------------------------------------------------
// minimum_image: shortest signed displacement of `d` under periodic boundaries.
//   In a periodic box, bead i may be closest to a *periodic image* of bead j.
//   The minimum-image convention folds any coordinate difference into
//   (-L/2, +L/2]. Using the rounded multiple of L is branch-free and identical
//   on host and device (round() ties-to-even on both), which protects
//   determinism. Units: same as the input (reduced length).
// ---------------------------------------------------------------------------
HPS_HD inline double minimum_image(double d, double box) {
    // round(d/box) is how many whole boxes to subtract to land in [-L/2, L/2].
    return d - box * std::nearbyint(d / box);
}

// ---------------------------------------------------------------------------
// ah_pair_force_over_r: the Ashbaugh-Hatch (HPS) non-bonded force.
//   Returns the scalar (F/r) so the caller multiplies by the displacement
//   vector to get the force components -- this avoids an extra sqrt per axis.
//   Also returns the pair POTENTIAL ENERGY via *u_out (for the energy report).
//
//   The Ashbaugh-Hatch potential modulates the standard 12-6 Lennard-Jones
//   U_LJ(r) = 4*eps*[(sigma/r)^12 - (sigma/r)^6] by a stickiness lambda:
//       r <= r_min (= 2^(1/6) sigma):   U = U_LJ(r) + (1 - lambda)*eps      [repulsive core, always present]
//       r >  r_min:                     U = lambda * U_LJ(r)                [attractive tail, scaled by lambda]
//   So lambda=1 recovers full LJ (maximally sticky) and lambda=0 leaves only
//   the shifted repulsive core (pure excluded volume). This single knob is how
//   the model encodes "hydrophobicity": sticky residues phase-separate, polar
//   ones do not. (Dignon, Zheng, Kim, Best, Mittal, PLoS Comput Biol 2018.)
//
//   Parameters:
//     r2       : squared distance between the two beads (units length^2). We
//                take r^2 (not r) to skip a sqrt where possible.
//     lambda   : pair stickiness in [0,1] (we use the arithmetic mean of the
//                two residues' lambda; see lambda_mean()).
//     sigma    : LJ length scale (bead diameter).
//     eps      : LJ energy scale.
//     r_cut    : cutoff; beyond it force and energy are zero (returns 0).
//     u_out    : out-param, the pair potential energy contribution.
//   Returns F/r (so force vector = (F/r) * displacement). Sign convention:
//   positive F/r means repulsion (pushes i away from j).
// ---------------------------------------------------------------------------
HPS_HD inline double ah_pair_force_over_r(double r2, double lambda,
                                          double sigma, double eps,
                                          double r_cut, double* u_out) {
    // Beyond the cutoff the interaction is truncated to exactly zero. This is
    // what makes an O(N) neighbour scheme possible in production; here (all
    // pairs) it still defines the model and keeps the tail integrable.
    if (r2 >= r_cut * r_cut) { *u_out = 0.0; return 0.0; }

    // Work in powers of (sigma^2 / r^2) to avoid a sqrt: s2 = (sigma/r)^2,
    // then s6 = (sigma/r)^6, s12 = (sigma/r)^12.
    const double s2  = (sigma * sigma) / r2;
    const double s6  = s2 * s2 * s2;
    const double s12 = s6 * s6;

    // Plain 12-6 Lennard-Jones energy and its (F/r).
    //   U_LJ      = 4 eps (s12 - s6)
    //   dU/dr     = -24 eps (2 s12 - s6) / r   =>   F/r = -dU/dr / r
    //   F/r       = 24 eps (2 s12 - s6) / r^2
    const double u_lj      = 4.0 * eps * (s12 - s6);
    const double f_over_r_lj = 24.0 * eps * (2.0 * s12 - s6) / r2;

    // r_min = 2^(1/6) sigma is the LJ minimum; r2 < r_min^2 is the repulsive side.
    // r_min^2 = 2^(1/3) sigma^2.
    const double rmin2 = 1.2599210498948732 * sigma * sigma;  // 2^(1/3) * sigma^2

    double u, f_over_r;
    if (r2 <= rmin2) {
        // Repulsive core region: the FULL LJ force, and the energy lifted by
        // (1 - lambda) eps so the curve is continuous at r_min. The repulsion is
        // present at full strength regardless of stickiness (excluded volume).
        u        = u_lj + (1.0 - lambda) * eps;
        f_over_r = f_over_r_lj;
    } else {
        // Attractive tail region: the whole LJ interaction is scaled by lambda.
        u        = lambda * u_lj;
        f_over_r = lambda * f_over_r_lj;
    }
    *u_out = u;
    return f_over_r;
}

// ---------------------------------------------------------------------------
// lambda_mean: combine two single-residue stickiness values into a pair value.
//   The HPS model uses the arithmetic mean lambda_ij = (lambda_i + lambda_j)/2.
//   (CALVADOS uses the same mixing rule.) Kept as its own function so the CPU
//   and GPU paths cannot drift apart in how they mix.
// ---------------------------------------------------------------------------
HPS_HD inline double lambda_mean(double li, double lj) {
    return 0.5 * (li + lj);
}

// ---------------------------------------------------------------------------
// bond_force_over_r: harmonic backbone bond between consecutive beads.
//   U_bond(r) = 0.5 k (r - r0)^2  =>  F = -k (r - r0) along the bond.
//   We return F/r (so force vector = (F/r) * displacement) and the energy.
//   This is the ONLY place r itself (not r^2) is needed, so we take r directly.
//   Sign: if r > r0 the bond is stretched and pulls the beads together
//   (F/r < 0 means attraction along the +displacement direction).
// ---------------------------------------------------------------------------
HPS_HD inline double bond_force_over_r(double r, double k_bond, double r0,
                                       double* u_out) {
    const double dr = r - r0;            // extension (can be negative = compressed)
    *u_out = 0.5 * k_bond * dr * dr;     // harmonic energy
    // F = -k dr (restoring). F/r = -k dr / r. Guard r==0 (coincident beads).
    return (r > 1e-12) ? (-k_bond * dr / r) : 0.0;
}

// ---------------------------------------------------------------------------
// bead_force: the WHOLE force on one bead `i`, summed over every other bead.
//   THIS is the function that makes CPU and GPU agree: the serial reference and
//   the GPU kernel both call it with i ranging over all beads, summing j in the
//   IDENTICAL fixed order j = 0..N-1 (skipping j == i). Because the order and
//   the FP64 arithmetic are the same, the two force vectors are bit-identical,
//   so the velocity-Verlet trajectories never diverge (see THEORY.md "verify").
//
//   Strategy ("gather"): thread i OWNS output bead i and READS all j -- there
//   are no concurrent writes, hence NO atomics and no race. The cost is the
//   classic all-pairs O(N^2): every bead looks at every other. Production HPS
//   codes replace this with a cutoff cell/neighbour list (O(N)); we keep
//   all-pairs because it is the clearest correct version to learn from, and the
//   cutoff still zeroes far interactions inside ah_pair_force_over_r().
//
//   Bonded pairs (|i-j| == 1 within the same chain) get the harmonic bond
//   INSTEAD of the non-bonded HPS term -- the standard MD exclusion so a bond
//   does not double as a steric clash.
//
//   Inputs (all device/host pointers to length-N arrays):
//     i        : index of the bead whose force we compute
//     N        : number of beads
//     x,y,z    : positions
//     lam      : per-bead stickiness
//     chain    : per-bead chain id (bonded neighbours share a chain id)
//     P        : parameters (box, sigma, eps, r_cut, k_bond, r0)
//   Outputs:
//     fx,fy,fz : the three force components on bead i (overwritten)
//     u_half   : HALF the potential energy bead i participates in (so summing
//                u_half over all i counts each pair once -> total PE).
// ---------------------------------------------------------------------------
HPS_HD inline void bead_force(int i, int N,
                              const double* x, const double* y, const double* z,
                              const double* lam, const int* chain,
                              const SimParams& P,
                              double* fx, double* fy, double* fz, double* u_half) {
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;  // running force accumulators
    double U  = 0.0;                       // running energy (pairs i participates in)
    const double xi = x[i], yi = y[i], zi = z[i], li = lam[i];
    const int    ci = chain[i];

    // Fixed-order scan over all other beads. The order is what guarantees
    // CPU/GPU determinism, so do NOT reorder this loop.
    for (int j = 0; j < N; ++j) {
        if (j == i) continue;             // a bead exerts no force on itself

        // Minimum-image displacement from j to i (periodic boundaries).
        double dx = minimum_image(xi - x[j], P.box);
        double dy = minimum_image(yi - y[j], P.box);
        double dz = minimum_image(zi - z[j], P.box);
        double r2 = dx * dx + dy * dy + dz * dz;

        // Are i and j bonded? They are if they are adjacent residues of the
        // SAME chain (|i-j| == 1 and same chain id). Within a chain beads are
        // stored consecutively, so adjacency in index == adjacency in sequence.
        const bool bonded = (chain[j] == ci) && (j == i - 1 || j == i + 1);

        double u_pair = 0.0, fr = 0.0;    // pair energy and F/r
        if (bonded) {
            double r = std::sqrt(r2);
            fr = bond_force_over_r(r, P.k_bond, P.r0, &u_pair);
        } else {
            double lij = lambda_mean(li, lam[j]);
            fr = ah_pair_force_over_r(r2, lij, P.sigma, P.epsilon, P.r_cut, &u_pair);
        }

        // Accumulate the force vector: F_i += (F/r) * (r_i - r_j).
        Fx += fr * dx;
        Fy += fr * dy;
        Fz += fr * dz;
        // Each pair's energy is shared by i and j; add half here so that summing
        // U over all beads counts every pair exactly once.
        U  += 0.5 * u_pair;
    }
    *fx = Fx; *fy = Fy; *fz = Fz;
    *u_half = U;
}
