// ===========================================================================
// src/condensate.h  --  Shared (host + device) coarse-grained condensate model
// ---------------------------------------------------------------------------
// Project 2.34 : Biophysical Simulation of Biomolecular Condensates
//                (Active Learning Loop)  --  REDUCED-SCOPE TEACHING VERSION
//
// WHAT THIS PROJECT COMPUTES (and why it is a teaching reduction)
//   Intrinsically disordered proteins (IDPs) such as FUS, TDP-43 and hnRNPA1
//   can undergo liquid-liquid phase separation (LLPS): many copies condense
//   into a dense, droplet-like "biomolecular condensate" while the surrounding
//   solution stays dilute. Two measurable properties of such a droplet are its
//   COMPACTNESS (how tightly the chain folds, here the radius of gyration Rg)
//   and the INTERNAL MOBILITY of the molecules (the self-diffusion coefficient
//   D, which controls whether a condensate is liquid-like or gel-like). Both
//   depend on ONE sequence-level knob in coarse-grained IDP force fields like
//   CALVADOS: the mean "stickiness" lambda of the residues (cohesive strength).
//
//   The frontier project the catalog describes is an ACTIVE-LEARNING LOOP:
//     (1) GPU coarse-grained MD simulates many candidate sequences,
//     (2) a GNN surrogate learns property(sequence),
//     (3) Bayesian optimization proposes the next sequence to simulate.
//   A full GNN + BoTorch + CALVADOS loop is research-grade and not a single
//   teachable CUDA kernel, so (per CLAUDE.md section 13) we ship a faithful
//   REDUCED-SCOPE version that keeps the load-bearing GPU compute pattern:
//
//     * Each candidate sequence is reduced to its single stickiness lambda.
//     * For each lambda we run an INDEPENDENT coarse-grained Brownian-dynamics
//       (overdamped Langevin) trajectory of a short bead-spring chain in an
//       effective cohesive potential -> this is the "GPU CG-MD per replica".
//     * From the trajectory we measure D (Einstein relation on the mean-square
//       displacement, MSD) and Rg -> the "GPU MSD diffusion-coefficient" step.
//     * A cheap deterministic surrogate + acquisition function then proposes
//       the next lambda to try -> the "Bayesian optimization proposes a new
//       sequence" step, made concrete and reproducible.
//
//   The ensemble-over-replicas mapping (one GPU thread integrates one full
//   trajectory) is exactly the flagship 9.02 / 13.02 pattern (PATTERNS.md
//   section 1, "the same ODE for many parameter sets"). See ../THEORY.md.
//
// WHY THE PHYSICS LIVES IN THIS HEADER
//   The per-replica integrator below is marked CND_HD == __host__ __device__,
//   so the CPU reference (reference_cpu.cpp, host compiler) and the GPU kernel
//   (kernels.cu, nvcc) call the SAME code. Identical math -> their results
//   match to a small, documented floating-point tolerance (PATTERNS.md s4).
//   Keep CUDA-only constructs (__global__, <<<>>>) OUT of this header so the
//   host compiler can include it.
//
//   The thermal noise uses a COUNTER-BASED hash RNG (below): the random force
//   on (replica r, step s, axis a) is a pure function of those integer indices
//   plus a seed -- no per-thread RNG state, no call-order dependence. That is
//   what makes the GPU run bit-reproducible AND identical to the CPU: thread
//   17 draws exactly the numbers the CPU's iteration 17 draws.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh.
// READ THIS BEFORE: reference_cpu.h, kernels.cuh, main.cu.
// ===========================================================================
#pragma once

// CND_HD decorates a function so it compiles for BOTH the CPU (host) and the
// GPU (device). Under nvcc, __CUDACC__ is defined and the decorators exist;
// under the plain host compiler they must vanish.
#ifdef __CUDACC__
#define CND_HD __host__ __device__
#else
#define CND_HD
#endif

#include <cstdint>   // std::uint32_t / std::uint64_t for the counter RNG
#include <cmath>     // std::sqrt, std::log, std::cos, std::sin

// ---------------------------------------------------------------------------
// Physical constants of the reduced model (one tidy struct so the CPU loader,
// the kernel and the analysis all agree on the SAME numbers). Units are
// "reduced MD units" (sigma for length, tau for time, kT for energy) -- the
// standard non-dimensionalization used by coarse-grained IDP models so the
// integrator is numerically well-scaled. THEORY section "The math" defines them.
// ---------------------------------------------------------------------------
struct CondensateModel {
    int    n_beads = 0;     // beads per chain (a short IDP, e.g. 12 residues)
    int    steps   = 0;     // Brownian-dynamics steps per trajectory
    double dt      = 0.0;   // integration timestep (reduced time units)
    double kT      = 0.0;   // thermal energy (sets the noise amplitude)
    double gamma   = 0.0;   // friction coefficient (overdamped drag)
    double k_bond  = 0.0;   // harmonic bond stiffness (chain connectivity)
    double r0      = 0.0;   // bond rest length (reduced length units)
    int    eq_steps = 0;    // equilibration steps discarded before measuring MSD
    int    lag      = 0;    // MSD time-lag (in steps) used to estimate mobility D
    std::uint32_t seed = 0; // global RNG seed (reproducibility)
};

// ---------------------------------------------------------------------------
// hash_u32: a small integer bit-mixer (SplitMix32-style finalizer). Given a
// 32-bit key it returns a well-scrambled 32-bit value. This is the heart of
// the COUNTER-BASED RNG: instead of evolving RNG state step by step, we HASH a
// counter built from (replica, step, axis, seed). Same indices -> same bits on
// CPU and GPU, in any order -> exact reproducibility (THEORY "Numerical").
//   x  : the integer key to scramble
//   ret: a uniformly-mixed 32-bit unsigned value
// ---------------------------------------------------------------------------
CND_HD inline std::uint32_t hash_u32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;   // odd multiplier -> full-period bijective mixing
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

// ---------------------------------------------------------------------------
// uniform01: map a hashed integer to a double in the open interval (0,1).
//   We divide by 2^32 and nudge off the endpoints so a later std::log() never
//   sees exactly 0 (which would be -infinity). 24 bits of mantissa would be
//   enough for floats; we keep the full 32 bits for the double path.
// ---------------------------------------------------------------------------
CND_HD inline double uniform01(std::uint32_t h) {
    // (h + 0.5) / 2^32  lands strictly inside (0,1).
    return (static_cast<double>(h) + 0.5) * (1.0 / 4294967296.0);
}

// ---------------------------------------------------------------------------
// gaussian_noise: one standard-normal sample N(0,1) for a specific
// (replica, step, bead, axis) via the Box-Muller transform on two independent
// counter hashes. Because the inputs are integer coordinates, the SAME draw is
// reproduced identically on CPU and GPU regardless of thread scheduling.
//
//   replica : ensemble member index (which candidate lambda)
//   step    : current Brownian-dynamics step
//   bead    : which bead of the chain
//   axis    : 0,1,2 for x,y,z
//   seed    : the model's global seed
//   ret     : a deterministic N(0,1) thermal-force sample
//
// Box-Muller: from two uniforms u1,u2 in (0,1),
//   z = sqrt(-2 ln u1) * cos(2 pi u2)  is N(0,1).
// We build two DISTINCT counters (offset the axis field) so u1 != u2.
// ---------------------------------------------------------------------------
CND_HD inline double gaussian_noise(int replica, int step, int bead, int axis,
                                    std::uint32_t seed) {
    // Pack the four indices into two different 32-bit counters. The bit shifts
    // give each field its own range so distinct (r,s,b,a) tuples almost never
    // collide for the sizes used here (replicas<2^10, steps<2^16, beads<2^6).
    std::uint32_t base = seed
                       + static_cast<std::uint32_t>(replica) * 2654435761U   // Knuth's multiplicative constant
                       + static_cast<std::uint32_t>(step)    * 40503U
                       + static_cast<std::uint32_t>(bead)    * 2246822519U
                       + static_cast<std::uint32_t>(axis)    * 3266489917U;
    const double u1 = uniform01(hash_u32(base));
    const double u2 = uniform01(hash_u32(base ^ 0x9e3779b9U));  // golden-ratio salt -> 2nd stream
    const double mag = std::sqrt(-2.0 * std::log(u1));
    const double ang = 6.283185307179586 * u2;                 // 2*pi*u2
    return mag * std::cos(ang);
}

// ---------------------------------------------------------------------------
// cohesive_lambda: map an ensemble member index to its candidate "stickiness"
// lambda on a uniform grid [lambda_lo, lambda_hi]. Each member is one candidate
// IDP sequence reduced to its mean residue stickiness (the CALVADOS knob).
//   m         : member index in [0, n_members)
//   n_members : total ensemble size
//   lo, hi    : stickiness range to scan
//   ret       : lambda for member m (dimensionless cohesive strength)
// This is the ensemble "parameter sweep": independent trajectories, one per m.
// ---------------------------------------------------------------------------
CND_HD inline double cohesive_lambda(int m, int n_members, double lo, double hi) {
    if (n_members <= 1) return lo;
    return lo + (hi - lo) * (static_cast<double>(m) / (n_members - 1));
}

// ---------------------------------------------------------------------------
// Per-replica result: the two condensate-property observables we measure plus a
// reproducible fingerprint of the final configuration (used in the demo output).
// ---------------------------------------------------------------------------
struct ReplicaResult {
    double lambda;       // the stickiness this replica simulated
    double diffusion;    // D from the Einstein relation (length^2 / time)
    double rg;           // time-averaged radius of gyration (length)
    double msd_final;    // MSD at the final lag (length^2) -- raw measurement
};

// ---------------------------------------------------------------------------
// cohesive_force_1d: the effective intra-condensate cohesive force on ONE
// coordinate of one bead. We model the dense environment as a soft harmonic
// well toward the chain's centre of mass whose stiffness GROWS with lambda:
// stickier sequences feel a deeper well -> more compact, slower diffusion. This
// is a deliberately simple stand-in for the many-body Ashbaugh-Hatch potential
// of CALVADOS (explained in THEORY "Where this sits in the real world").
//   coord     : this bead's position on one axis
//   com       : centre-of-mass position on the same axis
//   lambda    : cohesive strength (dimensionless)
//   k_cohese  : base cohesive stiffness scale
//   ret       : the restoring force component (= -dU/dcoord)
// ---------------------------------------------------------------------------
CND_HD inline double cohesive_force_1d(double coord, double com,
                                       double lambda, double k_cohese) {
    // U = 0.5 * (k_cohese*lambda) * (coord-com)^2  ->  F = -(k_cohese*lambda)*(coord-com)
    return -(k_cohese * lambda) * (coord - com);
}

// ---------------------------------------------------------------------------
// integrate_replica: run ONE coarse-grained Brownian-dynamics trajectory for a
// chain of n_beads beads in 3-D and return its measured (D, Rg). This is the
// single most important function in the project: the CPU reference loops it
// over members, and the GPU kernel runs one thread per member calling it.
//
// PHYSICS (overdamped Langevin / Brownian dynamics):
//   gamma * dx/dt = F(x) + sqrt(2 gamma kT) * xi(t)
//   discretized (Euler-Maruyama):
//     x_{t+1} = x_t + (dt/gamma) F + sqrt(2 kT dt / gamma) * N(0,1)
//   F on each bead = harmonic BONDS to its chain neighbours (connectivity)
//     + a cohesive pull toward the centre of mass (condensate environment).
//   The thermal kick uses the counter-based gaussian_noise() above, so the
//   whole trajectory is a deterministic function of (member, seed).
//
// MEASUREMENT (what makes D depend on lambda):
//   The CENTRE OF MASS of the whole chain diffuses FREELY no matter how sticky
//   the chain is, because the cohesive force is internal (it sums to zero over
//   the chain). So COM diffusion would NOT discriminate sequences. What lambda
//   actually controls is INTERNAL mobility: how far a bead can wander RELATIVE
//   TO THE COM before the cohesive well pulls it back. We therefore measure the
//   internal mean-square displacement at a fixed time-lag tau = lag*dt:
//
//     MSD_int(tau) = < | r_i(t+tau) - r_i(t) |^2 >   (r_i in the COM frame),
//
//   averaged over all beads i AND all time origins t in the production window
//   (a ring buffer holds the last lag+1 COM-frame snapshots so we can subtract).
//   An "apparent" diffusion coefficient follows from the 3-D Einstein relation
//   D = MSD_int(tau) / (6 tau). For a confined (sticky) chain MSD_int saturates
//   to a smaller plateau -> smaller D. Rg (radius of gyration) is averaged over
//   the production steps as the COMPACTNESS observable; it falls monotonically
//   with lambda. THEORY "The math" derives the plateau / confinement picture.
//
// PARAMETERS
//   M         : the model constants (timestep, kT, stiffnesses, lag, seed, ...)
//   replica   : this member's index (selects its lambda AND its noise stream)
//   lambda    : cohesive strength for this replica
//   k_cohese  : base cohesive stiffness scale (shared across replicas)
//   ret       : a filled ReplicaResult (lambda, D, Rg, MSD at the lag)
//
// COMPLEXITY: O(steps * n_beads) per replica; O(n_members * steps * n_beads)
// for the whole ensemble. Time is sequential WITHIN a replica (each step needs
// the previous), independent BETWEEN replicas -> the GPU parallelizes members.
//
// We cap n_beads and the lag with fixed-size local arrays so the whole state
// (positions + the lag ring buffer) lives in registers/local memory; no dynamic
// allocation inside the kernel. The ring buffer is the largest consumer:
// (CND_MAX_LAG+1) * CND_MAX_BEADS * 3 doubles per thread.
// ---------------------------------------------------------------------------
constexpr int CND_MAX_BEADS = 16;   // upper bound on chain length (local arrays)
constexpr int CND_MAX_LAG   = 24;   // upper bound on the MSD lag (ring-buffer depth)

CND_HD inline ReplicaResult integrate_replica(const CondensateModel& M,
                                              int replica, double lambda,
                                              double k_cohese) {
    const int n   = (M.n_beads < CND_MAX_BEADS) ? M.n_beads : CND_MAX_BEADS;
    const int lag = (M.lag    < CND_MAX_LAG)    ? M.lag     : CND_MAX_LAG;
    // Pre-compute the two Euler-Maruyama coefficients once (they never change):
    const double drift_coef = M.dt / M.gamma;                         // multiplies the force
    const double noise_coef = std::sqrt(2.0 * M.kT * M.dt / M.gamma); // multiplies N(0,1)

    // Bead coordinates in 3-D, stored in fixed local arrays (register/local mem,
    // not global) so each thread is fully self-contained. Initial condition: a
    // straight chain along x at spacing r0 (deterministic, no RNG) so every
    // replica starts identically and differences come only from lambda + noise.
    double x[CND_MAX_BEADS], y[CND_MAX_BEADS], z[CND_MAX_BEADS];
    for (int i = 0; i < n; ++i) { x[i] = i * M.r0; y[i] = 0.0; z[i] = 0.0; }

    // Scratch arrays for the NEXT configuration. We update all beads
    // SIMULTANEOUSLY (read old positions, write new) so the result is
    // independent of the bead iteration order -- the standard, order-stable
    // Euler-Maruyama update. An in-place update would let bead i+1 see bead i's
    // already-advanced position, which is a different (still valid but messier)
    // integrator; we avoid it so the math is unambiguous.
    double nxp[CND_MAX_BEADS], nyp[CND_MAX_BEADS], nzp[CND_MAX_BEADS];

    // Circular buffer of the last (lag+1) COM-frame snapshots, so at each
    // production step we can subtract the snapshot from exactly `lag` steps ago
    // and accumulate the internal MSD at that time-lag. Layout is
    // rel[slot][bead][axis]; `slot = (production_index) % (lag+1)` cycles the
    // (lag+1) slots. This is the largest per-thread allocation (size note above).
    double rel[CND_MAX_LAG + 1][CND_MAX_BEADS][3];
    int    prod_idx = 0;        // count of production snapshots stored so far

    double rg_sum   = 0.0;      // accumulates Rg over production steps (for the mean)
    int    rg_count = 0;        // number of Rg samples summed
    double msd_sum  = 0.0;      // accumulates internal MSD at the lag (sum over origins)
    int    msd_count = 0;       // number of (origin) MSD samples summed

    // ---- the Brownian-dynamics time loop (sequential in s) ----------------
    for (int s = 0; s < M.steps; ++s) {
        // (a) centre of mass of the current configuration (needed by the
        //     cohesive force and by both observables).
        double cx = 0.0, cy = 0.0, cz = 0.0;
        for (int i = 0; i < n; ++i) { cx += x[i]; cy += y[i]; cz += z[i]; }
        cx /= n; cy /= n; cz /= n;

        // (b) compute the NEW position of every bead. Force = harmonic bonds to
        //     chain neighbours + cohesive pull toward the COM, plus the thermal
        //     kick. Each bond is a PER-AXIS harmonic spring: along x the rest
        //     offset between consecutive beads is r0 (the initial layout is a
        //     straight line on x), along y/z it is 0. Linear, unconditionally
        //     stable for the timestep used -- the right tradeoff for teaching
        //     code (THEORY "Numerical considerations").
        for (int i = 0; i < n; ++i) {
            double fx = 0.0, fy = 0.0, fz = 0.0;
            if (i > 0) {       // harmonic bond to the previous bead (i-1)
                fx += -M.k_bond * ( (x[i]-x[i-1]) - M.r0 );
                fy += -M.k_bond * (  y[i]-y[i-1]          );
                fz += -M.k_bond * (  z[i]-z[i-1]          );
            }
            if (i < n-1) {     // harmonic bond to the next bead (i+1)
                fx += -M.k_bond * ( (x[i]-x[i+1]) + M.r0 );
                fy += -M.k_bond * (  y[i]-y[i+1]          );
                fz += -M.k_bond * (  z[i]-z[i+1]          );
            }
            // Cohesive pull toward the COM, stiffer for stickier (higher-lambda)
            // sequences -> compaction + slowdown grow with lambda.
            fx += cohesive_force_1d(x[i], cx, lambda, k_cohese);
            fy += cohesive_force_1d(y[i], cy, lambda, k_cohese);
            fz += cohesive_force_1d(z[i], cz, lambda, k_cohese);

            // Thermal kicks: independent N(0,1) per axis from the counter RNG.
            const double gx = gaussian_noise(replica, s, i, 0, M.seed);
            const double gy = gaussian_noise(replica, s, i, 1, M.seed);
            const double gz = gaussian_noise(replica, s, i, 2, M.seed);

            // Euler-Maruyama update into the scratch arrays (simultaneous).
            nxp[i] = x[i] + drift_coef * fx + noise_coef * gx;
            nyp[i] = y[i] + drift_coef * fy + noise_coef * gy;
            nzp[i] = z[i] + drift_coef * fz + noise_coef * gz;
        }
        // Commit the simultaneous update: new positions become current.
        for (int i = 0; i < n; ++i) { x[i] = nxp[i]; y[i] = nyp[i]; z[i] = nzp[i]; }

        // (c) production-phase measurements (after equilibration discards the
        //     startup transient).
        if (s >= M.eq_steps) {
            // Fresh COM of the just-updated configuration.
            double ncx = 0.0, ncy = 0.0, ncz = 0.0;
            for (int i = 0; i < n; ++i) { ncx += x[i]; ncy += y[i]; ncz += z[i]; }
            ncx /= n; ncy /= n; ncz /= n;

            // Radius of gyration: RMS distance of beads from the COM (compactness).
            double rg2 = 0.0;
            for (int i = 0; i < n; ++i) {
                const double dx = x[i]-ncx, dy = y[i]-ncy, dz = z[i]-ncz;
                rg2 += dx*dx + dy*dy + dz*dz;
            }
            rg_sum += std::sqrt(rg2 / n);
            ++rg_count;

            // Internal MSD at the lag. Once we have stored at least `lag` earlier
            // snapshots, the one from exactly `lag` production-steps ago lives at
            // slot (prod_idx - lag) mod (lag+1). Subtract it from the current
            // COM-frame positions to get the per-bead displacement over `lag`
            // steps, average over beads, and add to the running sum over origins.
            if (prod_idx >= lag) {
                const int old_slot = (prod_idx - lag) % (lag + 1);
                double d = 0.0;
                for (int i = 0; i < n; ++i) {
                    const double dx = (x[i]-ncx) - rel[old_slot][i][0];
                    const double dy = (y[i]-ncy) - rel[old_slot][i][1];
                    const double dz = (z[i]-ncz) - rel[old_slot][i][2];
                    d += dx*dx + dy*dy + dz*dz;
                }
                msd_sum += d / n;     // average over beads; sum over origins
                ++msd_count;
            }

            // Store THIS snapshot in its circular slot for future lags to read.
            const int slot = prod_idx % (lag + 1);
            for (int i = 0; i < n; ++i) {
                rel[slot][i][0] = x[i]-ncx;
                rel[slot][i][1] = y[i]-ncy;
                rel[slot][i][2] = z[i]-ncz;
            }
            ++prod_idx;
        }
    }

    // ---- turn the raw measurements into the two reported properties -------
    ReplicaResult out;
    out.lambda    = lambda;
    out.rg        = (rg_count  > 0) ? rg_sum  / rg_count  : 0.0;
    const double msd_lag = (msd_count > 0) ? msd_sum / msd_count : 0.0;
    out.msd_final = msd_lag;
    // 3-D Einstein relation at the chosen lag: MSD_int(tau) = 6 D tau.
    const double tau = lag * M.dt;
    out.diffusion = (tau > 0.0) ? msd_lag / (6.0 * tau) : 0.0;
    return out;
}
