// ===========================================================================
// src/smd_core.h  --  Shared (host + device) SMD physics: the ONE true formula
// ---------------------------------------------------------------------------
// Project 1.26 : Steered Molecular Dynamics (SMD)  (see ../THEORY.md for the
// full science -> math -> algorithm -> GPU-mapping treatment).
//
// WHY THIS HEADER IS SHARED (the HD-macro idiom, PATTERNS.md §2)
//   The whole verification strategy is: run the IDENTICAL stochastic trajectory
//   on the CPU reference and on the GPU, and require their per-trajectory work
//   values to agree EXACTLY. That only works if both sides use the same random
//   number generator and the same integrator. So both live here, in ONE header,
//   included by reference_cpu.cpp (host compiler) AND by kernels.cu / main.cu
//   (nvcc). SMD_HD expands to `__host__ __device__` under nvcc and to nothing
//   under the plain host compiler, so the same inline functions compile in both
//   worlds. Keep CUDA-only constructs (`__global__`, `<<<>>>`) OUT of this file
//   so cl.exe / g++ can include it.
//
// THE TEACHING MODEL (a deliberately reduced 1-D version of full-atom SMD)
//   Production SMD pulls a whole protein-ligand system in a 3-D MD force field
//   (NAMD, GROMACS, OpenMM). The physics that MATTERS for the teaching goal --
//   "pull along a coordinate, accumulate non-equilibrium work, recover a free
//   energy with Jarzynski's equality" -- survives in a 1-D caricature:
//
//     * A single reaction coordinate  xi  (e.g. ligand distance from a pocket),
//       in nanometres. Its surroundings are modelled by OVERDAMPED LANGEVIN
//       (Brownian) dynamics -- inertia is negligible on the diffusive time
//       scale, so we evolve position directly under force + friction + thermal
//       noise. This is the Brownian-dynamics limit of full MD.
//
//     * A fixed POTENTIAL OF MEAN FORCE  U(xi)  -- the free-energy landscape the
//       ligand actually feels, with a bound well and an unbound plateau. Here we
//       use a smooth double-well-like profile whose exact ΔG between the two end
//       states is KNOWN ANALYTICALLY, so we can check the Jarzynski estimate
//       against ground truth (PATTERNS.md §6: embed a known answer).
//
//     * A SMD HARMONIC SPRING of stiffness  k  whose attachment point (the
//       "dummy atom") moves at constant velocity  v :  center(t) = xi0 + v*t.
//       The spring force on the coordinate is  F_spring = k*(center - xi).
//       This is CONSTANT-VELOCITY SMD (the harmonic-guide variant); the
//       constant-force variant is discussed in THEORY.md.
//
//     * EXTERNAL WORK done by the moving spring is accumulated along the pull:
//       dW = F_spring * d(center) = F_spring * v * dt   (the dummy atom moves by
//       v*dt each step). W is a NON-EQUILIBRIUM work -- it depends on the random
//       trajectory, so different thermal histories give different W. That spread
//       is exactly what Jarzynski's equality consumes.
//
//   Many INDEPENDENT trajectories (different random seeds) -> a distribution of
//   work values W_i. Jarzynski's equality (1997) states, exactly:
//       <exp(-beta*W)> = exp(-beta*ΔG)      =>   ΔG = -(1/beta) ln <exp(-beta*W)>
//   where beta = 1/(kB*T) and the average is over the work distribution. More
//   trajectories -> a better-converged estimate. The GPU runs them in parallel.
// ===========================================================================
#pragma once

#include <cstdint>
#include <cmath>

#ifdef __CUDACC__
#define SMD_HD __host__ __device__
#else
#define SMD_HD
#endif

// ---------------------------------------------------------------------------
// Physical / simulation parameters (loaded from the data file; see data/README).
//   Units are chosen to make the numbers human-scale for a ligand-unbinding toy:
//   nanometres for length, picoseconds for time, kJ/mol for energy. They are
//   internally consistent -- the exact magnitudes are pedagogical, not a claim
//   about any real molecule (this is SYNTHETIC; CLAUDE.md §8).
// ---------------------------------------------------------------------------
struct SmdParams {
    double xi0;        // start of the pull (bound state), nm
    double xi_end;     // end of the pull (unbound state), nm
    int    n_traj;     // number of independent SMD trajectories (the ensemble)
    int    steps;      // integrator steps per trajectory
    double dt;         // timestep, ps
    double k_spring;   // SMD spring stiffness, kJ/mol/nm^2
    double v_pull;     // pulling velocity of the dummy atom, nm/ps
    double gamma;      // Langevin friction coefficient, (kJ/mol) ps / nm^2
    double kT;         // thermal energy kB*T, kJ/mol  (beta = 1/kT)
    // Double-well PMF U(xi) = A*(xi-xa)^2*(xi-xb)^2 / (xb-xa)^2 + slope*xi.
    // (A quartic with minima near xa,xb; `slope` tilts it so the two end states
    //  have a known free-energy difference -- see pmf_energy/pmf_force below.)
    double pmf_A;      // barrier height scale, kJ/mol
    double pmf_xa;     // bound-well centre, nm
    double pmf_xb;     // unbound-well centre, nm
    double pmf_slope;  // linear tilt, kJ/mol/nm  (sets the true ΔG end-to-end)
    uint64_t seed;     // base RNG seed (per-trajectory streams derive from it)
};

// ---------------------------------------------------------------------------
// RNG: splitmix64 counter-based stream -- identical bit-for-bit on host & device
//   (the same idiom as flagship 5.01). We do NOT use cuRAND here precisely
//   because we need the CPU reference and the GPU kernel to draw the SAME random
//   numbers so their per-trajectory work agrees EXACTLY (PATTERNS.md §4 "exact").
// ---------------------------------------------------------------------------
struct Rng { uint64_t state; };

// One splitmix64 step: advance `x` and return a well-mixed 64-bit value.
SMD_HD inline uint64_t splitmix64(uint64_t& x) {
    x += 0x9E3779B97F4A7C15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Seed an INDEPENDENT stream for trajectory `traj` from the base seed, so each
// trajectory is uncorrelated yet fully reproducible from (base, traj).
SMD_HD inline Rng rng_seed(uint64_t base, uint64_t traj) {
    Rng r;
    r.state = base ^ (traj * 0x9E3779B97F4A7C15ULL + 0xD1B54A32D192ED03ULL);
    splitmix64(r.state);   // warm up so nearby seeds decorrelate immediately
    return r;
}

// Uniform double in (0,1] from 53 random bits (identical math host/device).
// We return (0,1] rather than [0,1) so the log() in the Gaussian draw never
// sees log(0).
SMD_HD inline double rng_uniform01(Rng& r) {
    uint64_t z = splitmix64(r.state);
    // (z>>11) is a 53-bit integer in [0, 2^53); +1 then *2^-53 maps to (0,1].
    return ((z >> 11) + 1) * (1.0 / 9007199254740992.0);   // 2^-53
}

// Standard-normal sample N(0,1) via the Box-Muller transform. We generate one
// value per call but discard the paired sine term -- simplicity over the small
// efficiency win, and it keeps the random-number CONSUMPTION per step fixed
// (exactly two uniforms), which is what makes CPU and GPU streams line up.
SMD_HD inline double rng_gauss(Rng& r) {
    const double u1 = rng_uniform01(r);
    const double u2 = rng_uniform01(r);
    // sqrt(-2 ln u1) * cos(2*pi*u2) ~ N(0,1).
    const double TWO_PI = 6.283185307179586476925286766559;
    return sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2);
}

// ---------------------------------------------------------------------------
// The potential of mean force U(xi) and its force -dU/dxi.
//   U(xi) = A*((xi-xa)*(xi-xb))^2 / (xb-xa)^2 + slope*xi
//   The quartic term is a symmetric double well (minima near xa and xb, barrier
//   between them); `slope` tilts the whole landscape so the bound and unbound
//   ends differ in free energy. The end-to-end ΔG used as ground truth is just
//   U(xi_end) - U(xi0) (see pmf_delta_g below) -- a KNOWN answer to test
//   Jarzynski against.
// ---------------------------------------------------------------------------
SMD_HD inline double pmf_energy(const SmdParams& p, double xi) {
    const double w  = (p.pmf_xb - p.pmf_xa);
    const double q  = (xi - p.pmf_xa) * (xi - p.pmf_xb) / w;  // = sqrt(quartic)/?? : scaled
    return p.pmf_A * q * q + p.pmf_slope * xi;
}

// Force from the PMF: F_pmf = -dU/dxi. Differentiate U analytically so the CPU
// and GPU evaluate the exact same closed form (no finite differencing).
SMD_HD inline double pmf_force(const SmdParams& p, double xi) {
    const double w   = (p.pmf_xb - p.pmf_xa);
    const double g   = (xi - p.pmf_xa) * (xi - p.pmf_xb);     // inner product
    const double dg  = (xi - p.pmf_xa) + (xi - p.pmf_xb);     // d/dxi of g
    // d/dxi [ A*g^2/w^2 + slope*xi ] = 2*A*g*dg/w^2 + slope ; force = -that.
    const double dU  = p.pmf_A * 2.0 * g * dg / (w * w) + p.pmf_slope;
    return -dU;
}

// The TRUE end-to-end free-energy difference (ground truth for verification):
// because xi0 and xi_end sit in the two wells, ΔG = U(xi_end) - U(xi0).
SMD_HD inline double pmf_delta_g(const SmdParams& p) {
    return pmf_energy(p, p.xi_end) - pmf_energy(p, p.xi0);
}

// ---------------------------------------------------------------------------
// run_trajectory: simulate ONE constant-velocity SMD pull and return the total
//   external work W done by the moving spring. This is the per-element physics
//   shared by the CPU reference (loops it over trajectories) and the GPU kernel
//   (one thread runs it). Determinism: identical seed -> identical W on both.
//
//   Overdamped Langevin (Euler-Maruyama) update of the coordinate xi:
//       xi_{n+1} = xi_n + (F_total/gamma)*dt + sqrt(2*kT*dt/gamma) * N(0,1)
//   where F_total = F_pmf(xi) + F_spring, and the noise amplitude satisfies the
//   fluctuation-dissipation theorem (so the unbiased system would sample the
//   Boltzmann distribution of U). See THEORY.md "The math" for the derivation.
//
//   Work bookkeeping (Hummer-Szabo / Jarzynski convention): the dummy atom moves
//   by d(center) = v*dt each step; the work increment is the spring force times
//   that displacement, dW = F_spring * v * dt, evaluated BEFORE the coordinate
//   moves. Summing dW over the pull gives the trajectory's external work W.
// ---------------------------------------------------------------------------
SMD_HD inline double run_trajectory(const SmdParams& p, uint64_t traj) {
    Rng rng = rng_seed(p.seed, traj);

    double xi     = p.xi0;     // reaction coordinate starts in the bound well, nm
    double center = p.xi0;     // spring attachment (dummy atom) starts on top of it
    double W      = 0.0;       // accumulated external work, kJ/mol

    // Pre-compute the two constant update coefficients (same every step):
    //   drift_scale : dt/gamma multiplies the deterministic force.
    //   noise_scale : sqrt(2*kT*dt/gamma) multiplies the Gaussian kick.
    const double drift_scale = p.dt / p.gamma;
    const double noise_scale = sqrt(2.0 * p.kT * p.dt / p.gamma);

    for (int s = 0; s < p.steps; ++s) {
        // 1) Forces at the current coordinate.
        const double f_pmf    = pmf_force(p, xi);              // landscape force
        const double f_spring = p.k_spring * (center - xi);   // SMD guiding force

        // 2) Work done by the spring as its centre advances by v*dt THIS step.
        //    (Accumulate before moving so dW uses the current spring force.)
        W += f_spring * p.v_pull * p.dt;

        // 3) Overdamped Langevin step for the coordinate.
        const double f_total = f_pmf + f_spring;
        xi += drift_scale * f_total + noise_scale * rng_gauss(rng);

        // 4) Advance the spring attachment point at constant velocity.
        center += p.v_pull * p.dt;
    }
    return W;
}

// ---------------------------------------------------------------------------
// Jarzynski free-energy estimate from a set of trajectory works.
//   ΔG = -kT * ln( mean_i exp(-W_i / kT) )
//   We subtract the minimum work Wmin first (a standard numerical trick): the
//   exponentials exp(-(W_i - Wmin)/kT) are all <= 1, avoiding overflow, and the
//   shift cancels exactly in the final ln (we add Wmin back). The sum is done in
//   a FIXED index order so the result is bit-reproducible (PATTERNS.md §3).
//   This is computed ONCE on the host from the verified work array, so CPU and
//   GPU share the identical reduction (like the ensemble summary in 9.02).
// ---------------------------------------------------------------------------
inline double jarzynski_dg(const double* W, int n, double kT) {
    if (n <= 0) return 0.0;
    double Wmin = W[0];
    for (int i = 1; i < n; ++i) if (W[i] < Wmin) Wmin = W[i];
    double acc = 0.0;                                  // sum of shifted exponentials
    for (int i = 0; i < n; ++i) acc += exp(-(W[i] - Wmin) / kT);
    return Wmin - kT * log(acc / static_cast<double>(n));
}
