// ===========================================================================
// src/tps_physics.h  --  Shared (host + device) RNG, Langevin BD, and the
//                        single TPS "shooting move" -- the one true physics.
// ---------------------------------------------------------------------------
// Project 2.32 : Protein Folding Pathway Extraction (Transition Path Sampling)
//                -- a deliberately REDUCED-SCOPE teaching version (CLAUDE.md §13)
//
// WHY THIS HEADER IS SHARED  (the most important idea in this project)
//   Transition Path Sampling (TPS) is a Monte-Carlo method on the space of
//   TRAJECTORIES. To VERIFY a GPU implementation exactly, the CPU reference and
//   the GPU kernel must run the *identical* shooting moves and tally identical
//   counts. That only works if both compile the SAME RNG and the SAME dynamics.
//   So all per-shooter physics lives HERE, in one header included by
//   reference_cpu.cpp (host compiler) AND kernels.cu / main.cu (nvcc).
//
//   The TPS_HD macro expands to `__host__ __device__` under nvcc and to nothing
//   under a plain host compiler, so the same inline functions compile in both
//   worlds (PATTERNS.md §2, the HD-macro idiom). Keep CUDA-only types out of
//   this header (no `__global__`) so the host compiler can include it.
//   Production TPS uses cuRAND + a full MD engine (OpenMM); we use a shared,
//   reproducible counter-based RNG specifically so CPU and GPU histories are
//   bit-identical and the demo's stdout is deterministic -- see THEORY.md.
//
// ---------------------------------------------------------------------------
// THE SCIENCE, COMPRESSED TO ONE DIMENSION (the full picture is in THEORY.md)
//   Protein folding is a rare event: the molecule spends almost all its time in
//   the UNFOLDED basin (call it A) or the FOLDED basin (B), and only rarely
//   crosses the high free-energy BARRIER (the transition-state region) between
//   them. Brute-force MD wastes ~all its compute sitting inside a basin waiting
//   for a crossing that may take milliseconds. TPS sidesteps that: it samples
//   only the reactive trajectories (the actual A<->B crossings) by "shooting"
//   short trajectories from points near the barrier and keeping the ones that
//   connect the two basins.
//
//   We collapse the 3N-dimensional protein onto ONE reaction coordinate x (a
//   "folding order parameter" -- e.g. the fraction of native contacts, rescaled
//   so x runs roughly over [0,1]). The folding free-energy landscape along x is
//   a DOUBLE WELL:
//       V(x) = BARRIER * ( (x - x0)^2 / w^2 - 1 )^2
//   which has minima at x = x0 +/- w (the A and B basins) separated by a barrier
//   of height BARRIER at x = x0. The bead does overdamped Langevin (Brownian)
//   dynamics on this landscape:
//       x_{t+1} = x_t - (D*dt/kT) * V'(x_t) + sqrt(2*D*dt) * N(0,1)
//   D = diffusion constant, kT = thermal energy, dt = timestep. The first term
//   is the deterministic drift downhill; the second is thermal kicking (the only
//   thing that lets the bead ever climb the barrier). This is the GROMACS-style
//   "implicit-solvent BD on a free-energy surface" reduced to 1-D.
//
// ---------------------------------------------------------------------------
// THE TPS SHOOTING MOVE WE IMPLEMENT (aimless shooting, simplified)
//   1. Pick a SHOOTING POINT x_sp drawn near the barrier top (the interesting
//      region; aimless shooting in real codes selects it from an existing path).
//   2. Draw fresh random "momenta" -- here, a fresh RNG stream -- and integrate
//      the BD dynamics FORWARD until the bead first reaches basin A or basin B
//      (or a step budget runs out).
//   3. Integrate BACKWARD in time from the same x_sp with the time-reversed
//      noise until it reaches a basin. (For overdamped dynamics the backward
//      leg is statistically just another forward BD run from x_sp -- a standard
//      teaching simplification; THEORY.md §real-world explains the rigorous
//      momentum-reversal version.)
//   4. ACCEPT the trajectory as a transition path iff one end landed in A and
//      the other in B (it CONNECTS the basins). Reject A->A and B->B paths.
//   5. COMMITTOR: the forward leg's endpoint also tells us, for this shooting
//      point, whether *this* shot committed to B. Averaging that indicator over
//      many shots from the same x gives p_B(x), the committor -- and the true
//      transition state is the isosurface p_B = 1/2 (THEORY.md §committor).
//
//   Every shooter is INDEPENDENT -> one GPU thread per shooter (the catalog's
//   "embarrassingly parallel independent shooter array"). Each thread returns a
//   small fixed-size ShotResult; the caller tallies integer counts with
//   atomicAdd (GPU) or += (CPU). Integer tallies => order-independent => the GPU
//   result is deterministic AND equals the CPU's exactly (PATTERNS.md §3).
// ===========================================================================
#pragma once

#include <cstdint>
#include <cmath>

#ifdef __CUDACC__
#define TPS_HD __host__ __device__
#else
#define TPS_HD
#endif

// ---------------------------------------------------------------------------
// RNG: a splitmix64 counter-based stream (identical bit-for-bit on host/device).
// Counter-based (not stateful like Mersenne Twister) so that shooter `i` can
// deterministically reconstruct its OWN independent stream from (seed, i) with
// no cross-thread state -- exactly what a massively parallel RNG needs.
// ---------------------------------------------------------------------------
struct Rng { uint64_t state; };  // the whole RNG is one 64-bit word

// One splitmix64 step: advance `x` and return a well-mixed 64-bit value.
// (The three magic constants are the published splitmix64 finalizer; they
// scramble the counter so consecutive draws look independent.)
TPS_HD inline uint64_t splitmix64(uint64_t& x) {
    x += 0x9E3779B97F4A7C15ULL;                 // golden-ratio increment
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Seed an independent stream for shooter `shooter` from a base seed, so every
// shooter is uncorrelated yet exactly reproducible from (base, shooter).
TPS_HD inline Rng rng_seed(uint64_t base, uint64_t shooter) {
    Rng r;
    r.state = base ^ (shooter * 0x9E3779B97F4A7C15ULL + 0xD1B54A32D192ED03ULL);
    splitmix64(r.state);    // warm up so close seeds diverge immediately
    return r;
}

// Uniform double in [0,1) from 53 random bits (identical math host/device).
TPS_HD inline double rng_uniform(Rng& r) {
    uint64_t z = splitmix64(r.state);
    return (z >> 11) * (1.0 / 9007199254740992.0);   // multiply by 2^-53
}

// Standard normal sample N(0,1) via the Box-Muller transform. We use ONLY the
// cosine branch and discard the sine partner: that wastes one of the two
// Gaussians Box-Muller produces, but it keeps the RNG-draw count per step
// FIXED and SIMPLE, which is what makes the host and device draw the exact same
// stream (a cached second value would complicate the shared state). The thermal
// kick in BD is Gaussian, so this is the noise term sqrt(2 D dt)*N(0,1).
TPS_HD inline double rng_normal(Rng& r) {
    const double TWO_PI = 6.283185307179586476925286766559;
    double u1 = rng_uniform(r);
    double u2 = rng_uniform(r);
    // Guard log(0): u1 in [0,1) can be exactly 0, so floor it to a tiny value.
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2);
}

// ---------------------------------------------------------------------------
// Simulation parameters (read from the data file; see data/README.md).
// All in reduced (dimensionless) "folding units": x is an order parameter, not
// nanometres, and energies are in units of kT. This is standard for a 1-D
// free-energy-surface teaching model -- THEORY.md §math defines the mapping.
// ---------------------------------------------------------------------------
struct SimParams {
    double barrier;   // double-well barrier height in units of kT (e.g. 5.0)
    double x0;        // landscape centre (barrier-top position), e.g. 0.5
    double w;         // half-distance between the two basins, e.g. 0.4
    double D;         // diffusion constant on x (reduced units), e.g. 1.0
    double dt;        // integration timestep (reduced units), e.g. 0.0005
    double basin_tol; // how close to a basin minimum counts as "arrived", e.g. 0.05
    int    max_steps; // step budget per trajectory leg (rare-event safety net)
    int    n_shooters;// number of independent shooting moves to run
    int    n_bins;    // committor histogram resolution along x (e.g. 20)
    uint64_t seed;    // base RNG seed (shooter i uses stream (seed, i))
};

// The two basin minima of V(x): x0 - w (basin A, "unfolded") and x0 + w
// (basin B, "folded"). Defined once so the dynamics and the basin test agree.
TPS_HD inline double basin_A_min(const SimParams& P) { return P.x0 - P.w; }
TPS_HD inline double basin_B_min(const SimParams& P) { return P.x0 + P.w; }

// The double-well potential V(x) (in kT). Minima at x0 +/- w, barrier at x0.
TPS_HD inline double potential(const SimParams& P, double x) {
    double q = (x - P.x0) / P.w;          // dimensionless distance from centre
    double t = q * q - 1.0;               // zero at the two basin minima
    return P.barrier * t * t;             // quartic double well
}

// The force F(x) = -dV/dx, the deterministic drift in the BD update.
//   V(x) = barrier * (q^2 - 1)^2,  q = (x-x0)/w
//   dV/dx = barrier * 2 (q^2 - 1) * 2q * (1/w) = (4*barrier/w) q (q^2 - 1)
//   F = -dV/dx
TPS_HD inline double force(const SimParams& P, double x) {
    double q = (x - P.x0) / P.w;
    return -(4.0 * P.barrier / P.w) * q * (q * q - 1.0);
}

// Which basin (if any) is x currently in?  -1 = none, 0 = basin A, 1 = basin B.
// "In a basin" = within basin_tol of that basin's minimum. This is the absorbing
// boundary that ends a trajectory leg: BD walks until it first commits.
TPS_HD inline int basin_of(const SimParams& P, double x) {
    if (fabs(x - basin_A_min(P)) <= P.basin_tol) return 0;   // unfolded
    if (fabs(x - basin_B_min(P)) <= P.basin_tol) return 1;   // folded
    return -1;                                                // in transit
}

// One overdamped Langevin (Brownian-dynamics) step on V(x):
//   x_{t+1} = x_t + (D*dt/kT) * F(x_t) + sqrt(2*D*dt) * N(0,1)
// kT = 1 in our reduced units (energies already in kT), so it drops out.
// The first term slides downhill; the second is the thermal kick that, rarely,
// pushes the bead over the barrier -- the microscopic origin of the rare event.
TPS_HD inline double bd_step(const SimParams& P, double x, Rng& rng) {
    double drift = (P.D * P.dt) * force(P, x);        // deterministic part
    double kick  = sqrt(2.0 * P.D * P.dt) * rng_normal(rng);  // stochastic part
    return x + drift + kick;
}

// Integrate ONE BD trajectory leg from x_start until it first enters a basin or
// the step budget runs out. Returns the basin reached (0=A, 1=B, -1=neither).
// This is the inner loop shared by the forward and backward shooting legs; the
// caller decides what an A/B/none outcome MEANS for acceptance & committor.
TPS_HD inline int run_leg(const SimParams& P, double x_start, Rng& rng) {
    double x = x_start;
    for (int s = 0; s < P.max_steps; ++s) {
        x = bd_step(P, x, rng);
        // Reflect at the outer walls so the bead cannot wander off to +/-inf if
        // a huge kick overshoots; the wells are at x0 +/- w, walls a bit beyond.
        double lo = P.x0 - 2.0 * P.w, hi = P.x0 + 2.0 * P.w;
        if (x < lo) x = 2.0 * lo - x;    // mirror back in
        if (x > hi) x = 2.0 * hi - x;
        int b = basin_of(P, x);
        if (b >= 0) return b;            // committed to a basin -> stop
    }
    return -1;                           // never committed within the budget
}

// A compact, fixed-size result for one shooting move -- small enough to live in
// registers and be returned by value from a kernel thread. No dynamic memory,
// no per-thread arrays of unknown size: that keeps the GPU kernel simple and
// the CPU/GPU code paths identical.
struct ShotResult {
    int  fwd_basin;     // basin the forward leg reached  (0=A,1=B,-1=none)
    int  bwd_basin;     // basin the backward leg reached (0=A,1=B,-1=none)
    int  is_transition; // 1 if the path connects A and B (one end A, other B)
    int  committed_B;   // 1 if the forward leg committed to B (for committor)
    int  sp_bin;        // committor-histogram bin of this shot's shooting point
};

// Map a shooting-point position x to a committor-histogram bin in [0, n_bins).
// We bin over the transition region [x0 - w, x0 + w] (basin A min to basin B
// min). Clamped so out-of-range x never indexes out of bounds.
TPS_HD inline int committor_bin(const SimParams& P, double x) {
    double lo = basin_A_min(P), hi = basin_B_min(P);
    double f = (x - lo) / (hi - lo);          // 0 at A, 1 at B
    int b = static_cast<int>(f * P.n_bins);
    if (b < 0) b = 0;
    if (b >= P.n_bins) b = P.n_bins - 1;
    return b;
}

// Deterministically pick this shooter's SHOOTING POINT. We spread shooting
// points across the transition region so the committor histogram is populated
// everywhere: shooter i is placed at a fraction (i+0.5)/n along [A_min, B_min],
// then nudged by a small reproducible jitter. (Real aimless shooting picks the
// point from a stored path; we synthesise a representative spread instead so the
// single demo run sweeps the whole committor curve -- THEORY.md §algorithm.)
TPS_HD inline double shooting_point(const SimParams& P, int shooter, Rng& rng) {
    double lo = basin_A_min(P), hi = basin_B_min(P);
    double frac = (shooter + 0.5) / static_cast<double>(P.n_shooters);
    double jitter = (rng_uniform(rng) - 0.5) * (hi - lo) / P.n_shooters;
    double x = lo + frac * (hi - lo) + jitter;
    if (x < lo) x = lo;                       // keep inside the transition region
    if (x > hi) x = hi;
    return x;
}

// THE SHOOTING MOVE: run one full TPS shot for shooter `shooter` and return its
// ShotResult. This is the single function both the CPU loop and the GPU kernel
// call -- so a given shooter index produces byte-identical results on both.
//   * One RNG stream per shooter (rng_seed): independent and reproducible.
//   * Forward leg from x_sp; backward leg = an independent forward BD leg from
//     x_sp (the overdamped simplification noted above).
//   * Transition iff the two legs land in DIFFERENT basins (one A, one B).
TPS_HD inline ShotResult run_shot(const SimParams& P, int shooter) {
    Rng rng = rng_seed(P.seed, static_cast<uint64_t>(shooter));

    double x_sp = shooting_point(P, shooter, rng);   // this shot's shooting point

    int fwd = run_leg(P, x_sp, rng);                 // forward in time
    int bwd = run_leg(P, x_sp, rng);                 // backward (indep. leg)

    ShotResult R;
    R.fwd_basin = fwd;
    R.bwd_basin = bwd;
    // A transition path connects the two basins: one leg in A (0), one in B (1).
    R.is_transition = ((fwd == 0 && bwd == 1) || (fwd == 1 && bwd == 0)) ? 1 : 0;
    // Committor indicator: did the forward shot commit to the folded basin B?
    R.committed_B = (fwd == 1) ? 1 : 0;
    R.sp_bin = committor_bin(P, x_sp);
    return R;
}
