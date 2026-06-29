// ===========================================================================
// src/gamd.h  --  Shared (host + device) GaMD model: potential, boost,
//                 deterministic RNG, one-walker Langevin run, fixed-point tally
// ---------------------------------------------------------------------------
// Project 1.25 : Gaussian-Accelerated MD (GaMD)   (reduced-scope teaching version)
//
// WHAT THIS PROJECT COMPUTES (and why it is a *reduced* version)
//   Real GaMD (AMBER pmemd.cuda, NAMD) runs a full all-atom molecular-dynamics
//   simulation and, every step, adds a Gaussian-distributed BOOST POTENTIAL to
//   the system's total potential energy. The boost smooths energy barriers, so
//   the system escapes minima far faster than plain MD -- WITHOUT needing a
//   hand-chosen reaction coordinate. Afterwards the unbiased free-energy surface
//   (the "PMF", potential of mean force) is recovered by REWEIGHTING the boosted
//   trajectory, classically via a CUMULANT EXPANSION of the boost to 2nd order.
//
//   A full MD force field (bonds, angles, Lennard-Jones, electrostatics, a
//   thermostat, periodic boundaries) is research-grade and far past one teaching
//   project (CLAUDE.md §13: ship the simplest correct teaching version, describe
//   the full one in THEORY.md). So we keep EVERYTHING THAT MAKES GaMD *GaMD* --
//   the boost potential, the threshold/force-constant statistics, and the
//   2nd-order cumulant reweighting -- but replace the 3N-dimensional protein with
//   a ONE-DIMENSIONAL model potential: a double well
//
//        U(x) = U_BARRIER * (x^2 - 1)^2          (two minima at x = +/-1)
//
//   and replace Newtonian MD + thermostat with OVERDAMPED LANGEVIN (Brownian)
//   dynamics, the simplest canonical-ensemble sampler:
//
//        x_{t+1} = x_t - (dt/gamma) * U'(x_t) + sqrt(2 * dt * kT / gamma) * N(0,1)
//
//   The double well is the textbook stand-in for a two-state conformational
//   change (e.g. an open<->closed protein, a folded<->unfolded toggle). Plain
//   Langevin at low temperature is TRAPPED in one well; GaMD's boost lets a
//   single short run visit both -- exactly the enhanced-sampling win GaMD gives
//   on real drug targets, shown on a system you can plot by hand.
//
// THE ENSEMBLE / GPU MAPPING (PATTERNS.md §1 row: "same ODE for many sets ->
//   thread per trajectory", exemplars 9.02 SEIR, 13.02 PBPK, plus per-thread RNG
//   like 5.01 Monte-Carlo dose)
//   We run an ENSEMBLE of independent walkers (different RNG seeds). Each walker
//   is a sequential time loop but independent of the others, so each GPU thread
//   owns one walker. Every walker, every step, drops a REWEIGHTING CONTRIBUTION
//   into the histogram bin for its current x. Many threads hit the same bin, so
//   the tally must be a DETERMINISTIC reduction -> we accumulate FIXED-POINT
//   INTEGERS with atomicAdd (PATTERNS.md §3 rule 2): integer adds commute, so the
//   GPU result is bit-identical to the serial CPU result regardless of thread
//   order. Float atomics would NOT be reproducible.
//
// CPU/GPU PARITY (PATTERNS.md §2)
//   Everything a thread does -- the potential U, its derivative, the boost, the
//   hand-rolled counter-based RNG, the per-step fixed-point tally -- lives in this
//   ONE header as GAMD_HD (__host__ __device__) inline functions. The CPU
//   reference (reference_cpu.cpp) and the GPU kernel (kernels.cu) therefore run
//   BYTE-FOR-BYTE IDENTICAL math, so verification is EXACT (tolerance 0), not
//   approximate. GAMD_HD expands to __host__ __device__ under nvcc and to nothing
//   under the plain host compiler. Keep CUDA-only constructs (__global__, <<<>>>)
//   OUT of this header so cl.exe/g++ can include it too.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh.
// READ THIS BEFORE: reference_cpu.h, kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <cstdint>   // uint32_t, uint64_t, int64_t (the fixed-point accumulators)
#include <cmath>     // std::sqrt, std::exp, std::log, std::cos (host side)

// ---- The HD decorator (PATTERNS.md §2) ------------------------------------
// Under nvcc, mark these inline functions as callable from BOTH host and device.
// Under the host compiler the decorators do not exist, so define them away.
#ifdef __CUDACC__
#define GAMD_HD __host__ __device__
#else
#define GAMD_HD
#endif

// ===========================================================================
// 1. SIMULATION CONFIGURATION
//   One immutable struct describing the whole ensemble run. It is plain old data
//   (no pointers, no STL) so it can be passed BY VALUE straight into a kernel --
//   the same trick 9.02/13.02 use. Loaded from the sample file by load_config().
// ===========================================================================
struct GamdConfig {
    // --- model potential U(x) = u_barrier * (x^2 - 1)^2 ---------------------
    double u_barrier;     // barrier height in units of kT (sets how trapped we are)

    // --- thermodynamics + Langevin integrator ------------------------------
    double kT;            // thermal energy k_B*T (sets equilibrium populations)
    double gamma_fric;    // overdamped friction coefficient (drag); sets timescale
    double dt;            // Langevin timestep (dimensionless model time)
    int    steps;         // timesteps per walker
    int    equil_steps;   // leading steps NOT tallied (let walkers forget x0)

    // --- GaMD boost parameters ---------------------------------------------
    // Boost ON when U(x) < E_threshold. We use GaMD's "lower-bound" form with the
    // threshold pinned to the energy ceiling Vmax, so the boost is the standard
    //     dV(x) = 0.5 * k * (E - U(x))^2   for U(x) < E,   else 0,
    // with k chosen from (Vmax, Vmin) so the boost never exceeds the energy gap
    // (the harmonic-boundary condition of Miao et al. 2015). See compute_k().
    double e_threshold;   // E: potential ceiling below which we boost (model units)
    double v_min;         // Vmin: lowest U on the sampled grid (for k)
    double v_max;         // Vmax: highest relevant U (= e_threshold here)
    double k0;            // dimensionless 0<k0<=1 force-constant (GaMD's sigma0 knob)

    // --- ensemble + histogram ----------------------------------------------
    int    n_walkers;     // number of independent walkers (= GPU threads)
    double x_lo, x_hi;    // histogram range on x (e.g. -2 .. +2)
    int    n_bins;        // number of PMF histogram bins
    uint64_t seed;        // base RNG seed (per-walker streams derived from it)
};

// Bin width of the PMF histogram (model-x units per bin).
GAMD_HD inline double bin_width(const GamdConfig& c) {
    return (c.x_hi - c.x_lo) / c.n_bins;
}

// Map a position x to its histogram bin index, or -1 if outside [x_lo, x_hi).
GAMD_HD inline int bin_of(const GamdConfig& c, double x) {
    if (x < c.x_lo || x >= c.x_hi) return -1;
    int b = (int)((x - c.x_lo) / bin_width(c));
    if (b < 0) b = 0;
    if (b >= c.n_bins) b = c.n_bins - 1;
    return b;
}

// GaMD harmonic force constant k for the boost dV = 0.5*k*(E - U)^2.
//   Miao et al. require the boost to be a smooth (harmonic) "lid": choosing
//   k = k0 / (Vmax - Vmin) with 0 < k0 <= 1 guarantees the boosted surface keeps
//   the same *ordering* of states (no new minima are created) -- the key
//   correctness condition of the method. k0 -> 1 is the most aggressive boost.
GAMD_HD inline double compute_k(const GamdConfig& c) {
    const double span = c.v_max - c.v_min;
    return (span > 0.0) ? (c.k0 / span) : 0.0;
}

// ===========================================================================
// 2. THE MODEL POTENTIAL  U(x)  and its analytic derivative  U'(x)
//   U(x) = u_barrier * (x^2 - 1)^2 : symmetric double well, minima at x=+/-1
//   (U=0), barrier at x=0 (U=u_barrier). U'(x) = u_barrier * 4x(x^2 - 1).
//   The Langevin force is -U'(x). We use the EXACT analytic gradient (no finite
//   differences) so CPU and GPU compute the identical number.
// ===========================================================================
GAMD_HD inline double potential_U(const GamdConfig& c, double x) {
    const double t = x * x - 1.0;
    return c.u_barrier * t * t;
}

GAMD_HD inline double dU_dx(const GamdConfig& c, double x) {
    return c.u_barrier * 4.0 * x * (x * x - 1.0);
}

// ===========================================================================
// 3. THE GaMD BOOST POTENTIAL  dV(x)  and boosted force
//   Boost is applied only where U(x) < E (the "lower energy boundary"); above E
//   the system already explores freely, so we leave it alone. dV is a Gaussian-
//   shaped (quadratic-in-energy) lid that lifts deep minima toward E:
//       dV(x) = 0.5 * k * (E - U(x))^2     if U(x) < E
//             = 0                          otherwise
//   The boosted potential is V*(x) = U(x) + dV(x); the boosted FORCE is
//       -dV*/dx = -(1 + k*(E - U)) * U'(x)   for U < E.
//   i.e. the bias just SCALES the real force by a factor that shrinks barriers.
// ===========================================================================
GAMD_HD inline double boost_dV(const GamdConfig& c, double U) {
    const double E = c.e_threshold;
    if (U >= E) return 0.0;                 // no boost above the threshold
    const double k = compute_k(c);
    const double d = E - U;
    return 0.5 * k * d * d;
}

// Boosted force F* = -dV*/dx used to propagate the walker.
GAMD_HD inline double boosted_force(const GamdConfig& c, double x) {
    const double U  = potential_U(c, x);
    const double Up = dU_dx(c, x);          // U'(x)
    const double E  = c.e_threshold;
    if (U >= E) return -Up;                 // unboosted region: plain force
    const double k = compute_k(c);
    // chain rule on dV = 0.5 k (E-U)^2 :  d(dV)/dx = -k (E-U) U'(x)
    // total boosted force = -(U' + d(dV)/dx) = -(1 + k(E-U)) U'
    return -(1.0 + k * (E - U)) * Up;
}

// ===========================================================================
// 4. DETERMINISTIC, HAND-ROLLED RNG  (so CPU == GPU bit-for-bit)
//   We deliberately do NOT use cuRAND on the device and <random> on the host:
//   they would produce different streams, breaking exact verification. Instead a
//   tiny COUNTER-BASED generator (splitmix64) gives each (walker, step) its own
//   reproducible 64-bit value from pure integer hashing -- identical on host and
//   device, order-independent, and trivially parallel (no shared RNG state). This
//   is the same philosophy as a Philox counter-based RNG, kept readable.
// ===========================================================================

// splitmix64: hash a 64-bit counter to a well-mixed 64-bit value.
GAMD_HD inline uint64_t splitmix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Uniform double in (0,1). We take 53 bits (the double mantissa) and scale; the
// +0.5 keeps us strictly inside (0,1) so log() in the Gaussian below is finite.
GAMD_HD inline double u01(uint64_t r) {
    return ((r >> 11) + 0.5) * (1.0 / 9007199254740992.0);  // / 2^53
}

// One standard normal N(0,1) via the Box-Muller transform from two uniforms.
//   We hash (seed, walker, step, lane) deterministically so the SAME draw comes
//   out on host and device. lane 0 and 1 give the two independent uniforms.
GAMD_HD inline double gaussian(uint64_t seed, uint32_t walker, uint32_t step) {
    // Build two distinct 64-bit counters and hash each -> two uniforms.
    const uint64_t base = seed
                        + (uint64_t)walker * 0x100000000ULL   // walker in high bits
                        + (uint64_t)step  * 0x2545F4914F6CDD1DULL;
    const double u1 = u01(splitmix64(base + 1ULL));
    const double u2 = u01(splitmix64(base + 2ULL));
    // Box-Muller: sqrt(-2 ln u1) * cos(2*pi*u2) ~ N(0,1).
    const double TWO_PI = 6.283185307179586476925286766559;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(TWO_PI * u2);
}

// ===========================================================================
// 5. FIXED-POINT REWEIGHTING ACCUMULATORS (deterministic atomics, PATTERNS §3)
//   To recover the unbiased PMF we need, per bin, the cumulant-expansion sums:
//       n[b]   : number of (boosted) samples that fell in bin b      (integer)
//       S1[b]  : sum of dV over those samples            (1st moment of boost)
//       S2[b]  : sum of dV^2 over those samples          (2nd moment of boost)
//   On the GPU many threads add into the same bin. A floating S1/S2 would be
//   order-dependent (float add is not associative) -> NON-reproducible. So we
//   accumulate FIXED-POINT INTEGERS: multiply the real value by a large scale and
//   add as int64. Integer addition commutes, so the GPU sum equals the CPU sum
//   EXACTLY, independent of thread interleaving. We divide the scale back out at
//   the very end. (n[b] is already an integer count.)
//
//   Scale choice: boost energies here are O(1..50) kT; DV_SCALE = 2^20 keeps ~6
//   decimal digits with no overflow over millions of samples in int64.
// ===========================================================================
// constexpr (not plain `static const`) so the value is a compile-time constant
// usable inside __device__ code as well as host code -- a plain file-scope static
// is not addressable from device functions under nvcc.
constexpr double DV_SCALE = 1048576.0;   // 2^20 fixed-point scale for dV sums

// Convert a real boost value to its fixed-point integer representation.
//   round-to-nearest via +/- 0.5 so the host and device truncation agree exactly.
GAMD_HD inline int64_t dv_to_fixed(double v) {
    double scaled = v * DV_SCALE;
    return (int64_t)(scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5);
}

// The three per-bin accumulators, kept as plain arrays of int64 of length n_bins
// each. We store them in ONE flat device array laid out as [count | S1 | S2].
//   index helpers below keep main.cu / kernels.cu / reference_cpu.cpp in sync.
GAMD_HD inline int acc_count_idx(const GamdConfig& c, int b) { return b; }
GAMD_HD inline int acc_s1_idx   (const GamdConfig& c, int b) { return c.n_bins + b; }
GAMD_HD inline int acc_s2_idx   (const GamdConfig& c, int b) { return 2 * c.n_bins + b; }
GAMD_HD inline int acc_total    (const GamdConfig& c)        { return 3 * c.n_bins; }

// ===========================================================================
// 6. ONE WALKER'S CONTRIBUTION -- the heart of the simulation
//   Run a single overdamped-Langevin walker under the GaMD boost for c.steps
//   steps, and, after equilibration, deposit each visited sample's (1, dV, dV^2)
//   into its bin. The deposit is done through a caller-supplied ADD functor so
//   the SAME loop serves both the CPU (plain += ) and the GPU (atomicAdd). This
//   is the single function the whole correctness story rests on: because it is
//   GAMD_HD and the RNG/potential/boost are all deterministic integer/closed-form
//   math, walker w produces the identical samples on CPU and GPU.
//
//   Template parameter AddFn: a callable add(int flat_index, int64_t value).
//     CPU passes a lambda doing acc[i] += v.
//     GPU passes a lambda doing atomicAdd((unsigned long long*)&acc[i], v).
//   We template instead of branching so there is zero per-step overhead and the
//   numeric path is provably identical.
//
//   Returns the walker's final x (handy as a cheap deterministic sanity print).
// ===========================================================================
template <class AddFn>
GAMD_HD inline double run_walker(const GamdConfig& c, uint32_t w, AddFn add) {
    // Deterministic, walker-specific start: alternate walkers begin in the left
    // (x=-1) and right (x=+1) wells so the ensemble starts balanced and the demo
    // shows GaMD letting BOTH populations cross the barrier. (Plain Langevin would
    // leave each walker stuck in its starting well.)
    double x = (w & 1u) ? 1.0 : -1.0;

    const double inv_gamma = 1.0 / c.gamma_fric;            // mobility 1/gamma
    // Langevin noise amplitude: sqrt(2 dt kT / gamma). Precompute once per walker.
    const double noise_amp = std::sqrt(2.0 * c.dt * c.kT * inv_gamma);

    for (int s = 0; s < c.steps; ++s) {
        // --- propagate one overdamped-Langevin step under the BOOSTED force ---
        //   x <- x + (dt/gamma) F*(x) + noise_amp * N(0,1)
        const double F = boosted_force(c, x);              // biased force at x
        const double xi = gaussian(c.seed, w, (uint32_t)s); // deterministic N(0,1)
        x += c.dt * inv_gamma * F + noise_amp * xi;

        // --- after equilibration, tally this sample for reweighting -----------
        if (s >= c.equil_steps) {
            const int b = bin_of(c, x);                    // -1 if walker left grid
            if (b >= 0) {
                const double U  = potential_U(c, x);
                const double dV = boost_dV(c, U);          // boost we applied here
                // Deposit (count=1, S1=dV, S2=dV^2) in fixed point. The count uses
                // fixed scale 1 (it's already an integer); S1,S2 use DV_SCALE.
                add(acc_count_idx(c, b), (int64_t)1);
                add(acc_s1_idx(c, b),    dv_to_fixed(dV));
                add(acc_s2_idx(c, b),    dv_to_fixed(dV * dV));
            }
        }
    }
    return x;
}

// ===========================================================================
// 7. REWEIGHTING:  boosted histogram  ->  unbiased PMF (free energy)
//   GaMD's payoff. The boosted simulation oversamples high-energy regions, so a
//   raw histogram is WRONG. The exact reweight multiplies each sample by
//   exp(beta*dV); the cumulant expansion approximates the LOG of that ensemble
//   average to 2nd order (Miao, Sinko, Pierce, ... 2014):
//
//     ln<e^{beta dV}>_b  ~=  beta*<dV>_b + (beta^2/2)*( <dV^2>_b - <dV>_b^2 )
//                            \____1st____/   \________2nd cumulant (variance)____/
//
//   The unbiased free energy (PMF) of bin b, up to an additive constant, is
//     F(b) = -kT * [ ln p_boost(b) + ln<e^{beta dV}>_b ]
//   where p_boost(b) = n[b]/N is the boosted occupancy. We return F shifted so
//   its minimum is 0 (a PMF is only defined up to a constant). Bins never visited
//   get +infinity (sentinel = +1e30) so callers can skip them.
//
//   This function is GAMD_HD too, but it only runs on the HOST in this project
//   (post-processing a handful of bins); keeping it here documents the full
//   pipeline in one place.
// ===========================================================================
GAMD_HD inline double reweight_pmf_bin(const GamdConfig& c,
                                       int64_t count, int64_t s1_fixed, int64_t s2_fixed,
                                       double total_samples) {
    if (count <= 0 || total_samples <= 0.0) return 1e30;   // unvisited -> +inf
    const double beta = 1.0 / c.kT;                        // inverse temperature
    const double mean_dV  = ((double)s1_fixed / DV_SCALE) / (double)count;        // <dV>
    const double mean_dV2 = ((double)s2_fixed / DV_SCALE) / (double)count;        // <dV^2>
    double var_dV = mean_dV2 - mean_dV * mean_dV;          // variance of boost
    if (var_dV < 0.0) var_dV = 0.0;                        // guard tiny round-off
    // 2nd-order cumulant of ln<exp(beta dV)>:
    const double ln_reweight = beta * mean_dV + 0.5 * beta * beta * var_dV;
    const double p_boost = (double)count / total_samples;  // boosted occupancy
    // F = -kT [ ln p_boost + ln_reweight ]  (constant shift applied by caller)
    return -c.kT * (std::log(p_boost) + ln_reweight);
}

// Analytic reference PMF of the bare double well at the bin center, F(x)=U(x),
// shifted so its minimum is 0. Used as the SECOND, scientific check (PATTERNS §4):
// a correct reweight must recover the true barrier height u_barrier.
GAMD_HD inline double analytic_pmf(const GamdConfig& c, double x) {
    return potential_U(c, x);   // minima already at U=0, barrier at U=u_barrier
}
