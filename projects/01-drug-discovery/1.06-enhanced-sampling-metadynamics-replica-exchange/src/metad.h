// ===========================================================================
// src/metad.h  --  Shared (host + device) well-tempered metadynamics core
// ---------------------------------------------------------------------------
// Project 1.6 : Enhanced Sampling -- Metadynamics & Replica Exchange
//               (REDUCED-SCOPE TEACHING VERSION -- see README §Limitations and
//                THEORY.md "Where this sits in the real world").
//
// WHAT THIS PROJECT COMPUTES (the one-paragraph version)
//   A single particle diffuses on a 1-D *double-well* free-energy landscape
//   F0(s) along one collective variable (CV) s. Plain Langevin molecular
//   dynamics gets STUCK in whichever well it starts in: crossing the barrier
//   between the wells is a rare event on accessible timescales. WELL-TEMPERED
//   METADYNAMICS fixes this by periodically dropping small Gaussian "hills" of
//   bias potential at the walker's current position. The accumulated bias fills
//   up the well the walker sits in, pushing it out, so it explores BOTH wells.
//   Crucially, once converged the *negative, rescaled* accumulated bias
//   reconstructs the underlying free-energy surface:  F(s) ~= -(1 + 1/gamma) V_bias(s).
//   So metadynamics is simultaneously an accelerated sampler AND a free-energy
//   estimator -- exactly why it is a workhorse for drug-binding free energies.
//
// THE GPU ANGLE (multi-walker metadynamics; PATTERNS.md §1 "ensemble RK4")
//   Real metadynamics often runs MANY WALKERS at once (multi-walker MetaD), each
//   an independent trajectory exploring the same landscape. That maps perfectly
//   onto the GPU's "ensemble / thread-per-trajectory" pattern (cf. flagships 9.02
//   SEIR and 13.02 PBPK): one GPU thread integrates one walker's entire Langevin
//   + hill-deposition history in registers/local memory, writing back its private
//   bias grid and a summary. No inter-thread communication -> embarrassingly
//   parallel. (In production, walkers SHARE a single growing hill list; we keep
//   them independent here so the teaching kernel needs no atomics/sync -- see
//   THEORY.md "GPU mapping" and the Exercises for the shared-bias extension.)
//
// WHY A SHARED __host__ __device__ HEADER (the CPU/GPU-parity idiom)
//   Every formula a walker uses -- the force, the Gaussian-hill bias, the
//   Langevin half-steps, and the deterministic RNG -- lives HERE as
//   `__host__ __device__` inline functions. The CPU reference (reference_cpu.cpp)
//   and the GPU kernel (kernels.cu) therefore execute the *byte-for-byte identical*
//   sequence of double-precision operations. That makes verification EXACT
//   (machine precision) instead of approximate. METAD_HD expands to
//   `__host__ __device__` under nvcc and to nothing under the host compiler.
//
//   Keep this header free of CUDA-only constructs (no __global__, no <<<>>>), so
//   the plain host compiler can include it too. Only <cstdint>/<cmath> are used.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh.
// READ THIS BEFORE: kernels.cu, reference_cpu.cpp, main.cu.
// ===========================================================================
#pragma once

#include <cmath>      // sin, cos, exp, log, sqrt, M_PI (we define PI ourselves for MSVC)
#include <cstdint>    // uint32_t, uint64_t for the counter-based RNG

// --- The HD macro: decorate shared math as host+device under nvcc only. -----
#ifdef __CUDACC__
#define METAD_HD __host__ __device__
#else
#define METAD_HD
#endif

// We avoid relying on M_PI (not guaranteed by <cmath> on MSVC without a macro)
// and define the constants we need explicitly so host and device agree exactly.
namespace metad {

// Two-pi, used by the Box-Muller Gaussian transform below.
constexpr double TWO_PI = 6.283185307179586476925286766559;

// ===========================================================================
// 1. THE PHYSICAL SYSTEM:  a 1-D double-well free-energy landscape F0(s)
// ---------------------------------------------------------------------------
//   We use the canonical quartic double well
//       F0(s) = A * (s^2 - 1)^2
//   which has two symmetric minima at s = -1 and s = +1 (where F0 = 0) and a
//   barrier of height A at s = 0. This is the textbook stand-in for a molecular
//   CV with two metastable states (e.g. the phi/psi basins of alanine dipeptide,
//   or a bound vs. unbound ligand pose projected onto a single reaction
//   coordinate). The force is the negative gradient:
//       f(s) = -dF0/ds = -A * 2 (s^2 - 1) * 2 s = -4 A s (s^2 - 1).
//
//   In a REAL MD code F0 is never known in closed form -- it is the thing we are
//   trying to discover. Here we KNOW it, which is precisely what lets us VERIFY
//   that metadynamics reconstructs it (see main.cu's FES-recovery check).
// ===========================================================================

// Parameters of the model landscape + the thermostat. Plain doubles so the whole
// struct copies trivially to the device by value (passed as a kernel argument).
struct Model {
    double A;        // barrier height of F0 (in kT units); minima at s=+/-1
    double kT;       // thermal energy k_B*T (sets noise strength + hill temperature)
    double mass;     // particle mass (CV "inertia"); 1.0 in reduced units
    double friction; // Langevin friction coefficient gamma_L (1/time); damping
    double dt;       // integration timestep
    int    steps;    // total Langevin steps per walker

    // --- Metadynamics controls ---
    double hill_w;        // Gaussian hill HEIGHT (energy) at deposition, pre-tempering
    double hill_sigma;    // Gaussian hill WIDTH along s (std-dev)
    int    deposit_every; // deposit one hill every this many steps (the "pace")
    double bias_factor;   // well-tempered bias factor gamma (>1); gamma->inf = standard MetaD

    // --- Bias grid: we accumulate the bias on a uniform grid over [s_lo, s_hi]
    //     so that evaluating V_bias(s) and its gradient is O(1) per step instead
    //     of O(#hills). (Grid metadynamics; the alternative is summing every hill.)
    double s_lo, s_hi;    // grid extent along the CV
    int    nbins;         // number of grid points (resolution of the FES)
};

// Underlying (true) free energy F0(s) = A (s^2-1)^2.  Used only to (a) provide
// the conservative force and (b) compare against the metadynamics-recovered FES.
METAD_HD inline double true_fes(const Model& m, double s) {
    const double t = s * s - 1.0;
    return m.A * t * t;
}

// Conservative force f(s) = -dF0/ds = -4 A s (s^2 - 1) from the double well.
METAD_HD inline double well_force(const Model& m, double s) {
    return -4.0 * m.A * s * (s * s - 1.0);
}

// ===========================================================================
// 2. THE BIAS GRID:  index math + evaluation of V_bias(s) and dV_bias/ds
// ---------------------------------------------------------------------------
//   We store the bias potential sampled at `nbins` equally spaced grid points
//   s_j = s_lo + j*ds. Two helpers convert between s and grid coordinates; two
//   more read the bias and its gradient by LINEAR INTERPOLATION between the
//   nearest grid points (so the biased force is continuous).
// ===========================================================================

// Grid spacing ds between adjacent bins.
METAD_HD inline double grid_ds(const Model& m) {
    return (m.s_hi - m.s_lo) / (m.nbins - 1);
}

// Continuous grid coordinate x = (s - s_lo)/ds  (so bin j sits at integer x=j).
METAD_HD inline double grid_coord(const Model& m, double s) {
    return (s - m.s_lo) / grid_ds(m);
}

// Read the accumulated bias V_bias(s) by linear interpolation on the grid.
//   bias[] is the per-walker grid (length m.nbins). Clamp to the grid edges so a
//   walker that wanders just outside [s_lo,s_hi] still gets a sensible value.
METAD_HD inline double bias_value(const Model& m, const double* bias, double s) {
    double x = grid_coord(m, s);
    if (x <= 0.0)            return bias[0];
    if (x >= m.nbins - 1)    return bias[m.nbins - 1];
    int    j  = (int)x;            // left grid point
    double fr = x - j;             // fractional position in [0,1) between j and j+1
    return bias[j] * (1.0 - fr) + bias[j + 1] * fr;   // lerp
}

// Read the bias GRADIENT dV_bias/ds by differentiating the same linear
// interpolant. On a segment [j, j+1] the interpolant is linear in s, so its
// slope is (bias[j+1]-bias[j]) / ds. Outside the grid the bias is flat -> 0.
METAD_HD inline double bias_grad(const Model& m, const double* bias, double s) {
    double x = grid_coord(m, s);
    if (x <= 0.0 || x >= m.nbins - 1) return 0.0;
    int j = (int)x;
    return (bias[j + 1] - bias[j]) / grid_ds(m);
}

// ===========================================================================
// 3. DEPOSITING A WELL-TEMPERED HILL into the grid
// ---------------------------------------------------------------------------
//   Standard metadynamics adds a fixed-height Gaussian at the current CV value s*:
//       g(s) = w * exp(-(s - s*)^2 / (2 sigma^2)).
//   WELL-TEMPERED metadynamics SHRINKS each new hill by an exponential factor
//   that depends on how much bias has ALREADY piled up at s*:
//       w_eff = w * exp( -V_bias(s*) / (kT * (gamma - 1)) ).
//   As bias accumulates, w_eff -> 0, so the bias converges (it stops growing) and
//   the recovered FES is  F(s) = -(gamma/(gamma-1)) * V_bias(s) = -(1 + 1/(gamma-1)) V.
//   This guaranteed convergence is why well-tempered MetaD is the default choice.
//
//   We add the (tempered) Gaussian to EVERY grid point -- O(nbins) per deposit,
//   but deposits are infrequent (every `deposit_every` steps), so it is cheap.
// ===========================================================================
METAD_HD inline void deposit_hill(const Model& m, double* bias, double s_star) {
    // Well-tempered down-scaling of this hill's height, using the bias already
    // present at s_star. (gamma - 1) in the denominator; gamma = bias_factor.
    const double V_here = bias_value(m, bias, s_star);
    const double w_eff  = m.hill_w * std::exp(-V_here / (m.kT * (m.bias_factor - 1.0)));
    const double inv2s2 = 1.0 / (2.0 * m.hill_sigma * m.hill_sigma);

    for (int j = 0; j < m.nbins; ++j) {
        const double sj = m.s_lo + j * grid_ds(m);    // CV value at grid point j
        const double d  = sj - s_star;                // distance from hill centre
        bias[j] += w_eff * std::exp(-d * d * inv2s2); // add the Gaussian hill
    }
}

// ===========================================================================
// 4. DETERMINISTIC RNG:  a counter-based hash (no per-thread state to carry)
// ---------------------------------------------------------------------------
//   Langevin dynamics needs Gaussian random forces. For CPU/GPU PARITY and full
//   reproducibility we must generate the *same* noise sequence on both sides,
//   independent of thread scheduling. A COUNTER-BASED RNG is ideal: instead of
//   evolving a hidden state, it HASHES an explicit counter -> the n-th random
//   number is a pure function of (seed, walker, step, draw). Same inputs ->
//   same bits, on host and device, in any order. We use a small SplitMix64-style
//   finaliser, which has good avalanche and is exactly representable in uint64.
// ===========================================================================

// Mix a 64-bit integer into a well-scrambled 64-bit hash (SplitMix64 finalizer).
METAD_HD inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;                 // odd constant (golden ratio)
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// Turn a 64-bit hash into a double in [0,1) using the top 53 bits (the mantissa
// width of a double), which is the standard unbiased construction.
METAD_HD inline double u64_to_unit(uint64_t h) {
    return (h >> 11) * (1.0 / 9007199254740992.0);   // 2^53
}

// One standard-normal sample N(0,1) for (walker, step), drawn DETERMINISTICALLY.
//   We build two independent uniforms from two hashes of the same (seed,walker,
//   step) counter (with different salts), then Box-Muller transforms them into a
//   Gaussian. Using only the first Box-Muller output keeps the mapping
//   step -> noise a clean pure function (we trade one wasted cos() for clarity).
//   u1 is nudged off exactly 0 so log(u1) is finite.
METAD_HD inline double gaussian(uint64_t seed, int walker, int step) {
    // Compose a unique 64-bit counter from the three indices. The large odd
    // multipliers keep walker/step contributions from colliding.
    uint64_t c1 = seed ^ (uint64_t)(walker) * 0xD1B54A32D192ED03ULL
                       ^ (uint64_t)(step)   * 0x2545F4914F6CDD1DULL;
    uint64_t c2 = c1 ^ 0xA0761D6478BD642FULL;       // second salt for the 2nd uniform
    double u1 = u64_to_unit(splitmix64(c1));
    double u2 = u64_to_unit(splitmix64(c2));
    if (u1 < 1e-300) u1 = 1e-300;                    // guard log(0)
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(TWO_PI * u2);
}

// ===========================================================================
// 5. THE LANGEVIN INTEGRATOR (BAOAB-style velocity update) WITH METADYNAMICS BIAS
// ---------------------------------------------------------------------------
//   The walker obeys the Langevin equation
//       m dv/dt = f_total(s) - m*gamma_L*v + sqrt(2 m gamma_L kT) * xi(t),
//   where f_total = well_force(s) + bias_force(s),  bias_force = -dV_bias/ds.
//   We use a simple, stable splitting (a leapfrog-like velocity Verlet with an
//   Ornstein-Uhlenbeck thermostat half-step on the velocity). The exact
//   discretization is less important than the fact that BOTH sides run it
//   identically; we keep it deterministic and double precision.
//
//   This function advances ONE step in place and returns nothing; the caller
//   tracks when to deposit hills. Thread-to-data: one walker == one call-loop.
// ===========================================================================
struct Walker {
    double s;   // collective variable (position) -- this is what we histogram
    double v;   // CV velocity (momentum / mass)
};

// Advance a walker by one Langevin step under the combined (well + bias) force.
//   m         : the model/thermostat parameters
//   w         : the walker state (modified in place)
//   bias      : the walker's accumulated bias grid (read-only here)
//   seed,id,step : feed the deterministic Gaussian noise generator
METAD_HD inline void langevin_step(const Model& m, Walker& w, const double* bias,
                                   uint64_t seed, int id, int step) {
    // Ornstein-Uhlenbeck thermostat coefficients for a friction*dt half-step.
    //   c1 = exp(-gamma_L * dt/2)   (velocity damping over half a step)
    //   c2 = sqrt((1 - c1^2) * kT/m) (matching fluctuation, fluctuation-dissipation)
    const double c1 = std::exp(-m.friction * m.dt * 0.5);
    const double c2 = std::sqrt((1.0 - c1 * c1) * m.kT / m.mass);

    // --- (1) half thermostat kick on velocity: damp + add Gaussian noise ---
    w.v = c1 * w.v + c2 * gaussian(seed, id, 2 * step + 0);

    // --- (2) half deterministic kick from the total force (well + bias) ----
    double f = well_force(m, w.s) - bias_grad(m, bias, w.s);   // f_total at current s
    w.v += 0.5 * m.dt * f / m.mass;

    // --- (3) full drift: move the CV with the updated velocity -------------
    w.s += m.dt * w.v;

    // --- (4) second half force kick at the new position --------------------
    f = well_force(m, w.s) - bias_grad(m, bias, w.s);
    w.v += 0.5 * m.dt * f / m.mass;

    // --- (5) second half thermostat kick (symmetric splitting) -------------
    w.v = c1 * w.v + c2 * gaussian(seed, id, 2 * step + 1);
}

// ===========================================================================
// 6. RUN ONE FULL WALKER:  Langevin + periodic hill deposition + FES recovery
// ---------------------------------------------------------------------------
//   This is the routine the CPU reference LOOPS over (one call per walker) and
//   the GPU kernel runs ONCE PER THREAD. It integrates the walker for m.steps,
//   depositing a tempered hill every m.deposit_every steps, and fills in the
//   per-walker bias grid. The grid IS the result: F(s) = -(gamma/(gamma-1)) V_bias.
//
//   `bias` must point to m.nbins doubles, pre-zeroed. We also return a small
//   summary so main.cu can report deterministic numbers and verify CPU==GPU.
// ===========================================================================
struct WalkerResult {
    int    n_hills;        // how many hills this walker deposited
    int    n_crossings;    // times the walker crossed the barrier (sign change of s)
    double final_s;        // final CV value (for a deterministic spot-check)
    double max_bias;       // peak of the accumulated bias grid (energy)
};

// Integrate a single walker to completion, filling `bias` and returning a summary.
//   s0 : initial CV position (e.g. start in the left well at s=-1).
METAD_HD inline WalkerResult run_walker(const Model& m, double* bias,
                                        uint64_t seed, int id, double s0) {
    // Zero the bias grid (the walker starts with no accumulated bias).
    for (int j = 0; j < m.nbins; ++j) bias[j] = 0.0;

    Walker w{ s0, 0.0 };          // start at rest in the chosen well
    int n_hills = 0;
    int n_cross = 0;
    double prev_sign = (w.s >= 0.0) ? 1.0 : -1.0;   // which well we are in

    for (int step = 0; step < m.steps; ++step) {
        langevin_step(m, w, bias, seed, id, step);

        // Count barrier crossings: a sign change of s means we moved wells. This
        // is the headline "did enhanced sampling work?" diagnostic -- plain MD
        // would essentially never cross within m.steps.
        double sign = (w.s >= 0.0) ? 1.0 : -1.0;
        if (sign != prev_sign) { ++n_cross; prev_sign = sign; }

        // Deposit a tempered Gaussian hill on the metadynamics "pace".
        if ((step % m.deposit_every) == 0) {
            deposit_hill(m, bias, w.s);
            ++n_hills;
        }
    }

    // Find the peak of the accumulated bias (a deterministic scalar summary).
    double max_bias = 0.0;
    for (int j = 0; j < m.nbins; ++j)
        if (bias[j] > max_bias) max_bias = bias[j];

    WalkerResult out;
    out.n_hills     = n_hills;
    out.n_crossings = n_cross;
    out.final_s     = w.s;
    out.max_bias    = max_bias;
    return out;
}

// ===========================================================================
// 7. FES RECOVERY:  turn the accumulated bias into a free-energy estimate
// ---------------------------------------------------------------------------
//   Well-tempered reconstruction:  F_est(s) = -(gamma/(gamma-1)) * V_bias(s),
//   shifted so its minimum is zero (free energies are defined up to a constant).
//   The caller compares F_est against the known true_fes() over the grid -- the
//   "did metadynamics recover the physics?" check (the science validation, not
//   just CPU==GPU agreement). We write F_est into `fes_out` (length m.nbins).
// ===========================================================================
METAD_HD inline void recover_fes(const Model& m, const double* bias, double* fes_out) {
    const double scale = m.bias_factor / (m.bias_factor - 1.0);   // gamma/(gamma-1)
    double fmin = 1e300;
    for (int j = 0; j < m.nbins; ++j) {
        fes_out[j] = -scale * bias[j];
        if (fes_out[j] < fmin) fmin = fes_out[j];
    }
    // Shift so the lowest point is exactly 0 (align with true_fes minima = 0).
    for (int j = 0; j < m.nbins; ++j) fes_out[j] -= fmin;
}

}  // namespace metad
