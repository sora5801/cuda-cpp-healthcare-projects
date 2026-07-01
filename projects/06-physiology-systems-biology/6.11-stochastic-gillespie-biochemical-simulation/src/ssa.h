// ===========================================================================
// src/ssa.h  --  Shared (host + device) reaction network + Gillespie SSA core
// ---------------------------------------------------------------------------
// Project 6.11 : Stochastic (Gillespie) Biochemical Simulation
//
// WHAT THIS PROJECT COMPUTES
//   An ENSEMBLE of exact stochastic trajectories of a well-mixed chemical
//   reaction network, using the Gillespie Stochastic Simulation Algorithm (SSA,
//   "direct method"). When molecule counts are small (a handful of mRNAs, a
//   few dozen transcription factors) the smooth deterministic ODE (mass-action
//   kinetics) is the WRONG model: reactions fire one at a time, at random
//   times, and the copy number fluctuates. The SSA samples that randomness
//   EXACTLY -- it is a statistically exact realisation of the Chemical Master
//   Equation (CME). To get statistics (means, variances, distributions) you run
//   MANY independent trajectories and average. Each trajectory is independent,
//   so the natural GPU mapping is ONE TRAJECTORY PER THREAD (see kernels.cu).
//
// WHY THIS HEADER IS SHARED (the HD-macro idiom, PATTERNS.md section 2)
//   The whole verification strategy is: the CPU reference and the GPU kernel
//   run the *identical* trajectories, so their per-trajectory summaries must
//   match EXACTLY (to the last bit). That only works if BOTH sides use the same
//   random-number generator and the same reaction-selection logic. So all of
//   that lives HERE, in one header included by reference_cpu.cpp (host compiler)
//   AND by kernels.cu / main.cu (nvcc). SSA_HD expands to `__host__ __device__`
//   under nvcc and to nothing under the plain host compiler.
//
//   NOTE ON cuRAND. The catalog entry mentions cuRAND (one stream per thread).
//   Production GPU-SSA codes do use cuRAND for speed. We deliberately use a
//   SHARED counter-based RNG (splitmix64) instead, precisely so the CPU and GPU
//   draw the *same* random numbers and the demo is bit-for-bit reproducible and
//   exactly verifiable. THEORY.md section "Where this sits in the real world"
//   explains the cuRAND swap and what it would cost you in reproducibility.
//
// THE ALGORITHM (Gillespie direct method), for state x = molecule counts:
//   repeat until t >= t_end (or no reaction can fire):
//     1. compute propensity a_j(x) for each reaction j  (mass-action)
//     2. a0 = sum_j a_j                                  (total event rate)
//     3. tau = -ln(u1) / a0                              (time to next event;
//                                                         exponential waiting)
//     4. pick reaction k with P(k) = a_k / a0            (roulette on u2*a0)
//     5. apply reaction k's stoichiometry: x += nu_k;  t += tau
//   This is EXACT: no time step, no discretisation error -- every event is a
//   real reaction firing at a correctly-distributed random time.
//
// READ THIS AFTER: util/cuda_check.cuh.  READ BEFORE: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#pragma once

#include <cstdint>   // uint64_t, fixed-width integer molecule counts
#include <cmath>     // log()

// SSA_HD: decorate the shared inline functions so they compile for BOTH the
// host (reference_cpu.cpp via cl.exe/g++) and the device (kernels.cu via nvcc).
#ifdef __CUDACC__
#define SSA_HD __host__ __device__
#else
#define SSA_HD
#endif

// Compile-time caps on network size. Keeping these fixed (rather than dynamic)
// lets a whole trajectory's working set (state + stoichiometry + rates) live in
// registers / local memory on the GPU with no dynamic allocation -- exactly what
// we want for the "one lightweight thread per trajectory" pattern.
#define SSA_MAX_SPECIES   4    // number of chemical species tracked
#define SSA_MAX_REACTIONS 6    // number of reactions in the network

// ---------------------------------------------------------------------------
// Rng: a splitmix64 counter-based random stream.
//   splitmix64 is a tiny, fast, well-tested 64-bit mixer. It is fully
//   deterministic and identical on host and device (only integer ops + shifts),
//   which is what makes CPU and GPU trajectories bit-identical. Each trajectory
//   gets its OWN independent stream, seeded from (base_seed, trajectory_id).
// ---------------------------------------------------------------------------
struct Rng { uint64_t state; };

// One splitmix64 step: advance the state and return a well-mixed 64-bit value.
SSA_HD inline uint64_t splitmix64(uint64_t& x) {
    x += 0x9E3779B97F4A7C15ULL;                  // golden-ratio odd increment
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; // avalanche mixing constants
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Seed an independent stream for trajectory `traj` from a global base seed.
//   Mixing the trajectory index into the seed guarantees that trajectory 0 and
//   trajectory 1 are uncorrelated, yet every run reproduces the same streams.
SSA_HD inline Rng rng_seed(uint64_t base, uint64_t traj) {
    Rng r;
    r.state = base ^ (traj * 0x9E3779B97F4A7C15ULL + 0xD1B54A32D192ED03ULL);
    splitmix64(r.state);   // one warm-up step so nearby seeds diverge immediately
    return r;
}

// Uniform double in [0,1) built from 53 random bits (identical math host/device).
//   53 bits = the mantissa width of a double, so this evenly fills [0,1).
SSA_HD inline double rng_uniform(Rng& r) {
    uint64_t z = splitmix64(r.state);
    return (z >> 11) * (1.0 / 9007199254740992.0);   // multiply by 2^-53
}

// ---------------------------------------------------------------------------
// ReactionNetwork: a well-mixed mass-action network in a fixed-size, POD form.
//   * n_species / n_reactions: how many of the SSA_MAX_* slots are in use.
//   * x0[s]        : initial molecule count of species s (integer).
//   * k[j]         : the rate constant of reaction j (units depend on order).
//   * order[j]     : reaction order 0/1/2 -- selects the mass-action formula.
//   * reactant1[j] : index of the (first) reactant species of reaction j,
//                    or -1 if the reaction has no reactant of that slot.
//   * reactant2[j] : index of the second reactant species (for 2nd-order), else -1.
//   * nu[j][s]     : stoichiometric change of species s when reaction j fires
//                    (e.g. -1 for a consumed reactant, +1 for a product).
//   POD (plain-old-data, no pointers) so the WHOLE struct is passed BY VALUE to
//   the kernel -- it rides along in constant/parameter memory, no cudaMalloc for
//   the network itself. See kernels.cu.
// ---------------------------------------------------------------------------
struct ReactionNetwork {
    int    n_species;
    int    n_reactions;
    uint64_t x0[SSA_MAX_SPECIES];                        // initial counts
    double k[SSA_MAX_REACTIONS];                         // rate constants
    int    order[SSA_MAX_REACTIONS];                     // 0, 1, or 2
    int    reactant1[SSA_MAX_REACTIONS];                 // species index or -1
    int    reactant2[SSA_MAX_REACTIONS];                 // species index or -1
    long   nu[SSA_MAX_REACTIONS][SSA_MAX_SPECIES];       // stoichiometry matrix
    double t_end;                                        // simulation end time
    uint64_t base_seed;                                  // global RNG base seed
};

// ---------------------------------------------------------------------------
// propensity: the mass-action rate a_j(x) at which reaction j fires *right now*.
//   Mass action counts the number of distinct reactant combinations available:
//     order 0 (source):        a = k                 (e.g. 0 -> X, constant birth)
//     order 1 (A -> ...):      a = k * x_A           (each A can react)
//     order 2, A + B:          a = k * x_A * x_B     (distinct A-B pairs)
//     order 2, A + A (dimer):  a = k * x_A*(x_A-1)/2 (distinct unordered pairs)
//   We detect the homodimer case (reactant1 == reactant2) and use the n*(n-1)/2
//   combinatorial count, which is the correct discrete-molecule form.
// ---------------------------------------------------------------------------
SSA_HD inline double propensity(const ReactionNetwork& net, int j,
                                const uint64_t* x) {
    const double k = net.k[j];
    const int r1 = net.reactant1[j];
    const int r2 = net.reactant2[j];
    switch (net.order[j]) {
        case 0:
            return k;                                    // zeroth order: constant
        case 1:
            return k * static_cast<double>(x[r1]);       // first order: k * x_A
        case 2:
            if (r1 == r2) {
                // Homodimer A + A: number of unordered pairs = n*(n-1)/2.
                const double n = static_cast<double>(x[r1]);
                return k * n * (n - 1.0) * 0.5;
            }
            // Heterodimer A + B: number of A-B pairs = x_A * x_B.
            return k * static_cast<double>(x[r1]) * static_cast<double>(x[r2]);
        default:
            return 0.0;                                  // unreachable by construction
    }
}

// ---------------------------------------------------------------------------
// TrajectoryResult: the deterministic per-trajectory summary the analysis wants.
//   Chosen so the ensemble MEAN of `time_avg` recovers a KNOWN analytic quantity
//   (the CME stationary mean), giving us a science check on top of CPU==GPU.
//     * final_count[s]  : molecule count of species s at t = t_end.
//     * time_avg[s]     : time-weighted average of species s over [0, t_end],
//                         i.e. (1/T) * integral_0^T x_s(t) dt. Because x is a
//                         step function (constant between events), this integral
//                         is an EXACT finite sum -- no quadrature error.
//     * n_events        : how many reactions fired (a stochastic count).
// ---------------------------------------------------------------------------
struct TrajectoryResult {
    uint64_t final_count[SSA_MAX_SPECIES];
    double   time_avg[SSA_MAX_SPECIES];
    uint64_t n_events;
};

// ---------------------------------------------------------------------------
// simulate_trajectory: run ONE exact Gillespie SSA trajectory to t_end.
//   Shared by the CPU reference and the GPU kernel (one thread calls it once).
//   `traj` selects this trajectory's independent RNG stream, so the CPU's
//   trajectory 7 and the GPU's trajectory 7 draw the SAME randoms -> identical
//   result. Returns a fully-populated TrajectoryResult.
//
//   Complexity: O(n_events * n_reactions). n_reactions is tiny (<= 6), and
//   n_events scales with the copy numbers and t_end. There is no time-step loop:
//   the loop advances event-by-event, which is what makes the method exact.
// ---------------------------------------------------------------------------
SSA_HD inline TrajectoryResult simulate_trajectory(const ReactionNetwork& net,
                                                   uint64_t traj) {
    // Local working state: current molecule counts, copied from the initial x0.
    uint64_t x[SSA_MAX_SPECIES];
    double   integ[SSA_MAX_SPECIES];   // running time-integral of each x_s
    for (int s = 0; s < net.n_species; ++s) {
        x[s] = net.x0[s];
        integ[s] = 0.0;
    }

    Rng rng = rng_seed(net.base_seed, traj);   // this trajectory's own stream
    double t = 0.0;                            // simulation clock
    uint64_t events = 0;                       // reactions fired so far

    // Guard against a pathological runaway (e.g. an explosive network); a real
    // study would size this to the expected event count. 5,000,000 is ample for
    // the teaching models here and keeps every GPU thread bounded.
    const uint64_t MAX_EVENTS = 5000000ULL;

    while (t < net.t_end && events < MAX_EVENTS) {
        // --- 1. propensities and their total ------------------------------
        double a[SSA_MAX_REACTIONS];
        double a0 = 0.0;
        for (int j = 0; j < net.n_reactions; ++j) {
            a[j] = propensity(net, j, x);
            a0 += a[j];
        }

        // If nothing can react (a0 == 0), the system is frozen: fast-forward the
        // clock to t_end and stop. (Its state stays constant, so the time
        // integral just accumulates x*(remaining time).)
        if (a0 <= 0.0) {
            const double dt = net.t_end - t;
            for (int s = 0; s < net.n_species; ++s)
                integ[s] += static_cast<double>(x[s]) * dt;
            t = net.t_end;
            break;
        }

        // --- 2. time to the next event: exponential(a0) -------------------
        // tau = -ln(u1)/a0. Use (1 - u) in (0,1] so we never take log(0).
        const double u1 = 1.0 - rng_uniform(rng);
        double tau = -log(u1) / a0;

        // Do not overshoot the horizon: if the next event is after t_end, the
        // state is unchanged over the remaining window -- integrate and stop.
        if (t + tau > net.t_end) {
            const double dt = net.t_end - t;
            for (int s = 0; s < net.n_species; ++s)
                integ[s] += static_cast<double>(x[s]) * dt;
            t = net.t_end;
            break;
        }

        // Accumulate the time integral over [t, t+tau] where x is constant.
        for (int s = 0; s < net.n_species; ++s)
            integ[s] += static_cast<double>(x[s]) * tau;

        // --- 3. choose which reaction fires: roulette on u2 * a0 ----------
        // Walk the cumulative propensities; the first bin whose running sum
        // exceeds the threshold is the chosen reaction k. This is O(n_reactions)
        // linear search -- fine for tiny networks; a prefix-sum + binary search
        // (Thrust, per the catalog) wins only for large reaction sets.
        double thresh = rng_uniform(rng) * a0;
        int k = net.n_reactions - 1;   // default to last (covers rounding at the top)
        double cum = 0.0;
        for (int j = 0; j < net.n_reactions; ++j) {
            cum += a[j];
            if (thresh < cum) { k = j; break; }
        }

        // --- 4. apply reaction k's stoichiometry --------------------------
        for (int s = 0; s < net.n_species; ++s) {
            // nu can be negative (reactant consumed); cast the count to signed,
            // add, and clamp at 0 so integer underflow can never wrap uint64.
            long long v = static_cast<long long>(x[s]) + net.nu[k][s];
            x[s] = (v < 0) ? 0ULL : static_cast<uint64_t>(v);
        }

        // --- 5. advance the clock and event counter -----------------------
        t += tau;
        ++events;
    }

    // Package the summary. time_avg = integral / T (T = t_end); this is the
    // exact time-average of a piecewise-constant trajectory.
    TrajectoryResult out;
    out.n_events = events;
    const double T = net.t_end;
    for (int s = 0; s < net.n_species; ++s) {
        out.final_count[s] = x[s];
        out.time_avg[s] = (T > 0.0) ? integ[s] / T : static_cast<double>(x[s]);
    }
    return out;
}
