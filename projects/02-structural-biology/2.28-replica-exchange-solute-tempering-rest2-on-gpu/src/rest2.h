// ===========================================================================
// src/rest2.h  --  Shared (host + device) REST2 physics core
// ---------------------------------------------------------------------------
// Project 2.28 : Replica Exchange Solute Tempering (REST2) on GPU
//
// WHAT THIS PROJECT COMPUTES (the short version -- full story in ../THEORY.md)
//   REST2 ("Replica Exchange with Solute Tempering, version 2") is an ENHANCED
//   SAMPLING method. Plain molecular dynamics at body temperature gets trapped
//   in one conformational basin for a very long time; to cross energy barriers
//   you would normally heat the whole system -- but heating thousands of water
//   molecules is wasteful. REST2's trick: heat ONLY the solute (the protein /
//   ligand) by SCALING its slice of the potential energy, and run a LADDER of
//   replicas at different effective solute temperatures that periodically SWAP
//   configurations. The hot replicas hop barriers; the swaps funnel that
//   diversity down to the cold (physical, 300 K) replica.
//
//   To make this a SELF-CONTAINED, EXACTLY-VERIFIABLE teaching version we do NOT
//   run full explicit-solvent MD. Instead we keep the *exact* REST2 mathematics
//   (the three-way energy split, the lambda scaling, the Metropolis exchange
//   criterion) on a tiny toy system sampled by Metropolis MONTE CARLO. The MC
//   sampler is the standard didactic stand-in for MD here: it removes forces and
//   thermostats (which add non-determinism) and lets us focus on REST2 itself.
//   Everything that is "REST2" -- the Hamiltonian scaling and the swap rule -- is
//   faithful to the real method (THEORY.md "Where this sits in the real world").
//
// THE ENERGY DECOMPOSITION (the heart of REST2)
//   The total potential of a solvated system splits into three physically
//   distinct groups of interactions:
//       E_pp  = solute-solute   ("protein-protein", the solute's internal energy)
//       E_pw  = solute-solvent  ("protein-water")
//       E_ww  = solvent-solvent ("water-water")
//   REST2 builds a per-replica EFFECTIVE energy by scaling only the first two:
//       E_eff(lambda) = lambda * E_pp + sqrt(lambda) * E_pw + E_ww
//   where lambda_m = beta_m / beta_0 = T_0 / T_m in (0,1]. The cold replica has
//   lambda = 1 (nothing scaled -> true physics); hotter replicas have lambda < 1
//   (solute energies shrunk -> barriers look smaller -> easier hopping). The
//   sqrt on the cross term is the REST2-v2 fix that makes the math consistent
//   with rescaling the solute's charges/epsilon by sqrt(lambda) (see THEORY.md).
//
// HOW CPU AND GPU STAY BIT-IDENTICAL (PATTERNS.md sections 2 & 4)
//   Every formula a replica needs -- the energies, the effective energy, the RNG,
//   and ONE Monte-Carlo sweep -- lives here as `__host__ __device__` inline
//   functions. The CPU reference (reference_cpu.cpp) loops them; the GPU kernel
//   (kernels.cu) calls the SAME functions from one thread per replica. Because
//   the operations are identical (and the RNG is a deterministic counter-based
//   hash, not a stateful library generator), the two sides produce EXACTLY the
//   same configurations and acceptance counts -> we verify with tolerance 0.
//
// READ THIS AFTER: util/cuda_check.cuh.   READ BEFORE: reference_cpu.h, kernels.cuh.
// ===========================================================================
#pragma once

#include <cstdint>   // uint32_t / uint64_t for the counter-based RNG
#include <cmath>     // std::sqrt, std::exp, std::cos (host side)

// HD = "host+device". Under nvcc this expands to __host__ __device__ so the
// function compiles for BOTH targets; under the plain host compiler (which
// builds reference_cpu.cpp) it expands to nothing. This is the single most
// useful idiom in the repo (PATTERNS.md section 2): one source of truth for the
// physics means CPU and GPU cannot silently diverge. Keep this header free of
// CUDA-only types (no __global__, no <cuda_runtime.h>) so cl.exe can include it.
#ifdef __CUDACC__
#define REST2_HD __host__ __device__
#else
#define REST2_HD
#endif

// ---------------------------------------------------------------------------
// Model dimensions (compile-time so the per-replica state lives in registers).
//   N_SOLUTE : number of coarse-grained solute beads (a tiny "protein"). Each
//              bead has a 1-D coordinate x in our toy landscape -- enough to
//              exhibit a real double-well barrier that enhanced sampling must
//              cross, while staying small enough to keep the whole replica state
//              in registers (no global-memory traffic inside the MC loop).
// ---------------------------------------------------------------------------
static constexpr int N_SOLUTE = 8;   // solute beads (toy "protein" chain length)

// ---------------------------------------------------------------------------
// Counter-based RNG: squares64 / a SplitMix-style hash.
//   Stateful generators (curand, std::mt19937) carry hidden state that makes a
//   thread's stream depend on history -- bad for reproducibility across CPU/GPU
//   and across re-runs. Instead we DERIVE every random number from an explicit
//   (key, counter) pair by hashing it. Same inputs -> same bits on every device,
//   every run. The "counter" is built from (replica, step, sub-stream) so no two
//   draws ever collide. This is exactly how RNGs like Philox/Random123 work.
//
//   rng_hash64: a strong 64-bit integer hash (mix constants from SplitMix64).
// ---------------------------------------------------------------------------
REST2_HD inline uint64_t rng_hash64(uint64_t z) {
    // Three xor-shift / multiply rounds avalanche the bits so adjacent counters
    // map to uncorrelated outputs (a tiny, well-tested finalizer).
    z += 0x9E3779B97F4A7C15ULL;                  // golden-ratio increment
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; // mix high bits down
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; // mix again
    return z ^ (z >> 31);                         // final fold
}

// Turn a (key, counter) pair into a double uniformly in [0, 1).
//   key      : the per-replica RNG seed (distinct per replica -> independent
//              streams), counter : a unique draw index. We hash key XORed with a
//              shuffled counter so neither argument alone controls the output.
//   We take the top 53 bits (the mantissa width of a double) and divide by 2^53,
//   the standard way to get a uniform double with full precision and no bias.
REST2_HD inline double rng_uniform(uint64_t key, uint64_t counter) {
    uint64_t h = rng_hash64(key ^ rng_hash64(counter));  // combine both inputs
    // 0x1.0p-53 == 2^-53. Multiplying the high 53 bits by this lands in [0,1).
    return (double)(h >> 11) * (1.0 / 9007199254740992.0);
}

// ---------------------------------------------------------------------------
// Toy energy landscape for one solute bead.
//   Each bead sits in a symmetric DOUBLE-WELL potential
//       u(x) = h * (x^2 - 1)^2
//   with minima near x = -1 and x = +1 separated by a barrier of height `h` at
//   x = 0. This is the canonical 1-D model of a conformational switch (e.g. a
//   dihedral flipping between two rotamers): sampling BOTH wells requires
//   crossing the barrier, which is exactly what enhanced sampling buys you.
//
//   We add a small linear TILT (-tilt*x) that lowers the RIGHT well, making it
//   the GLOBAL free-energy minimum and the left well merely METASTABLE. This
//   gives the demo an unambiguous "correct answer": the equilibrium population
//   should sit in the right well, but a 300 K replica started on the LEFT will
//   stay trapped there for a very long time unless it can cross the barrier --
//   precisely the situation REST2 is designed to rescue. `barrier_h` and `tilt`
//   are read from the input so the demo can dial the difficulty. This is the
//   per-bead "internal" (solute-solute) contribution.
// ---------------------------------------------------------------------------
REST2_HD inline double well_energy(double x, double barrier_h, double tilt) {
    const double t = x * x - 1.0;       // 0 at the minima, 1 at the barrier top
    return barrier_h * t * t - tilt * x; // tilted quartic double well (right = global min)
}

// ---------------------------------------------------------------------------
// The three REST2 energy groups for a full solute configuration.
//   We model:
//     E_pp (solute internal) = sum of each bead's double-well energy PLUS a weak
//                              harmonic "bond" coupling neighbouring beads (so
//                              the chain has internal structure, like a protein).
//     E_pw (solute-solvent)  = a coupling of each bead to an implicit solvent
//                              field: k_pw * (x - x_solvent)^2 style restraint.
//                              This is the term REST2 scales by sqrt(lambda).
//     E_ww (solvent-solvent) = a CONSTANT background here (our implicit solvent
//                              has no internal degrees of freedom to move). REST2
//                              never scales it, so for sampling the solute its
//                              value is irrelevant -- we keep it explicit only to
//                              show *where it would go* in the effective energy.
//   x[] : the N_SOLUTE bead coordinates.  Returns the three groups by reference.
// ---------------------------------------------------------------------------
REST2_HD inline void rest2_energies(const double* x, double barrier_h, double tilt,
                                    double k_bond, double k_pw, double x_solvent,
                                    double& E_pp, double& E_pw, double& E_ww) {
    double pp = 0.0;   // accumulate solute-solute energy
    double pw = 0.0;   // accumulate solute-solvent energy
    // Per-bead double-well + coupling to the solvent field.
    for (int i = 0; i < N_SOLUTE; ++i) {
        pp += well_energy(x[i], barrier_h, tilt);    // internal (tilted) double well
        const double d = x[i] - x_solvent;           // displacement from solvent
        pw += k_pw * d * d;                           // solute-solvent restraint
    }
    // Weak harmonic bonds between neighbouring beads (a 1-D bead-spring chain):
    // gives the solute internal correlations a single bead would not have.
    for (int i = 0; i + 1 < N_SOLUTE; ++i) {
        const double b = x[i + 1] - x[i];            // bond stretch
        pp += k_bond * b * b;                         // bond energy -> still solute-solute
    }
    E_pp = pp;
    E_pw = pw;
    E_ww = 0.0;   // implicit solvent: constant internal energy (drops out of MC)
}

// ---------------------------------------------------------------------------
// The REST2 EFFECTIVE energy of a configuration in a replica with parameter
// lambda in (0,1]:
//       E_eff = lambda * E_pp + sqrt(lambda) * E_pw + E_ww
//   lambda = 1 -> physical energy (cold replica). lambda < 1 -> solute and
//   solute-solvent terms shrink, flattening barriers so the hot replica explores
//   freely. This single function is what makes the replicas different; sampling
//   in replica m uses E_eff(lambda_m) as its Boltzmann energy.
// ---------------------------------------------------------------------------
REST2_HD inline double rest2_effective(double lambda, double E_pp, double E_pw, double E_ww) {
    // sqrt(lambda) on the cross term is the REST2-v2 correction (THEORY.md "math").
    return lambda * E_pp + sqrt(lambda) * E_pw + E_ww;
}

// Convenience: effective energy directly from coordinates (used in the MC step).
REST2_HD inline double rest2_effective_x(const double* x, double lambda,
                                         double barrier_h, double tilt, double k_bond,
                                         double k_pw, double x_solvent) {
    double pp, pw, ww;
    rest2_energies(x, barrier_h, tilt, k_bond, k_pw, x_solvent, pp, pw, ww);
    return rest2_effective(lambda, pp, pw, ww);
}

// ---------------------------------------------------------------------------
// Per-replica fixed parameters. Bundled in a struct so the kernel can take one
// argument per replica and the host can build a ladder of them.
// ---------------------------------------------------------------------------
struct ReplicaParams {
    double lambda;     // REST2 scaling = T_0 / T_m in (0,1]; 1 = cold/physical
    uint64_t seed;     // RNG key for THIS replica's independent MC stream
};

// Global (shared by all replicas) physics + run settings.
struct SimConfig {
    int    n_replicas = 0;     // number of replicas in the temperature ladder
    int    sweeps_per_round = 0;  // MC sweeps between exchange attempts
    int    n_rounds = 0;       // number of (sample, then attempt-exchange) rounds
    double barrier_h = 0.0;    // double-well barrier height (energy units, kT=1)
    double tilt = 0.0;         // linear bias -tilt*x; lowers the RIGHT well (global min)
    double k_bond = 0.0;       // bead-bead bond stiffness (solute internal)
    double k_pw = 0.0;         // solute-solvent coupling stiffness
    double x_solvent = 0.0;    // position of the implicit solvent field
    double step_size = 0.0;    // Monte-Carlo trial move half-width (in x units)
    double T0 = 0.0;           // physical temperature of the cold replica
    double Tmax = 0.0;         // effective solute temperature of the hottest replica
};

// ---------------------------------------------------------------------------
// One Metropolis MONTE-CARLO SWEEP of a single replica.
//   A "sweep" proposes a small random displacement to EACH bead in turn and
//   accepts/rejects it by the Metropolis rule using the replica's EFFECTIVE
//   energy E_eff(lambda). This is the per-replica "MD" of our teaching version:
//   it samples the Boltzmann distribution of E_eff at inverse temperature 1
//   (we fold the temperature into lambda, so every replica samples at kT = 1).
//
//   Arguments:
//     x[]        : in/out solute coordinates (updated in place).
//     p          : this replica's (lambda, seed).
//     cfg        : shared physics + step size.
//     rng_counter: in/out running draw index -> guarantees every random number
//                  (trial displacement AND acceptance roll) is unique, so the
//                  stream never repeats. Passing it by reference threads the
//                  deterministic RNG through the whole simulation.
//   Returns the number of accepted moves this sweep (for the acceptance-ratio
//   diagnostic). Integer return -> exactly reproducible reduction later.
//
//   WHY METROPOLIS: proposing dx and accepting with prob min(1, exp(-dE)) leaves
//   the Boltzmann distribution invariant (detailed balance). It needs only energy
//   DIFFERENCES, no forces -- ideal for a transparent, force-free teaching code.
// ---------------------------------------------------------------------------
REST2_HD inline int mc_sweep(double* x, const ReplicaParams& p,
                             const SimConfig& cfg, uint64_t& rng_counter) {
    int accepted = 0;                                   // moves accepted this sweep
    for (int i = 0; i < N_SOLUTE; ++i) {
        // --- propose: displace bead i by a uniform move in [-step, +step] -----
        // Two fresh random draws: one for the displacement, one for the
        // acceptance test. Each consumes a unique counter value.
        const double r_disp = rng_uniform(p.seed, rng_counter++);   // in [0,1)
        const double r_acc  = rng_uniform(p.seed, rng_counter++);   // in [0,1)
        const double dx = (2.0 * r_disp - 1.0) * cfg.step_size;     // [-step,+step]

        // --- evaluate the energy change of moving ONLY bead i -----------------
        // We could recompute the whole energy, but moving one bead changes only
        // its own well, its two bonds, and its solvent term. For clarity (this is
        // teaching code) we recompute the full effective energy before and after;
        // the cost is tiny for N_SOLUTE beads and keeps the logic obvious. The
        // faster "local energy delta" version is left as an exercise (README).
        const double E_old = rest2_effective_x(x, p.lambda, cfg.barrier_h, cfg.tilt,
                                               cfg.k_bond, cfg.k_pw, cfg.x_solvent);
        const double x_save = x[i];      // remember so we can reject cheaply
        x[i] = x_save + dx;              // tentatively apply the move
        const double E_new = rest2_effective_x(x, p.lambda, cfg.barrier_h, cfg.tilt,
                                               cfg.k_bond, cfg.k_pw, cfg.x_solvent);

        // --- Metropolis accept/reject ----------------------------------------
        // dE <= 0 : always accept (downhill). dE > 0 : accept with prob e^{-dE}.
        // (kT = 1 because temperature is folded into lambda inside E_eff.)
        const double dE = E_new - E_old;
        bool accept = (dE <= 0.0) || (r_acc < exp(-dE));
        if (accept) {
            ++accepted;                  // keep the move (x[i] already updated)
        } else {
            x[i] = x_save;               // reject: restore the old coordinate
        }
    }
    return accepted;
}

// ---------------------------------------------------------------------------
// The REST2 EXCHANGE (swap) criterion for a neighbouring pair of replicas m, n.
//   Replicas periodically try to swap their CONFIGURATIONS. Swapping is accepted
//   with the Metropolis rule built from how each configuration's energy changes
//   when evaluated under the OTHER replica's lambda. For REST2 the famous
//   simplification is that only the SOLUTE terms enter (E_ww cancels):
//       Delta = (lambda_m - lambda_n) * (E_pp(n) - E_pp(m))
//             + (sqrt(lambda_m) - sqrt(lambda_n)) * (E_pw(n) - E_pw(m))
//   Accept the swap with probability min(1, exp(-Delta)). A POSITIVE acceptance
//   means the cold replica can inherit a barrier-crossed configuration from a
//   hot one -- that is the whole point.
//
//   Inputs are the two replicas' lambdas and their CURRENT energy groups. We
//   pass energies (not coordinates) because the caller has already computed them
//   and swapping is about energies. Returns Delta; the caller rolls a uniform
//   and compares to exp(-Delta) so the RNG draw stays in the deterministic chain.
// ---------------------------------------------------------------------------
REST2_HD inline double rest2_exchange_delta(double lambda_m, double lambda_n,
                                            double Epp_m, double Epw_m,
                                            double Epp_n, double Epw_n) {
    const double d_pp = (lambda_m - lambda_n) * (Epp_n - Epp_m);                 // internal term
    const double d_pw = (sqrt(lambda_m) - sqrt(lambda_n)) * (Epw_n - Epw_m);     // cross term
    return d_pp + d_pw;   // Delta in the exchange Metropolis factor exp(-Delta)
}
