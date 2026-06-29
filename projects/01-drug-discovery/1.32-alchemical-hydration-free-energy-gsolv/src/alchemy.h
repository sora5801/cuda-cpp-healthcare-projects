// ===========================================================================
// src/alchemy.h  --  Shared (host + device) alchemical physics + MC walker
// ---------------------------------------------------------------------------
// Project 1.32 : Alchemical Hydration Free Energy (delta-G_solv)
//
// WHAT THIS PROJECT COMPUTES
//   The HYDRATION FREE ENERGY delta-G_hyd of a small solute -- the reversible
//   work to move it from vacuum into water. Experimental chemists tabulate this
//   (the FreeSolv database has 643 values); it underlies LogP, solubility, and
//   membrane permeability in ADMET modelling.
//
//   We do it the way real free-energy codes do: ALCHEMICALLY. We define a
//   coupling parameter lambda in [0,1] that smoothly "switches on" the solute's
//   interaction with the solvent. At lambda=1 the solute is fully coupled
//   (solvated); at lambda=0 it is a non-interacting ghost (in vacuum). The free
//   energy of that switch is, by THERMODYNAMIC INTEGRATION (TI),
//
//       delta-G = integral_0^1  <dU/dlambda>_lambda  d-lambda
//
//   where <.>_lambda is a Boltzmann (canonical-ensemble) average of the
//   lambda-derivative of the potential energy, sampled at fixed lambda. Because
//   delta-G_solv = -delta-G(switch on) we report -integral (see THEORY.md).
//
// THE TEACHING REDUCTION (CLAUDE.md section 13)
//   A production calculation runs full GPU molecular dynamics with particle-mesh
//   Ewald electrostatics in a periodic water box -- thousands of lines and a real
//   force field. That is a black box to a learner. Instead we keep EVERY idea
//   that matters (lambda-windows, SOFT-CORE potentials, TI, BAR, the
//   ensemble-over-threads GPU pattern) on a tractable model:
//
//     * the solute is ONE Lennard-Jones + (optional) charged particle,
//     * the "solvent" is a fixed bath of N solvent sites around it,
//     * configurations are sampled by METROPOLIS MONTE CARLO (move the solute,
//       accept/reject by the Boltzmann criterion) instead of MD,
//     * many independent MC WALKERS per lambda-window are the ensemble we run in
//       parallel -- one GPU thread per walker, exactly the pattern from flagship
//       9.02 (ensemble integration).
//
//   This is a real, correct TI/BAR calculation of a model delta-G; it is NOT a
//   force-field-accurate prediction of any experimental number (see THEORY
//   section 7 and the README "Limitations").
//
//   The soft-core energy, its lambda-derivative, the RNG, and the Metropolis
//   walker all live here as ALCH_HD (= __host__ __device__) inline functions, so
//   the CPU reference (reference_cpu.cpp) and the GPU kernel (kernels.cu) run
//   BYTE-FOR-BYTE identical math -> their per-walker results match to round-off.
//
// READ THIS AFTER: nothing (start here), then reference_cpu.h, kernels.cuh.
// ===========================================================================
#pragma once

// ALCH_HD expands to "__host__ __device__" when compiled by nvcc (so the same
// function is emitted for both the CPU and every GPU thread), and to nothing
// under the plain host compiler that builds reference_cpu.cpp. This is the
// CPU/GPU-parity idiom from docs/PATTERNS.md section 2.
#ifdef __CUDACC__
#define ALCH_HD __host__ __device__
#else
#define ALCH_HD
#endif

#include <cstdint>   // uint32_t, uint64_t : the RNG state types
#include <cmath>     // exp, sqrt, pow      : the energy + Metropolis math

// ---------------------------------------------------------------------------
// Physical / model constants. Reduced ("LJ") units throughout: energies are in
// units of epsilon, lengths in units of sigma, temperature in epsilon/k_B. Using
// reduced units keeps the numbers O(1) and the teaching focus on the METHOD
// (TI/BAR), not on unit bookkeeping. THEORY section 2 maps these to kcal/mol.
// ---------------------------------------------------------------------------
struct SystemParams {
    int    n_solvent;     // number of fixed solvent sites surrounding the solute
    double box;           // half-width of the cubic sampling box (solute roams in [-box,box]^3)
    double temperature;   // T in reduced units (epsilon/k_B); beta = 1/T
    double epsilon;       // Lennard-Jones well depth (energy scale)        [eps]
    double sigma;         // Lennard-Jones diameter (length scale)          [sigma]
    double q_solute;      // solute partial charge (Coulomb term, screened) [reduced]
    double alpha_sc;      // soft-core alpha (dimensionless, ~0.5 is standard)
    double max_step;      // Metropolis trial-move max displacement per axis [sigma]
};

// ---------------------------------------------------------------------------
// A solvent bath: n_solvent fixed 3-D sites. We keep them FIXED (only the solute
// moves) so the model is cheap and the sampling is purely over the solute's
// position -- the one coordinate whose coupling we are switching. The sites are
// laid out deterministically by make_synthetic-style code (see reference_cpu).
//   Layout: x[i], y[i], z[i] are SoA arrays of length n_solvent (coalesced reads
//   on the GPU: consecutive threads would read consecutive i, though here each
//   walker reads the whole bath).
// ---------------------------------------------------------------------------
struct SolventBath {
    const double* x;   // [n_solvent] solvent x-coordinates  [sigma]
    const double* y;   // [n_solvent] solvent y-coordinates  [sigma]
    const double* z;   // [n_solvent] solvent z-coordinates  [sigma]
    int           n;   // number of solvent sites
};

// ===========================================================================
// 1. Counter-based RNG (deterministic, parallel-safe)
// ---------------------------------------------------------------------------
//   Monte Carlo needs random numbers, but for a REPRODUCIBLE demo (CLAUDE.md
//   section 12, PATTERNS.md section 3) every walker must draw the SAME stream on
//   the CPU and on the GPU. A counter-based hash RNG gives us that: the n-th
//   draw of walker w is a pure function hash(seed, w, n) -- no shared mutable
//   state, so thread w on the GPU and iteration w on the CPU produce identical
//   sequences. We use a SplitMix64-style finalizer; it passes basic statistical
//   tests and is one multiply-xor-shift chain (cheap on a GPU).
// ===========================================================================
struct Rng {
    uint64_t state;   // advanced once per draw; seeded from (seed, walker, salt)
};

// Build a walker's RNG by hashing its (seed, walker-id) so every walker has an
// independent, repeatable stream. The constant is SplitMix64's increment.
ALCH_HD inline Rng rng_init(uint64_t seed, uint64_t walker) {
    Rng r;
    // Mix the seed and walker id so nearby walker ids give far-apart streams.
    r.state = seed + walker * 0x9E3779B97F4A7C15ULL;
    return r;
}

// Return the next uint64 and advance the state (SplitMix64).
ALCH_HD inline uint64_t rng_next_u64(Rng& r) {
    r.state += 0x9E3779B97F4A7C15ULL;             // additive Weyl step
    uint64_t z = r.state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;  // avalanche the high bits down
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Uniform double in [0,1). We take the top 53 bits (the mantissa width of a
// double) so every representable value is reachable and the result is exact.
ALCH_HD inline double rng_uniform(Rng& r) {
    return (rng_next_u64(r) >> 11) * (1.0 / 9007199254740992.0);  // 2^-53
}

// Uniform double in [-h, +h]; used for the symmetric Metropolis trial move.
ALCH_HD inline double rng_uniform_pm(Rng& r, double h) {
    return (2.0 * rng_uniform(r) - 1.0) * h;
}

// ===========================================================================
// 2. The alchemical (soft-core) potential and its lambda-derivative
// ---------------------------------------------------------------------------
//   THE PROBLEM soft-core solves. The plain Lennard-Jones energy between solute
//   and a solvent site at separation r is
//
//       U_LJ(r) = 4 eps [ (sigma/r)^12 - (sigma/r)^6 ].
//
//   If we coupled it LINEARLY as lambda*U_LJ, then near lambda=0 a solvent atom
//   could sit right on top of the (now nearly ghostly) solute: r->0 makes U_LJ
//   diverge, and lambda*U_LJ * (its derivative) blows up -> the famous
//   "end-point catastrophe" that makes TI variance explode. SOFT-CORE fixes this
//   by softening the singularity at small lambda: it replaces r^6 by an
//   effective r^6 + alpha*sigma^6*(1-lambda), so when the atom is decoupled the
//   energy stays finite even at r=0.
//
//   The standard (Beutler) soft-core LJ energy used here:
//
//       let  s6 = sigma^6
//            denom = alpha*s6*(1-lambda) + r^6           (the softened r^6)
//            x = s6 / denom                              (a dimensionless ratio)
//       U_sc(r,lambda) = lambda * 4 eps ( x^2 - x )      (x^2 ~ repulsion, x ~ attraction)
//
//   At lambda=1 (denom=r^6, x=(sigma/r)^6) this is exactly plain LJ; at lambda=0
//   it is zero and FINITE for all r. Its lambda-derivative dU/dlambda is what TI
//   integrates -- we derive it analytically below so CPU and GPU evaluate the
//   identical closed form (no finite differencing).
// ===========================================================================

// Squared distance between the solute at (sx,sy,sz) and solvent site i.
ALCH_HD inline double dist2(double sx, double sy, double sz,
                            double ox, double oy, double oz) {
    const double dx = sx - ox, dy = sy - oy, dz = sz - oz;
    return dx * dx + dy * dy + dz * dz;
}

// Soft-core LJ energy of the solute at (sx,sy,sz) against the WHOLE bath, at a
// given lambda. Sums over all solvent sites. Returns energy in eps units.
//   Why a function (not inlined at the call site): the Metropolis walker, the
//   energy report, AND the dU/dlambda estimator all need this exact expression;
//   sharing one definition guarantees they stay consistent.
ALCH_HD inline double softcore_energy(double sx, double sy, double sz,
                                      const SolventBath& bath,
                                      const SystemParams& p, double lambda) {
    const double sigma3 = p.sigma * p.sigma * p.sigma;   // sigma^3
    const double sigma6 = sigma3 * sigma3;               // sigma^6 (sets the LJ length scale)
    const double soft = p.alpha_sc * sigma6 * (1.0 - lambda);  // soft-core offset
    double e = 0.0;
    for (int i = 0; i < bath.n; ++i) {
        const double r2 = dist2(sx, sy, sz, bath.x[i], bath.y[i], bath.z[i]);
        const double r6 = r2 * r2 * r2;          // r^6
        const double denom = soft + r6;          // softened r^6 (never zero for lambda<1)
        const double x = sigma6 / denom;         // dimensionless ratio
        // Lennard-Jones in the soft-core variable, scaled linearly by lambda.
        e += lambda * 4.0 * p.epsilon * (x * x - x);
        // Optional screened-Coulomb term, also alchemically scaled by lambda.
        // (A crude reaction-field-free Coulomb; teaching only.) Skipped if q=0.
        if (p.q_solute != 0.0) {
            const double r = std::sqrt(r2 + 1e-12);
            e += lambda * p.q_solute / r;        // ~1/r, decoupled with lambda
        }
    }
    return e;
}

// The lambda-DERIVATIVE of the soft-core energy at fixed configuration -- the TI
// integrand <dU/dlambda>. We differentiate U_sc analytically:
//
//   U_sc = lambda * 4 eps (x^2 - x),   x = sigma6 / (alpha sigma6 (1-lambda) + r^6)
//
//   dU/dlambda = 4 eps (x^2 - x)                          [explicit lambda factor]
//              + lambda * 4 eps (2x - 1) * dx/dlambda      [x depends on lambda]
//
//   dx/dlambda = -sigma6 / denom^2 * d(denom)/dlambda,
//   d(denom)/dlambda = -alpha sigma6  (since denom = alpha sigma6 (1-lambda)+r^6)
//   => dx/dlambda = sigma6 * alpha sigma6 / denom^2 = alpha * sigma6 * x^2 ... etc.
//
//   We compute it term by term below; keeping it analytic (not a finite
//   difference) is what lets the CPU and GPU agree to round-off.
ALCH_HD inline double softcore_dudl(double sx, double sy, double sz,
                                    const SolventBath& bath,
                                    const SystemParams& p, double lambda) {
    const double sigma3 = p.sigma * p.sigma * p.sigma;
    const double sigma6 = sigma3 * sigma3;
    const double soft = p.alpha_sc * sigma6 * (1.0 - lambda);
    double dudl = 0.0;
    for (int i = 0; i < bath.n; ++i) {
        const double r2 = dist2(sx, sy, sz, bath.x[i], bath.y[i], bath.z[i]);
        const double r6 = r2 * r2 * r2;
        const double denom = soft + r6;
        const double x = sigma6 / denom;                 // the ratio again
        // d(denom)/dlambda = -alpha*sigma6  => dx/dlambda = -sigma6 * (-alpha sigma6)/denom^2
        const double dx_dl = p.alpha_sc * sigma6 * sigma6 / (denom * denom);
        const double u_per_eps = 4.0 * p.epsilon * (x * x - x);          // U/lambda part
        const double dterm = lambda * 4.0 * p.epsilon * (2.0 * x - 1.0) * dx_dl;
        dudl += u_per_eps + dterm;                       // explicit + implicit lambda terms
        if (p.q_solute != 0.0) {
            const double r = std::sqrt(r2 + 1e-12);
            dudl += p.q_solute / r;                      // d(lambda*q/r)/dlambda = q/r
        }
    }
    return dudl;
}

// ===========================================================================
// 3. The per-walker Metropolis Monte Carlo sampler (the ensemble member)
// ---------------------------------------------------------------------------
//   ONE walker = one independent Markov chain sampling the solute position at a
//   FIXED lambda. Each step proposes a small random displacement of the solute,
//   accepts it with the Metropolis probability min(1, exp(-beta*dU)), and -- after
//   an equilibration burn-in -- accumulates the TI integrand dU/dlambda and (for
//   BAR) the energies at the two neighbouring lambda values.
//
//   This is the function we run in parallel: the CPU loops it over walkers; the
//   GPU gives each walker its own thread. Identical code -> identical numbers.
// ===========================================================================

// What one walker reports back. Doubles so CPU and GPU match to round-off.
struct WalkerResult {
    double sum_dudl;     // sum of dU/dlambda over the production samples (TI integrand)
    double sum_du_fwd;   // sum of [U(lambda_next) - U(lambda)]  (for BAR, forward)
    double sum_du_bwd;   // sum of [U(lambda_prev) - U(lambda)]  (for BAR, backward)
    long   n_samples;    // number of production samples taken (for the mean)
    long   n_accept;     // accepted moves (a sampling-quality diagnostic)
};

// Run ONE Metropolis walker at window `w` (coupling `lambda`), with neighbouring
// window couplings lambda_prev / lambda_next supplied for the BAR energy
// differences (use lambda itself at the ends, which makes those deltas zero).
//
//   seed        : global RNG seed (same on CPU + GPU)
//   walker_uid  : a globally-unique walker id (window*walkers + w) so every chain
//                 is independent and reproducible
//   n_equil     : burn-in steps discarded before sampling (decorrelate the chain)
//   n_prod      : production steps; every step contributes one sample
ALCH_HD inline WalkerResult run_walker(const SolventBath& bath, const SystemParams& p,
                                       double lambda, double lambda_prev, double lambda_next,
                                       uint64_t seed, uint64_t walker_uid,
                                       int n_equil, int n_prod) {
    Rng rng = rng_init(seed, walker_uid);
    const double beta = 1.0 / p.temperature;     // inverse temperature 1/(k_B T)

    // Start the solute at a reproducible position inside the box (drawn from the
    // walker's own stream so different walkers start differently but each one is
    // deterministic). Sampling the start avoids every chain beginning identically.
    double sx = rng_uniform_pm(rng, p.box);
    double sy = rng_uniform_pm(rng, p.box);
    double sz = rng_uniform_pm(rng, p.box);
    double e_cur = softcore_energy(sx, sy, sz, bath, p, lambda);   // current energy

    WalkerResult out{0.0, 0.0, 0.0, 0, 0};
    const int total = n_equil + n_prod;
    for (int step = 0; step < total; ++step) {
        // --- propose a symmetric random displacement of the solute ----------
        const double nx = sx + rng_uniform_pm(rng, p.max_step);
        const double ny = sy + rng_uniform_pm(rng, p.max_step);
        const double nz = sz + rng_uniform_pm(rng, p.max_step);
        bool inside = (nx > -p.box && nx < p.box &&
                       ny > -p.box && ny < p.box &&
                       nz > -p.box && nz < p.box);   // hard-wall box (reflecting via reject)
        double e_new = inside ? softcore_energy(nx, ny, nz, bath, p, lambda) : e_cur;

        // --- Metropolis accept/reject --------------------------------------
        // Accept with probability min(1, exp(-beta*(e_new-e_cur))). We ALWAYS
        // draw the uniform (even when the move is uphill-free) so the RNG stream
        // stays in lock-step between CPU and GPU regardless of the branch taken.
        const double du = e_new - e_cur;
        const double xi = rng_uniform(rng);
        const bool accept = inside && (du <= 0.0 || xi < std::exp(-beta * du));
        if (accept) {
            sx = nx; sy = ny; sz = nz;
            e_cur = e_new;
            if (step >= n_equil) ++out.n_accept;
        }

        // --- accumulate TI / BAR observables during PRODUCTION only ---------
        if (step >= n_equil) {
            out.sum_dudl += softcore_dudl(sx, sy, sz, bath, p, lambda);
            // BAR needs U at the neighbouring lambdas for the SAME configuration.
            const double e_here = e_cur;
            out.sum_du_fwd += softcore_energy(sx, sy, sz, bath, p, lambda_next) - e_here;
            out.sum_du_bwd += softcore_energy(sx, sy, sz, bath, p, lambda_prev) - e_here;
            ++out.n_samples;
        }
    }
    return out;
}
