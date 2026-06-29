// ===========================================================================
// src/alchemy.h  --  Shared (host + device) alchemical model, RNG, and MC sampler
// ---------------------------------------------------------------------------
// Project 1.5 : Free Energy Perturbation / Thermodynamic Integration (FEP/TI)
//   (a deliberately REDUCED-SCOPE teaching model -- see ../THEORY.md "real world")
//
// WHAT THIS PROJECT COMPUTES
//   The free-energy difference DeltaG between two states A and B of a system,
//   using Thermodynamic Integration (TI) over an alchemical lambda-pathway --
//   exactly the idea behind production FEP/TI in drug discovery, but stripped to
//   a model small enough to verify against a CLOSED-FORM answer.
//
//   The system is ONE particle on a line (1-D), and the two "states" are two
//   harmonic springs (think: a ligand atom whose stiffness/centre is morphed as
//   we mutate ligand A into ligand B):
//       U_A(x) = 1/2 kA (x - x0A)^2          (state A, lambda = 0)
//       U_B(x) = 1/2 kB (x - x0B)^2          (state B, lambda = 1)
//   We connect them with LINEAR alchemical coupling (the simplest lambda-schedule):
//       U(x, lambda) = (1 - lambda) U_A(x) + lambda U_B(x).
//
//   THERMODYNAMIC INTEGRATION.  The free energy along the path obeys
//       dG/dlambda = < dU/dlambda >_lambda           (an equilibrium average)
//   so
//       DeltaG = G_B - G_A = integral_0^1 < dU/dlambda >_lambda  dlambda.
//   Here dU/dlambda = U_B(x) - U_A(x), and < . >_lambda is the Boltzmann average
//   in the coupled ensemble at coupling lambda. We estimate each < dU/dlambda >
//   by Markov-chain Monte Carlo (MC) sampling -- one INDEPENDENT MC chain per
//   lambda-window -- then integrate over lambda with the trapezoid rule.
//
//   WHY THIS IS VERIFIABLE.  A 1-D harmonic oscillator has an analytic free
//   energy (see ../THEORY.md):  G(k) = 1/2 kT ln( k / (2 pi kT) )  (up to an
//   additive constant that cancels in DeltaG), giving the EXACT result
//       DeltaG_analytic = G_B - G_A = 1/2 kT ln( kB / kA ).
//   The TI estimate from sampling must converge to this -- that is our second,
//   stronger correctness check (the science, not just CPU==GPU agreement).
//
// THE GPU PATTERN (PATTERNS.md row "same ODE/sampler for many parameter sets")
//   Each lambda-window is an INDEPENDENT sampling job. We give each window its
//   own GPU thread; the thread runs the whole MC chain in registers and writes
//   one number, < dU/dlambda >. This is the same "ensemble of independent jobs"
//   mapping as the SEIR ensemble (9.02) and PBPK (13.02).
//
// CPU/GPU PARITY (PATTERNS.md section 2: the __host__ __device__ core)
//   The per-element physics -- potentials, the RNG, and ONE MC chain -- live in
//   this header as ALCH_HD (= __host__ __device__ under nvcc, nothing on the host
//   compiler). The CPU reference loops over windows calling run_chain(); the GPU
//   kernel calls the SAME run_chain() from one thread per window. Same code +
//   same deterministic RNG  =>  the two agree to round-off, and verification is
//   essentially exact.
//
// DETERMINISM (PATTERNS.md section 3)
//   Monte Carlo needs random numbers, but the demo's stdout must be identical
//   every run. We therefore use a COUNTER-BASED RNG: the n-th random number of
//   window w is a pure hash of the pair (w, n) -- no mutable global seed, no
//   thread-ordering dependence. CPU and GPU hash the same (w, n) pairs, so they
//   draw the SAME stream and produce byte-identical results. (This is exactly
//   why production GPU-MC codes favour counter-based RNGs like Philox.)
//
//   Keep this header free of CUDA-only types (no __global__, no <cuda_runtime.h>)
//   so the host compiler can include it. READ THIS AFTER: util/, then
//   reference_cpu.h; the GPU twin is kernels.cu.
// ===========================================================================
#pragma once

#include <cmath>     // std::exp, std::log, std::sqrt
#include <cstdint>   // std::uint64_t

// ALCH_HD expands to __host__ __device__ when compiled by nvcc, and to nothing
// when compiled by the plain host C++ compiler (which does not know those
// keywords). One source of truth, two compilers, identical math.
#ifdef __CUDACC__
#define ALCH_HD __host__ __device__
#else
#define ALCH_HD
#endif

// ---------------------------------------------------------------------------
// AlchemyConfig: everything that defines an FEP/TI run.
//   The two harmonic end states (kA,x0A) and (kB,x0B), the temperature, the
//   number of lambda-windows (the discretisation of the [0,1] path), and the MC
//   sampling settings. Plain old data so it can be passed BY VALUE into a kernel.
// ---------------------------------------------------------------------------
struct AlchemyConfig {
    double kA   = 0.0;   // state-A spring constant  (energy / length^2)
    double x0A  = 0.0;   // state-A equilibrium position (length)
    double kB   = 0.0;   // state-B spring constant  (energy / length^2)
    double x0B  = 0.0;   // state-B equilibrium position (length)
    double kT   = 0.0;   // temperature in energy units (kB_Boltzmann * T); kB=1 here
    int    windows    = 0;   // number of lambda-windows incl. endpoints (>= 2)
    int    equil      = 0;   // MC steps discarded as burn-in before averaging
    int    samples    = 0;   // MC steps used in the < dU/dlambda > average
    double step       = 0.0; // MC trial-move half-width (length); tunes accept rate
    double x_init     = 0.0; // starting position of every chain (length)
};

// Number of independent sampling jobs = number of lambda-windows.
ALCH_HD inline int n_windows(const AlchemyConfig& c) { return c.windows; }

// The lambda value of window w on a uniform grid over [0, 1].
//   w = 0      -> lambda = 0 (pure state A)
//   w = W - 1  -> lambda = 1 (pure state B)
ALCH_HD inline double window_lambda(const AlchemyConfig& c, int w) {
    return (c.windows > 1) ? static_cast<double>(w) / (c.windows - 1) : 0.0;
}

// ---- The alchemical potential and its lambda-derivative -------------------

// Harmonic potential of one end state: U = 1/2 k (x - x0)^2.
ALCH_HD inline double harmonic(double x, double k, double x0) {
    const double d = x - x0;
    return 0.5 * k * d * d;
}

// Coupled potential U(x, lambda) = (1-lambda) U_A + lambda U_B. This is what the
// MC sampler feels at coupling `lam`.
ALCH_HD inline double U_coupled(double x, double lam, const AlchemyConfig& c) {
    return (1.0 - lam) * harmonic(x, c.kA, c.x0A) + lam * harmonic(x, c.kB, c.x0B);
}

// The TI integrand's per-configuration value:  dU/dlambda = U_B(x) - U_A(x).
//   Averaging THIS over the coupled ensemble at `lam` gives < dU/dlambda >_lambda,
//   the height of the TI curve at that lambda. (Linear coupling makes the
//   derivative independent of lambda for a fixed x -- a clean teaching case.)
ALCH_HD inline double dU_dlambda(double x, const AlchemyConfig& c) {
    return harmonic(x, c.kB, c.x0B) - harmonic(x, c.kA, c.x0A);
}

// ---------------------------------------------------------------------------
// Counter-based RNG (deterministic, stateless).
//   We need a stream of uniform doubles in [0,1) that depends ONLY on a key, so
//   that the CPU and GPU -- and indeed any run -- produce the SAME numbers. We
//   hash a 64-bit counter with the well-known SplitMix64 finalizer (a strong
//   bit-mixer used in many PRNGs) and scale to [0,1). No mutable state, no
//   per-thread seed array: random number `n` of window `w` is purely a function
//   of the key we build from (w, n). This is the GPU-friendly, reproducible way
//   to do Monte Carlo (cf. NVIDIA cuRAND's Philox counter-based generator).
// ---------------------------------------------------------------------------
ALCH_HD inline std::uint64_t splitmix64(std::uint64_t z) {
    // Three xor-shift / odd-multiply rounds: avalanche every input bit.
    z += 0x9E3779B97F4A7C15ull;             // golden-ratio increment (decorrelates keys)
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Map a 64-bit hash to a uniform double in [0,1). Taking the top 53 bits gives a
// double with full mantissa precision and no low-bit bias, then scaling by
// 2^-53 lands it in [0,1).  (2^53 = 9007199254740992; a double represents it
// exactly, so the division is exact.)
ALCH_HD inline double u01_from_hash(std::uint64_t h) {
    return (h >> 11) * (1.0 / 9007199254740992.0 /* = 2^-53 */);
}

// The `n`-th uniform draw of window `w`, optionally on "channel" `ch` (0 = the
// accept/move proposal, 1 = the Metropolis accept test) so the two draws a step
// needs never collide. Combine the three integers into one 64-bit key, then mix.
ALCH_HD inline double rng_uniform(int w, std::uint64_t n, std::uint64_t ch) {
    // Pack: window in the high bits, step counter in the middle, channel low.
    // (window < 2^20 and channel < 2^4 in practice, so no overlap with `n`.)
    std::uint64_t key = (static_cast<std::uint64_t>(w) << 40)
                      ^ (n << 4)
                      ^ ch;
    return u01_from_hash(splitmix64(key));
}

// ---------------------------------------------------------------------------
// run_chain: run ONE Metropolis MC chain for window `w` and return the estimate
//            of < dU/dlambda >_lambda for that window.
//   Algorithm (textbook Metropolis sampling of the Boltzmann distribution
//   exp(-U/kT) at coupling lambda):
//     repeat for (equil + samples) steps:
//       1. propose  x' = x + (2u - 1) * step      [u uniform, channel 0]
//       2. dE = U(x',lambda) - U(x,lambda)
//       3. accept with probability min(1, exp(-dE/kT))   [channel 1]
//       4. after burn-in, accumulate dU/dlambda(x)
//   Returns the mean of dU/dlambda over the post-burn-in samples.
//
//   This is SHARED by the CPU reference (loops over windows) and the GPU kernel
//   (one thread per window). Same RNG keys => identical chains => exact match.
//   Returns the count of accepted moves via `accepted` (for an acceptance-rate
//   diagnostic on stderr; not part of the deterministic result).
// ---------------------------------------------------------------------------
ALCH_HD inline double run_chain(const AlchemyConfig& c, int w, long long* accepted) {
    const double lam = window_lambda(c, w);
    double x = c.x_init;                 // start every chain at the same x (determinism)
    double Ux = U_coupled(x, lam, c);    // cached current potential energy
    double sum = 0.0;                    // running sum of dU/dlambda over samples
    long long acc = 0;                   // accepted-move counter (diagnostic only)

    const long long total = static_cast<long long>(c.equil) + c.samples;
    for (long long n = 0; n < total; ++n) {
        // 1. Propose a symmetric random displacement in [-step, +step].
        const double u_move = rng_uniform(w, static_cast<std::uint64_t>(n), 0);
        const double x_new  = x + (2.0 * u_move - 1.0) * c.step;

        // 2. Energy change of the proposed move at this coupling.
        const double U_new = U_coupled(x_new, lam, c);
        const double dE    = U_new - Ux;

        // 3. Metropolis criterion: always accept downhill (dE<=0); accept uphill
        //    with probability exp(-dE/kT). exp() of a non-positive argument when
        //    dE<=0 would be >=1, so we only draw the random test when dE>0.
        bool accept;
        if (dE <= 0.0) {
            accept = true;
        } else {
            const double u_acc = rng_uniform(w, static_cast<std::uint64_t>(n), 1);
            accept = (u_acc < std::exp(-dE / c.kT));
        }
        if (accept) { x = x_new; Ux = U_new; ++acc; }

        // 4. Accumulate the TI integrand AFTER burn-in (the chain has, by then,
        //    relaxed to the Boltzmann distribution of this window).
        if (n >= c.equil) {
            sum += dU_dlambda(x, c);
        }
    }
    if (accepted) *accepted = acc;
    return (c.samples > 0) ? sum / c.samples : 0.0;   // < dU/dlambda >_lambda
}

// ---------------------------------------------------------------------------
// trapezoid_ti: integrate the per-window averages over lambda in [0,1] with the
//   composite trapezoid rule to get DeltaG_TI.  windows are at lambda = i/(W-1),
//   so the uniform spacing is h = 1/(W-1) and
//       integral ~ h * ( f0/2 + f1 + ... + f_{W-2} + f_{W-1}/2 ).
//   Pure host-side post-processing (small W), but kept here next to the model.
// ---------------------------------------------------------------------------
ALCH_HD inline double trapezoid_ti(const double* dvals, int W) {
    if (W < 2) return 0.0;
    const double h = 1.0 / (W - 1);
    double s = 0.5 * (dvals[0] + dvals[W - 1]);
    for (int i = 1; i < W - 1; ++i) s += dvals[i];
    return h * s;
}

// ---------------------------------------------------------------------------
// analytic_delta_g: the closed-form free-energy difference for the two harmonic
//   end states.  G(k) = 1/2 kT ln( k / (2 pi kT) ) (+const), so the const and
//   the well centres x0 cancel and
//       DeltaG = 1/2 kT ln( kB / kA ).
//   This is the ground truth the TI estimate must approach (../THEORY.md).
// ---------------------------------------------------------------------------
ALCH_HD inline double analytic_delta_g(const AlchemyConfig& c) {
    return 0.5 * c.kT * std::log(c.kB / c.kA);
}
