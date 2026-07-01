// ===========================================================================
// src/stray_physics.h  --  Shared (host + device) RNG + stray-dose transport
// ---------------------------------------------------------------------------
// Project 5.10 : Secondary Cancer Risk & Stray-Dose Monte Carlo
//                (reduced-scope teaching version -- see ../THEORY.md)
//
// WHY THIS HEADER IS SHARED (the HD-macro idiom, PATTERNS.md section 2)
//   The whole point of Monte Carlo *verification* is that the CPU reference and
//   the GPU kernel simulate the IDENTICAL particle histories, so their dose
//   tallies must match EXACTLY. That only works if both sides use the same RNG
//   and the same transport logic -- so all of the per-history physics lives here,
//   in ONE header, included by:
//       * reference_cpu.cpp   (compiled by the host C++ compiler), and
//       * kernels.cu / main.cu (compiled by nvcc).
//   The HD macro expands to `__host__ __device__` under nvcc and to nothing under
//   the plain host compiler, so the same inline functions compile in both worlds.
//   (Production stray-dose MC uses cuRAND; we use a shared, reproducible
//    counter-based RNG on purpose, so CPU and GPU histories are bit-identical and
//    the demo is deterministic. THEORY.md section "Numerical considerations".)
//
// THE PHYSICS PROBLEM (why this project exists)
//   Radiotherapy aims a high-dose beam at a tumour, but radiation does not stop
//   at the field edge. STRAY DOSE reaches distant, healthy organs three to four
//   ORDERS OF MAGNITUDE below the target dose, through three channels:
//       (1) SCATTER  -- primary photons Compton-scatter inside the patient and
//                       redirect a little energy sideways to out-of-field organs;
//       (2) LEAKAGE  -- photons that punch through the machine head / collimator
//                       and irradiate the whole body roughly uniformly;
//       (3) NEUTRONS -- (proton/high-energy photon therapy) nuclear reactions in
//                       high-Z nozzle parts make secondary neutrons. We model this
//                       as an extra low-weight leakage channel and DISCUSS the full
//                       hadronic-transport version in THEORY.md; a real INCL/BERT
//                       cascade is research-grade (CLAUDE.md section 13).
//   Those tiny stray doses matter because they act on large volumes of healthy
//   tissue over a patient's remaining lifetime -> SECONDARY CANCER RISK, which we
//   quantify with BEIR-VII organ risk coefficients (see risk_model.h).
//
// THE CENTRAL GPU/MC LESSON: VARIANCE REDUCTION
//   Stray dose is a RARE signal. A naive "analog" MC would fire ~1e11-1e12
//   histories to get a few counts in a distant organ -- intractable. We teach the
//   two workhorse variance-reduction (VR) techniques instead, both applied
//   per-thread:
//       * SURVIVAL BIASING + RUSSIAN ROULETTE: a photon is never simply
//         "absorbed and killed". At each interaction it keeps a fractional
//         statistical WEIGHT (weight *= scatter_fraction) and continues. When the
//         weight falls below a floor, Russian roulette either kills it (freeing the
//         thread) or boosts it back up -- an UNBIASED way to stop tracking
//         negligible particles.
//       * FORCED DETECTION (a.k.a. expectation / "next-event" scoring): instead of
//         waiting for the rare event that a scattered photon happens to travel to a
//         distant organ AND deposit there, at every scatter site we DETERMINISTICALLY
//         add the expected contribution to each downstream organ
//         (weight * p_scatter_toward_organ * exp(-attenuation) * deposit_frac).
//         This converts a rare stochastic hit into a small guaranteed tally every
//         history -> orders of magnitude less variance for the same history count.
//   Both are standard in EGSnrc / TOPAS / GATE; here they are written by hand so
//   the learner can see exactly what they do.
//
// DETERMINISM NOTE (PATTERNS.md section 3)
//   Weights are floating point, but a float atomicAdd is order-dependent and would
//   make the GPU tally irreproducible and unequal to the CPU. So we score into
//   FIXED-POINT INTEGERS: each deposit's (weight * energy) is scaled by
//   DOSE_FIXED_SCALE and truncated to a 64-bit integer BEFORE accumulation. Integer
//   adds commute, so the GPU sum is deterministic and bit-identical to the CPU sum.
//   (Same trick as 5.01's integer energy quanta and 11.09's fixed-point centroids.)
//
// READ THIS AFTER: nothing (start here) ; READ NEXT: risk_model.h, reference_cpu.h.
// ===========================================================================
#pragma once

#include <cstdint>
#include <cmath>

// HD = "host+device". Under nvcc, decorate the inline physics so it compiles for
// both the CPU reference and the GPU kernel from ONE definition. Under the host
// compiler the decorators do not exist, so the macro expands to nothing.
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ---------------------------------------------------------------------------
// Fixed-point dose scale. A "unit" of primary beam energy is 1.0; stray dose is
// ~1e-3..1e-6 of that. Multiplying weighted energy by 1e9 and truncating to a
// 64-bit integer keeps ~9 significant digits of the fraction while making the
// accumulation exact and order-independent. (2^63 / 1e9 ~ 9.2e9 units of headroom
// -- far more than any per-organ tally in this teaching problem.)
//
// `constexpr` (not `static const`) so nvcc treats it as a true compile-time
// constant usable from BOTH host and __device__ code -- a plain namespace-scope
// `static const double` is not addressable in device code.
// ---------------------------------------------------------------------------
constexpr double DOSE_FIXED_SCALE = 1.0e9;

// Convert a floating-point weighted-energy deposit into fixed-point integer units
// for atomic accumulation. Truncation (not rounding) is used identically on host
// and device so the two sides agree to the last bit.
HD inline unsigned long long dose_to_fixed(double weighted_energy) {
    double v = weighted_energy * DOSE_FIXED_SCALE;
    if (v < 0.0) v = 0.0;                 // weights are non-negative; guard anyway
    return static_cast<unsigned long long>(v);   // truncate toward zero
}

// ---------------------------------------------------------------------------
// RNG: splitmix64, a counter-based stream that is IDENTICAL on host and device.
// We seed one independent stream per history so particles are uncorrelated yet
// each history is exactly reproducible from (base_seed, history_index).
// ---------------------------------------------------------------------------
struct Rng { uint64_t state; };

// One splitmix64 step: advance `x` and return a well-mixed 64-bit value. This is
// a tiny, fast, high-quality mixer -- ideal for reproducible MC where we do not
// need cryptographic strength, just good equidistribution.
HD inline uint64_t splitmix64(uint64_t& x) {
    x += 0x9E3779B97F4A7C15ULL;                   // golden-ratio increment
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;  // avalanche stage 1
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;  // avalanche stage 2
    return z ^ (z >> 31);
}

// Seed an independent stream for `history` from the base seed. The multiply +
// xor spreads nearby history indices into far-apart states so consecutive
// histories are uncorrelated.
HD inline Rng rng_seed(uint64_t base, uint64_t history) {
    Rng r;
    r.state = base ^ (history * 0x9E3779B97F4A7C15ULL + 0xD1B54A32D192ED03ULL);
    splitmix64(r.state);   // warm up once so state != seed
    return r;
}

// Uniform double in [0,1) from the top 53 random bits (identical math on both
// sides). 2^-53 is the spacing of representable doubles in [0.5,1), so this is
// the standard "53-bit uniform".
HD inline double rng_uniform(Rng& r) {
    uint64_t z = splitmix64(r.state);
    return (z >> 11) * (1.0 / 9007199254740992.0);   // z/2^53
}

// ---------------------------------------------------------------------------
// The phantom + beam parameters (read from the data file). The "phantom" is a
// reduced 1-D stack of ORGAN SLABS along the patient's long axis (head->foot).
// Slab 0 is the treated target; slabs beyond field_end are out-of-field organs
// that receive only stray dose. This 1-D reduction keeps the transport legible
// while preserving the essential out-of-field falloff physics; THEORY.md explains
// the step up to full 3-D ICRP-110 voxel phantoms.
// ---------------------------------------------------------------------------
struct SimParams {
    int    n_organs;        // number of organ slabs in the phantom (target + OOF)
    int    field_end;       // organ index where the treatment field ends (0..field_end-1 are in-field)
    double mu;              // linear attenuation coefficient of tissue (1/cm)
    double organ_cm;        // thickness of each organ slab along the axis (cm)
    double scatter_frac;    // fraction of an interacting photon's weight that scatters (vs. absorbed)
    double sidescatter;     // fraction of scattered weight redirected out-of-field per organ (forced-detection kernel)
    double leakage_frac;    // machine-head leakage: uniform low weight added to EVERY organ per primary
    double neutron_frac;    // secondary-neutron surrogate: extra uniform weight, distance-weighted (see note above)
    double roulette_floor;  // weight below which Russian roulette is played
    double roulette_survive;// survival probability in roulette (weight boosted by 1/this on survival)
    unsigned long long n_histories; // number of primary photon histories to simulate
    uint64_t seed;          // base RNG seed
};

// Maximum number of (organ, fixed-dose) deposits one history can record. A
// history deposits into: its interaction organs (primary), the forced-detection
// contributions to downstream organs, plus per-organ leakage/neutron surrogate.
// n_organs is small in the teaching problem, so a generous fixed cap avoids any
// dynamic allocation inside the kernel (no malloc on the device hot path).
static const int STRAY_MAX_DEPOSITS = 256;

// A single history's output: parallel arrays of (organ index, fixed-point dose).
// The caller applies them to the tally -- atomicAdd on the GPU, plain += on the
// CPU. Splitting "compute deposits" from "accumulate deposits" is what lets the
// identical physics feed two different accumulation strategies (see kernels.cu).
struct DepositList {
    int organ[STRAY_MAX_DEPOSITS];
    unsigned long long dose[STRAY_MAX_DEPOSITS];   // fixed-point (see dose_to_fixed)
    int count;
};

// Helper: append one (organ, weighted-energy) deposit, converting to fixed point.
// Silently drops deposits past the cap (cannot happen with the teaching sizes;
// the guard keeps the kernel memory-safe if a learner cranks the parameters up).
HD inline void push_deposit(DepositList& dl, int organ, double weighted_energy) {
    if (dl.count < STRAY_MAX_DEPOSITS) {
        dl.organ[dl.count] = organ;
        dl.dose[dl.count]  = dose_to_fixed(weighted_energy);
        ++dl.count;
    }
}

// ---------------------------------------------------------------------------
// simulate_history: transport ONE primary photon through the phantom and record
// every dose deposit (in-field target dose + all out-of-field stray dose).
//
//   P    : phantom + beam + variance-reduction parameters
//   rng  : this history's private RNG stream (advanced by reference; identical
//          sequence on host and device)
//   dl   : output deposit list (cleared here, filled here)
//
// KEY GEOMETRY POINT: the treatment beam is COLLIMATED to the field. The primary
// photon therefore only ever travels through the IN-FIELD organs (indices
// 0..field_end-1) -- it does NOT traverse the whole body. Out-of-field organs
// (index >= field_end) receive NO primary dose; their entire dose is STRAY:
// forced-detected scatter from in-field interaction sites + leakage + neutrons.
// This is why their dose is 3-5 orders of magnitude below the target -- exactly
// the effect the project is about. (An earlier draft let the primary pass through
// every organ, which wrongly made distant organs almost as hot as the target.)
//
// ALGORITHM (per history), all in statistical WEIGHT (survival biasing):
//   * LEAKAGE + NEUTRON channels (all organs): machine leakage sprays a tiny
//     uniform weight onto every organ; the neutron surrogate adds a small weight
//     that falls off with distance from the field edge. These model the MACHINE,
//     not the stochastic patient path, so they are deposited deterministically ->
//     a low-variance contribution present even in the farthest organ.
//   * PRIMARY WALK (in-field organs only, i = 0..field_end-1): the photon starts
//     with weight w = 1.0 and energy E = 1.0. In each in-field organ it interacts
//     with probability p_int = 1 - exp(-mu*organ_cm) (Beer-Lambert over one slab):
//         - a fraction 'scatter_frac' SCATTERS (survives), the rest is ABSORBED and
//           deposits locally as the large in-field TARGET dose;
//         - FORCED DETECTION: at the scatter site we add, to EACH out-of-field
//           organ j (>= field_end), the *expected* stray contribution
//               w_scatter * sidescatter * exp(-mu * distance_to_j) * E
//           deterministically -- no waiting for a rare lateral-scatter event. This
//           is the variance-reduction heart of the project.
//         - the surviving scattered weight continues to the next in-field organ.
//   * RUSSIAN ROULETTE: once the surviving weight drops below roulette_floor, play
//     roulette: with prob 'roulette_survive' keep the photon but divide its weight
//     by roulette_survive (unbiased boost); otherwise terminate the history.
//
// Returns nothing; deposits are in `dl`. Complexity: O(field_end * n_organs)
// (each in-field interaction forced-detects to every out-of-field organ), tiny
// here and fully independent per history -> embarrassingly parallel on the GPU.
// ---------------------------------------------------------------------------
HD inline void simulate_history(const SimParams& P, Rng& rng, DepositList& dl) {
    dl.count = 0;

    const double E = 1.0;   // one unit of primary energy per photon (arbitrary teaching unit)

    // -- Machine-driven channels: leakage (uniform) + neutron surrogate (distance
    //    weighted). These model the treatment MACHINE, not the stochastic patient
    //    path, so they are the same for every history -> deposited deterministically.
    //    They apply to OUT-OF-FIELD organs (the stray-dose sites); in-field organs
    //    are dominated by primary dose so we do not add the negligible machine bath
    //    there (it would be lost in the target dose anyway).
    for (int j = P.field_end; j < P.n_organs; ++j) {
        // Leakage: photons escaping the collimator/head irradiate the whole body
        // ~uniformly. Tiny per-history weight; sums to a measurable whole-body bath.
        double leak = P.leakage_frac * E;

        // Neutron surrogate: secondary neutrons are produced near the nozzle and
        // their fluence falls off away from the field. We approximate that as an
        // exponential in the number of organs past the field edge. (The real thing
        // needs hadronic transport; see THEORY.md "Where this sits in the real world".)
        double past = static_cast<double>(j - P.field_end + 1);   // organs past the field edge
        double neut = P.neutron_frac * E * exp(-0.15 * past);      // gentle falloff

        if (leak + neut > 0.0) push_deposit(dl, j, leak + neut);
    }

    // -- Primary stochastic walk with survival biasing + forced detection.
    //    The beam is collimated to the field, so the primary only visits in-field
    //    organs [0, field_end).
    double w = 1.0;                              // statistical weight of the primary
    const double p_int = 1.0 - exp(-P.mu * P.organ_cm);   // per-slab interaction prob (Beer-Lambert)

    for (int i = 0; i < P.field_end; ++i) {
        // Sample whether the (surviving) primary interacts in in-field organ i.
        double xi = rng_uniform(rng);
        if (xi < p_int) {
            // --- It interacts. Split weight into scattered vs absorbed. ---
            double w_scat = w * P.scatter_frac;      // survives, keeps travelling
            double w_abs  = w - w_scat;              // absorbed here -> local dose

            // Primary (absorbed) dose deposited locally -> the large in-field target dose.
            if (w_abs > 0.0) push_deposit(dl, i, w_abs * E);

            // --- FORCED DETECTION: deterministic stray contribution from this
            //     in-field scatter site to every OUT-OF-FIELD organ. Instead of
            //     hoping a scattered photon randomly lands in a distant organ, we add
            //     its EXPECTED deposit to each one, every time -- the big variance win.
            for (int j = P.field_end; j < P.n_organs; ++j) {
                double dist = (j - i) * P.organ_cm;          // path length to organ j (cm)
                double atten = exp(-P.mu * dist);            // attenuation along the way
                double contrib = w_scat * P.sidescatter * atten * E;
                if (contrib > 0.0) push_deposit(dl, j, contrib);
            }

            // The surviving scattered weight continues forward, reduced.
            w = w_scat;

            // --- RUSSIAN ROULETTE: stop tracking negligible weight without bias. ---
            if (w < P.roulette_floor) {
                double xr = rng_uniform(rng);
                if (xr < P.roulette_survive) {
                    w /= P.roulette_survive;   // survive -> boost weight (unbiased)
                } else {
                    break;                      // killed -> history ends, free the thread
                }
            }
        }
        // If no interaction in organ i, the full weight simply advances to i+1.
    }
}
