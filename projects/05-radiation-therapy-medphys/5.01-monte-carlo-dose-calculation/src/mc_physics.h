// ===========================================================================
// src/mc_physics.h  --  Shared (host + device) RNG and photon transport
// ---------------------------------------------------------------------------
// Project 5.01 : Monte Carlo Dose Calculation (simplified slab)
//
// WHY THIS HEADER IS SHARED
//   The whole point of Monte Carlo verification here is that the CPU reference
//   and the GPU kernel simulate the *identical* particle histories, so their
//   dose tallies must match EXACTLY. That only works if both use the same RNG
//   and the same transport logic -- so both live here, in ONE header included
//   by reference_cpu.cpp (host compiler) AND kernels.cu / main.cu (nvcc).
//
//   The RNG_HD macro expands to `__host__ __device__` under nvcc and to nothing
//   under the plain host compiler, so the same inline functions compile in both
//   worlds. (Production GPU MC uses cuRAND; we use a shared, reproducible
//   counter-based RNG specifically so CPU and GPU histories are bit-identical
//   and the demo is deterministic -- see THEORY.md.)
//
// THE SIMPLIFIED PHYSICS (a deliberately reduced teaching model; THEORY.md has
// the full picture)
//   1-D slab of thickness L, uniform attenuation coefficient mu. A photon starts
//   at depth 0 moving forward with integer "energy quanta" E0. Repeatedly:
//     * sample a step  s = -ln(xi)/mu   (exponential free path),
//     * advance depth z += s; if z >= L the photon escapes,
//     * else it interacts: with probability p_abs it is ABSORBED (deposits all
//       remaining energy in this depth bin and stops); otherwise it SCATTERS
//       (deposits one quantum-packet locally and continues forward).
//   Energy is integer quanta so dose tallies are exact under atomic add.
// ===========================================================================
#pragma once

#include <cstdint>
#include <cmath>

#ifdef __CUDACC__
#define RNG_HD __host__ __device__
#else
#define RNG_HD
#endif

// --- RNG: a splitmix64 counter-based stream (identical on host and device) ---
struct Rng { uint64_t state; };

// One splitmix64 step: advance `x` and return a well-mixed 64-bit value.
RNG_HD inline uint64_t splitmix64(uint64_t& x) {
    x += 0x9E3779B97F4A7C15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Seed an independent stream for history `history` (so each particle is
// uncorrelated yet reproducible from (base, history)).
RNG_HD inline Rng rng_seed(uint64_t base, uint64_t history) {
    Rng r;
    r.state = base ^ (history * 0x9E3779B97F4A7C15ULL + 0xD1B54A32D192ED03ULL);
    splitmix64(r.state);     // warm up
    return r;
}

// Uniform double in [0,1) from 53 random bits (identical math host/device).
RNG_HD inline double rng_uniform(Rng& r) {
    uint64_t z = splitmix64(r.state);
    return (z >> 11) * (1.0 / 9007199254740992.0);   // 2^-53
}

// --- Simulation parameters (read from the data file) ---
struct SimParams {
    double L;                 // slab thickness (cm)
    int    n_bins;            // number of depth bins
    double mu;                // attenuation coefficient (1/cm)
    double p_abs;             // probability an interaction is absorption
    unsigned long long E0;          // starting energy quanta per photon
    unsigned long long scatter_dep; // quanta deposited per scatter event
};

// Maximum deposits a single history can record (energy/scatter_dep + 1, padded).
static const int MC_MAX_DEPOSITS = 64;

// Simulate ONE photon history. Records each (bin, amount) deposit into the
// caller's arrays and returns the deposit count. The caller applies them to the
// dose tally (atomicAdd on the GPU, plain += on the CPU) -- this split is what
// lets identical physics feed two different accumulation strategies.
RNG_HD inline int simulate_photon(const SimParams& P, Rng& rng,
                                 int* bins, unsigned long long* amts) {
    const double dz = P.L / P.n_bins;          // depth-bin thickness
    double z = 0.0;                            // current depth
    unsigned long long energy = P.E0;          // remaining quanta
    int nd = 0;                                // number of deposits recorded

    for (int guard = 0; guard < 100000; ++guard) {  // guard against runaway loops
        const double xi = 1.0 - rng_uniform(rng);    // in (0,1], avoids log(0)
        z += -log(xi) / P.mu;                        // exponential free path
        if (z >= P.L) break;                         // photon escapes the slab

        int bin = static_cast<int>(z / dz);
        if (bin >= P.n_bins) bin = P.n_bins - 1;     // clamp the boundary case

        const double xi2 = rng_uniform(rng);
        if (xi2 < P.p_abs) {                          // ABSORPTION: dump it all
            if (nd < MC_MAX_DEPOSITS) { bins[nd] = bin; amts[nd] = energy; ++nd; }
            energy = 0;
            break;
        } else {                                      // SCATTER: local packet + go on
            unsigned long long dep = (energy < P.scatter_dep) ? energy : P.scatter_dep;
            if (nd < MC_MAX_DEPOSITS) { bins[nd] = bin; amts[nd] = dep; ++nd; }
            energy -= dep;
            if (energy == 0) break;
        }
    }
    return nd;
}
