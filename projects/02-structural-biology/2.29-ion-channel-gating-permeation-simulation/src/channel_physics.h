// ===========================================================================
// src/channel_physics.h  --  Shared (host + device) Brownian-dynamics core
// ---------------------------------------------------------------------------
// Project 2.29 : Ion Channel Gating & Permeation Simulation
//
// WHY THIS HEADER IS SHARED  (PATTERNS.md §2, the HD-macro idiom)
//   We verify the GPU result by running the *identical* ion trajectories on the
//   CPU and asserting the two tallies match EXACTLY. That only works if both
//   sides use the same random-number generator and the same per-step physics.
//   So both live here, in ONE header included by:
//       reference_cpu.cpp  (compiled by the host C++ compiler, cl.exe), and
//       kernels.cu / main.cu (compiled by nvcc).
//   The CP_HD macro expands to `__host__ __device__` under nvcc and to nothing
//   under the plain host compiler, so the same inline functions compile in both
//   worlds and produce bit-identical numbers.
//
//   (Production ion-channel codes use cuRAND for device RNG. We deliberately use
//    a shared, reproducible counter-based RNG so CPU and GPU trajectories are
//    bit-identical and the demo's stdout is deterministic -- see THEORY.md
//    "Numerical considerations".)
//
// THE PHYSICS  (a deliberately reduced 1-D teaching model; THEORY.md has the
// full multi-dimensional picture and the real biology)
//   We model permeation of a single ionic species along the pore axis z, reduced
//   to ONE coordinate z in [0, L] (nanometres). z=0 is the intracellular mouth,
//   z=L the extracellular mouth. Each ion is an independent Brownian (overdamped
//   Langevin) walker in a free-energy landscape:
//
//       U(z) = U_barrier * exp(-((z - L/2)^2) / (2 * sigma^2))   (PMF barrier)
//            - q * V * (z / L)                                    (applied field)
//
//   The first term is the potential of mean force (PMF): the desolvation /
//   selectivity-filter barrier an ion must climb to cross the pore. The second
//   term is the transmembrane voltage V dropped linearly across the field (this
//   is the "voltage-clamp" / applied-field modification to the integrator named
//   in the catalog). q is the ion charge in units of e. We use the SIGN
//   convention that a POSITIVE V drives a positive ion in the +z (forward,
//   intracellular->extracellular) direction, so positive V => forward current --
//   intuitive for reading the demo. (Flip the sign of q or V for the opposite.)
//
//   Overdamped Langevin (Brownian dynamics) update, the Ermak-McCammon scheme:
//
//       z_{n+1} = z_n  -  (D / kT) * dU/dz * dt  +  sqrt(2 * D * dt) * xi
//                         \__________________/      \________________/
//                          deterministic DRIFT       random DIFFUSION
//
//   where xi ~ N(0,1) is a unit Gaussian, D is the diffusion coefficient, kT the
//   thermal energy. When an ion reaches z>=L it has PERMEATED forward (a current-
//   carrying crossing): we count it and re-inject it at the z=0 bath. If it falls
//   back through z<=0 it is a reverse crossing: counted and re-injected at z=L.
//   The NET forward count over all ions and steps is proportional to the single-
//   channel current, hence the conductance -- exactly what a patch-clamp measures.
//
//   DETERMINISM TRICK (PATTERNS.md §3): every observable we *report* is an
//   INTEGER -- an occupancy histogram (how many ion-steps land in each z-bin) and
//   integer crossing counters. Integer atomicAdds commute, so the GPU tally is
//   order-independent and equals the CPU tally bit-for-bit. (A floating-point
//   current sum would depend on atomic ordering and would NOT reproduce.)
// ===========================================================================
#pragma once

#include <cstdint>
#include <cmath>

#ifdef __CUDACC__
#define CP_HD __host__ __device__
#else
#define CP_HD            // host compiler: the CUDA decorators do not exist
#endif

// ---------------------------------------------------------------------------
// RNG: a splitmix64 counter-based stream, identical on host and device.
//   We need each ion to have its own *independent yet reproducible* random
//   stream, derived from (base_seed, ion_index). splitmix64 is a tiny, high-
//   quality bijection -- perfect for seeding per-thread streams without any
//   shared state, so the CPU loop and the GPU threads generate the very same
//   sequence for the very same ion index.
// ---------------------------------------------------------------------------
struct Rng { uint64_t state; };   // the whole RNG state is a single 64-bit word

// One splitmix64 step: advance `x` in place and return a well-mixed 64-bit value.
CP_HD inline uint64_t splitmix64(uint64_t& x) {
    x += 0x9E3779B97F4A7C15ULL;                 // add the golden-ratio constant
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; // avalanche the high bits down
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; // a second mixing multiply
    return z ^ (z >> 31);                         // final xor-shift
}

// Seed an independent stream for ion `ion` from a base seed. The multiply-by-
// odd-constant + xor scatters nearby ion indices to far-apart states, so ions
// 0,1,2,... are uncorrelated yet fully reproducible from (base, ion).
CP_HD inline Rng rng_seed(uint64_t base, uint64_t ion) {
    Rng r;
    r.state = base ^ (ion * 0x9E3779B97F4A7C15ULL + 0xD1B54A32D192ED03ULL);
    splitmix64(r.state);    // warm up once so the first draw is well-mixed
    return r;
}

// Uniform double in [0,1) built from 53 random bits (identical math host/device).
CP_HD inline double rng_uniform(Rng& r) {
    uint64_t z = splitmix64(r.state);
    return (z >> 11) * (1.0 / 9007199254740992.0);   // multiply by 2^-53
}

// Standard-normal sample xi ~ N(0,1) via the Box-Muller transform.
//   Brownian dynamics needs a GAUSSIAN kick each step. Box-Muller turns two
//   uniforms (u1,u2) into one normal: sqrt(-2 ln u1) * cos(2 pi u2). We draw a
//   fresh pair every call (we discard the second normal) -- simple and, crucially,
//   it consumes a FIXED number of RNG draws per step, so the host and device
//   streams stay perfectly in lockstep. We guard u1 away from 0 to avoid log(0).
CP_HD inline double rng_normal(Rng& r) {
    double u1 = 1.0 - rng_uniform(r);    // in (0,1], so log(u1) is finite
    double u2 = rng_uniform(r);
    const double TWO_PI = 6.283185307179586476925286766559;
    return sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2);
}

// ---------------------------------------------------------------------------
// ChannelParams: the physical description of the pore + bath + protocol.
//   All lengths in nanometres, energies in kT (so kT = 1 in reduced units),
//   time in the dimensionless step count. Keeping kT=1 makes the Boltzmann
//   factors and the drift term clean and is standard for teaching BD.
// ---------------------------------------------------------------------------
struct ChannelParams {
    double L;            // pore length along z (nm); ion travels z in [0, L]
    int    n_bins;       // number of z-bins for the occupancy histogram
    double U_barrier;    // PMF barrier height at the pore centre (kT)
    double sigma;        // barrier width (nm); Gaussian std-dev of the PMF bump
    double q;            // ion charge (units of e); +1 for K+/Na+, -1 for Cl-
    double V;            // applied transmembrane voltage (reduced: e*V in kT)
    double D;            // diffusion coefficient (nm^2 per step-unit)
    double dt;           // Brownian-dynamics time step (step-unit)
    int    n_steps;      // BD steps per ion trajectory
};

// ---------------------------------------------------------------------------
// pmf_force: the deterministic force  F(z) = -dU/dz  on an ion at depth z.
//   U(z) = U_barrier * exp(-(z-L/2)^2 / (2 sigma^2))  -  q V (z/L)
//   so  dU/dz = U_barrier * exp(...) * ( -(z-L/2)/sigma^2 )  -  q V / L
//   and the force is the negative of that. This single function is THE shared
//   physics: the CPU reference and the GPU kernel both call it, guaranteeing the
//   landscapes are identical. (Real codes read U(z) from an umbrella-sampling
//   PMF table; we use a closed-form Gaussian so the demo is self-contained.)
//   With the minus sign on the field term, the field force +q*V/L pushes a
//   positive ion forward (+z) for positive V -- see the sign note above.
// ---------------------------------------------------------------------------
CP_HD inline double pmf_force(const ChannelParams& P, double z) {
    const double dz_c = z - 0.5 * P.L;                 // distance from pore centre
    const double gauss = P.U_barrier *
                         exp(-(dz_c * dz_c) / (2.0 * P.sigma * P.sigma));
    const double dU_barrier = gauss * (-(dz_c) / (P.sigma * P.sigma));
    const double dU_field   = -P.q * P.V / P.L;         // field gradient (forward for +V)
    return -(dU_barrier + dU_field);                    // F = -dU/dz
}

// ---------------------------------------------------------------------------
// IonResult: what one ion trajectory contributes to the global tallies.
//   We keep these as plain integers so the CPU '+=' and the GPU atomicAdd give
//   bit-identical totals. The occupancy histogram is accumulated separately (it
//   is large), so here we return only the scalar crossing counters.
// ---------------------------------------------------------------------------
struct IonResult {
    unsigned long long fwd;   // forward permeations (z crossed L, current-carrying)
    unsigned long long rev;   // reverse permeations (z fell back through 0)
};

// ---------------------------------------------------------------------------
// simulate_ion: integrate ONE ion's Brownian trajectory and tally it.
//   Inputs:
//     P     : channel/protocol parameters (read-only)
//     rng   : this ion's private RNG stream (advanced in place)
//     occ   : occupancy histogram [n_bins]; we add 1 to occ[bin(z)] every step.
//             On the GPU this is the device tally written with atomicAdd; on the
//             CPU it is a plain array written with '+='. We pass it as a raw
//             pointer plus a tiny "add" callback-free contract: the CALLER chose
//             how to accumulate, but here we just need to know the bin -- so we
//             return nothing for occupancy and instead let the caller bin it.
//
//   To keep occupancy accumulation identical yet allow atomic vs plain adds, we
//   factor the per-step work into `bd_step` (below) and let the kernel / CPU loop
//   own the histogram write. simulate_ion therefore only returns the scalar
//   crossing counts; the loop that calls bd_step does the binning. See the two
//   callers (reference_cpu.cpp and kernels.cu) -- they share bd_step exactly.
//
//   This split mirrors the flagship 5.01 design (shared physics, caller owns the
//   accumulation strategy) and is what makes verification exact.
// ---------------------------------------------------------------------------

// One Brownian-dynamics step applied to position z. Returns the new z and writes
// the crossing deltas through the out-pointers. Pure shared physics; both the
// CPU loop and the GPU thread call this and then bin/accumulate themselves.
CP_HD inline double bd_step(const ChannelParams& P, Rng& rng, double z,
                            unsigned long long* fwd, unsigned long long* rev) {
    // Ermak-McCammon overdamped Langevin update (kT = 1 reduced units, so the
    // mobility D/kT is just D):
    //   drift  = (D) * F(z) * dt           (down the free-energy gradient)
    //   diff   = sqrt(2 D dt) * xi         (thermal Gaussian kick)
    const double F   = pmf_force(P, z);
    const double drift = P.D * F * P.dt;
    const double diff  = sqrt(2.0 * P.D * P.dt) * rng_normal(rng);
    double zn = z + drift + diff;

    // Periodic re-injection with crossing counts: a permeation event is an ion
    // leaving one bath and being replaced from the other (a steady-state current
    // picture). This keeps exactly one ion in the pore at all times -- the
    // single-file, one-ion-at-a-time approximation appropriate for a narrow
    // selectivity filter.
    if (zn >= P.L) {            // crossed the extracellular mouth -> forward current
        (*fwd)++;
        zn -= P.L;             // re-enter from z=0 (carry the overshoot, conserve flux)
    } else if (zn < 0.0) {     // fell back through the intracellular mouth -> reverse
        (*rev)++;
        zn += P.L;             // re-enter from z=L
    }
    return zn;
}

// bin_of: map a continuous position z in [0,L) to an integer histogram bin.
//   Used by both callers to accumulate the occupancy histogram identically.
//   Clamped defensively in case floating round-off pushes z a hair out of range.
CP_HD inline int bin_of(const ChannelParams& P, double z) {
    int b = static_cast<int>((z / P.L) * P.n_bins);
    if (b < 0) b = 0;
    if (b >= P.n_bins) b = P.n_bins - 1;
    return b;
}
