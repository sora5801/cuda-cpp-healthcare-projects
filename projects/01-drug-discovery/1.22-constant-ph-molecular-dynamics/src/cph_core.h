// ===========================================================================
// src/cph_core.h  --  Shared (host + device) constant-pH titration physics
// ---------------------------------------------------------------------------
// Project 1.22 : Constant-pH Molecular Dynamics (reduced-scope teaching model)
//
// WHY THIS HEADER IS SHARED  (the HD-core idiom, PATTERNS.md §2)
//   The whole point of the verification in this project is that the CPU
//   reference and the GPU kernel run the *identical* Monte Carlo chains, so the
//   protonation statistics they tally must match BIT-FOR-BIT. That only works if
//   both sides use the same RNG and the same energy function -- so both live
//   here, in ONE header included by reference_cpu.cpp (compiled by the host C++
//   compiler) AND by kernels.cu / main.cu (compiled by nvcc).
//
//   The CPH_HD macro expands to `__host__ __device__` under nvcc and to nothing
//   under a plain host compiler, so the same inline functions compile in both
//   worlds. Keep CUDA-only constructs (`__global__`, `<<<>>>`) OUT of this header
//   so the host compiler can include it.
//
// THE SCIENCE IN ONE PARAGRAPH (full derivation in ../THEORY.md)
//   A titratable residue (Asp, Glu, His, Cys, Lys, ...) can be protonated (it
//   holds an extra H+, net charge q_prot) or deprotonated (it has released the
//   H+, net charge q_deprot). Which state it favours depends on the solution pH
//   versus the residue's *intrinsic* pKa, AND on the electrostatic field of the
//   OTHER charged residues nearby -- a protonated (more positive) neighbour makes
//   it harder to protonate this one. "Constant-pH" simulation samples the joint
//   distribution over all residues' protonation states at a fixed pH. We do that
//   with Metropolis Monte Carlo: repeatedly propose flipping one residue's
//   protonation, accept or reject by the Metropolis rule, and tally how often
//   each residue ends up protonated. The fraction-protonated vs pH is the
//   titration curve; the pH at which a residue is 50% protonated is its
//   (coupling-shifted) pKa.
//
//   THIS IS A DELIBERATELY REDUCED TEACHING MODEL. Production CpHMD (AMBER
//   pmemd.cuda) runs full molecular dynamics and recomputes solvation energies
//   each move; here the "energy" is a compact analytic surrogate (intrinsic pKa
//   term + pairwise Coulomb coupling on FIXED positions). The Monte Carlo
//   machinery, the pH coupling, and the titration-curve / pKa readout are the
//   real, transferable ideas. See ../THEORY.md "Where this sits in the real
//   world".
// ===========================================================================
#pragma once

#include <cstdint>
#include <cmath>

// --- The HD decorator: real on nvcc, empty on the host compiler --------------
#ifdef __CUDACC__
#define CPH_HD __host__ __device__
#else
#define CPH_HD
#endif

// ln(10): converts a pKa/pH difference (base-10) into the natural-log units used
// by the Boltzmann factor exp(-dG/kT). Appears in the standard CpHMD energy.
#ifndef CPH_LN10
#define CPH_LN10 2.302585092994045901   // ln(10) to ~double precision
#endif

// Hard caps so per-thread state lives in registers/local arrays of known size.
// A teaching system has only a handful of titratable residues.
static const int CPH_MAX_RESIDUES = 16;   // titratable residues per system

// ---------------------------------------------------------------------------
// Residue: the fixed (pH-independent) parameters of one titratable site.
//   * pKa_intrinsic : the "model" pKa the residue would show in isolation
//                     (e.g. Asp ~ 4.0, His ~ 6.5, Lys ~ 10.5). Dimensionless.
//   * q_prot / q_deprot : the net charge (in units of e) of the protonated and
//                     deprotonated forms. For an ACID (Asp/Glu/Cys) the
//                     protonated form is neutral (0) and deprotonated is -1; for
//                     a BASE (His/Lys) protonated is +1 and deprotonated is 0.
//                     We store both explicitly so acids and bases share one code
//                     path -- the electrostatics only cares about the charge.
//   * x, y, z       : a fixed 3-D position (Angstrom). Real CpHMD moves these
//                     with MD; we freeze them so the demo is reproducible and the
//                     coupling is a pure function of protonation state.
// ---------------------------------------------------------------------------
struct Residue {
    double pKa_intrinsic;   // model/intrinsic pKa (dimensionless)
    double q_prot;          // net charge when PROTONATED   (units of e)
    double q_deprot;        // net charge when DEPROTONATED  (units of e)
    double x, y, z;         // fixed position (Angstrom)
};

// ---------------------------------------------------------------------------
// CphSystem: a complete titration problem -- the residues plus the run controls
// that are shared by CPU and GPU. (How many pH values and replicas to run is a
// host-side concern and lives in reference_cpu.h, not here.)
//   * coulomb_k : Coulomb prefactor k = 332.06 / epsilon  (kcal*A / (mol*e^2)).
//                 332.06 converts e^2/Angstrom to kcal/mol; epsilon is an
//                 effective dielectric that screens the interaction (water ~ 80,
//                 protein interior ~ 4..20). Pre-divided so the inner loop is one
//                 multiply. Set coulomb_k = 0 to switch coupling OFF (then each
//                 residue titrates at exactly its intrinsic pKa -- the analytic
//                 check in main.cu).
//   * kT        : thermal energy k_B*T in kcal/mol (~0.593 at 298 K). Sets the
//                 scale of the Metropolis acceptance.
//   * n_res     : number of residues actually used (<= CPH_MAX_RESIDUES).
//   * sweeps    : Monte Carlo sweeps per chain (1 sweep = n_res attempted flips).
//   * burn_in   : leading sweeps discarded before tallying (equilibration).
// ---------------------------------------------------------------------------
struct CphSystem {
    Residue res[CPH_MAX_RESIDUES];
    double  coulomb_k;   // Coulomb prefactor 332.06/epsilon
    double  kT;          // thermal energy (kcal/mol)
    int     n_res;       // number of titratable residues in use
    int     sweeps;      // MC sweeps per (system,pH,replica) chain
    int     burn_in;     // sweeps discarded before tallying
};

// ===========================================================================
// RNG: a splitmix64 counter-based stream -- identical math on host and device.
//   We do NOT use cuRAND here on purpose: a tiny, self-contained, deterministic
//   generator lets the CPU reproduce the GPU's *exact* random decisions, which
//   is what makes the verification an EXACT integer match rather than a fuzzy
//   statistical one. (Production codes use cuRAND; see THEORY.md.)
// ===========================================================================
struct Rng { uint64_t state; };

// One splitmix64 step: advance the state and return a well-mixed 64-bit value.
// Chosen because it is one of the simplest generators with good avalanche and
// no warm-up/seeding subtleties -- ideal for a portable host==device RNG.
CPH_HD inline uint64_t splitmix64(uint64_t& x) {
    x += 0x9E3779B97F4A7C15ULL;                 // golden-ratio odd increment
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// chain_id: pack a (pH index k, replica r) pair into one unique 64-bit id so the
// RNG stream of every chain is reproducible. The CPU loop (reference_cpu.cpp) and
// the GPU thread (kernels.cu) BOTH call this, so chain (k,r) draws the identical
// random numbers on both sides -- the foundation of the exact-match verification.
// The large stride on k guarantees no (k,r) collision for any sane replica count.
CPH_HD inline uint64_t chain_id(int k, int r) {
    return static_cast<uint64_t>(k) * 1000003ULL + static_cast<uint64_t>(r);
}

// Seed an independent stream from a (base, chain) pair so every Monte Carlo
// chain is uncorrelated yet fully reproducible from its indices. `chain` encodes
// (system, pH index, replica) into one 64-bit id via chain_id() above.
CPH_HD inline Rng rng_seed(uint64_t base, uint64_t chain) {
    Rng r;
    r.state = base ^ (chain * 0x9E3779B97F4A7C15ULL + 0xD1B54A32D192ED03ULL);
    splitmix64(r.state);   // one warm-up step to decorrelate nearby chain ids
    return r;
}

// Uniform double in [0,1) from the top 53 bits (the full mantissa). Identical
// host/device math, so a Metropolis "accept if u < p" decision is reproducible.
CPH_HD inline double rng_uniform(Rng& r) {
    uint64_t z = splitmix64(r.state);
    return (z >> 11) * (1.0 / 9007199254740992.0);   // multiply by 2^-53
}

// ===========================================================================
// THE ENERGY FUNCTION (the one true formula shared by CPU and GPU)
// ---------------------------------------------------------------------------
// charge_of: the net charge of residue i given its protonation bit.
//   prot == 1 -> protonated form charge; prot == 0 -> deprotonated form charge.
// ---------------------------------------------------------------------------
CPH_HD inline double charge_of(const CphSystem& S, int i, int prot) {
    return prot ? S.res[i].q_prot : S.res[i].q_deprot;
}

// ---------------------------------------------------------------------------
// pairwise_distance: Euclidean distance between residues i and j (Angstrom).
//   Used by the Coulomb coupling term. Positions are fixed, so this is constant
//   across the chain -- a real code would cache it; we keep it inline and clear.
// ---------------------------------------------------------------------------
CPH_HD inline double pairwise_distance(const CphSystem& S, int i, int j) {
    double dx = S.res[i].x - S.res[j].x;
    double dy = S.res[i].y - S.res[j].y;
    double dz = S.res[i].z - S.res[j].z;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

// ---------------------------------------------------------------------------
// delta_G_flip: the free-energy change (kcal/mol) of FLIPPING residue `i` from
// its current protonation `state[i]` to the opposite, at the given pH, holding
// every other residue fixed. This single function is the heart of the model and
// is what both the CPU loop and the GPU thread call for each Metropolis step.
//
//   dG = dG_intrinsic + dG_coupling
//
//   (1) INTRINSIC pH TERM. For a protonation change, standard CpHMD writes the
//       reference (decoupled) free energy as
//            dG_intr(protonate)  =  -kT * ln(10) * (pKa - pH)
//       i.e. protonating is favourable (negative dG) when pH < pKa, and costly
//       when pH > pKa -- exactly the Henderson-Hasselbalch behaviour. We compute
//       the dG for going protonated and negate it when the move is the reverse.
//
//   (2) COUPLING TERM. The electrostatic interaction of residue i with every
//       other residue j depends on i's charge, which changes when we flip it.
//       The flip changes i's charge by  dq_i = q_new - q_old, so the coupling
//       contribution to the flip energy is
//            dG_coup = sum_{j != i}  k * dq_i * q_j(state_j) / r_ij
//       (k = 332.06/epsilon from CphSystem). This is what shifts a residue's
//       apparent pKa away from its intrinsic value -- the physics this project
//       exists to show.
//
//   Sign convention: dG is the energy of (proposed state) minus (current state).
//   The Metropolis rule then accepts with probability min(1, exp(-dG/kT)).
// ---------------------------------------------------------------------------
CPH_HD inline double delta_G_flip(const CphSystem& S, const int* state,
                                  int i, double pH) {
    const int cur = state[i];          // current protonation bit (0/1)
    const int prop = cur ^ 1;          // proposed bit = the opposite

    // (1) Intrinsic term. Define dG for the *protonation* direction, then orient
    //     it to the actual move. protonate is favoured when pH < pKa.
    const double dG_protonate =
        -S.kT * CPH_LN10 * (S.res[i].pKa_intrinsic - pH);
    // If the proposed move IS protonation (prop==1, cur==0) use +dG_protonate;
    // if it is deprotonation (prop==0) the energy is the negative of that.
    const double dG_intrinsic = prop ? dG_protonate : -dG_protonate;

    // (2) Coupling term. The change in residue i's charge upon the flip...
    const double dq_i = charge_of(S, i, prop) - charge_of(S, i, cur);
    double dG_coupling = 0.0;
    // ...interacts with every OTHER residue's current charge over distance r_ij.
    for (int j = 0; j < S.n_res; ++j) {
        if (j == i) continue;                       // no self-interaction
        const double qj = charge_of(S, j, state[j]);
        if (qj == 0.0 || dq_i == 0.0) continue;     // skip neutral pairs (no force)
        const double r = pairwise_distance(S, i, j);
        // Guard against a degenerate zero distance (coincident test positions).
        const double r_safe = (r > 1e-6) ? r : 1e-6;
        dG_coupling += S.coulomb_k * dq_i * qj / r_safe;
    }

    return dG_intrinsic + dG_coupling;
}

// ---------------------------------------------------------------------------
// run_chain: simulate ONE Metropolis Monte Carlo titration chain for a single
// (system, pH) at a given seed, and return how many *post-burn-in sweeps* left
// each residue PROTONATED. Writing the answer as integer counts (not a float
// fraction) is deliberate: integer accumulation is order-independent, so the GPU
// and CPU agree EXACTLY (PATTERNS.md §3/§4). The caller divides by the number of
// tallied sweeps to get the fraction protonated.
//
//   S        : the system (residues + controls), shared host/device.
//   pH       : the fixed pH of this chain.
//   rng      : a per-chain RNG stream (seeded by the caller from its indices).
//   prot_count : OUT, length n_res; prot_count[i] += 1 for every tallied sweep
//                in which residue i ended protonated. Caller zeroes it first.
//   returns  : the number of sweeps that were tallied (sweeps - burn_in), so the
//              caller can form the fraction without re-deriving it.
//
// Algorithm (one sweep = attempt to flip every residue once, in index order):
//   for sweep in 0..sweeps:
//     for i in 0..n_res:
//       dG = delta_G_flip(...); u = uniform();
//       if (dG <= 0) OR (u < exp(-dG/kT)) : accept -> flip state[i]
//     if sweep >= burn_in: for each i, prot_count[i] += state[i]
//
// Complexity: O(sweeps * n_res^2) per chain (the n_res factor is the coupling
// sum). Chains are fully independent -> embarrassingly parallel across the
// ensemble, which is exactly why this maps so cleanly onto one GPU thread/chain.
// ---------------------------------------------------------------------------
CPH_HD inline int run_chain(const CphSystem& S, double pH, Rng& rng,
                            int* prot_count) {
    // Per-chain protonation state, one bit per residue. Lives in registers/local
    // memory on the GPU (n_res is tiny). Start fully DEPROTONATED so the burn-in
    // has to climb to equilibrium -- identical start on host and device.
    int state[CPH_MAX_RESIDUES];
    for (int i = 0; i < S.n_res; ++i) state[i] = 0;

    int tallied = 0;
    for (int sweep = 0; sweep < S.sweeps; ++sweep) {
        // --- one sweep: attempt a flip on every residue in turn ---
        for (int i = 0; i < S.n_res; ++i) {
            const double dG = delta_G_flip(S, state, i, pH);
            // Metropolis: always accept downhill moves; accept uphill ones with
            // probability exp(-dG/kT). Drawing `u` even for downhill moves keeps
            // the RNG stream position identical between host and device.
            const double u = rng_uniform(rng);
            const bool accept = (dG <= 0.0) || (u < exp(-dG / S.kT));
            if (accept) state[i] ^= 1;     // commit the flip
        }
        // --- tally after equilibration ---
        if (sweep >= S.burn_in) {
            for (int i = 0; i < S.n_res; ++i) prot_count[i] += state[i];
            ++tallied;
        }
    }
    return tallied;
}
