// ===========================================================================
// src/bnct_physics.h  --  Shared (host + device) RNG + BNCT neutron transport
// ---------------------------------------------------------------------------
// Project 5.13 : BNCT Dose Calculation & Optimization  (reduced-scope teaching
//                Monte Carlo; see ../THEORY.md for the full clinical picture).
//
// WHY THIS HEADER IS SHARED
//   Monte Carlo verification here rests on ONE fact: the CPU reference and the
//   GPU kernel must simulate the *identical* neutron histories, so their dose
//   tallies match EXACTLY. That only works if both sides use the same RNG and
//   the same transport logic -- so both live here, in ONE header included by
//   reference_cpu.cpp (host compiler) AND kernels.cu / main.cu (nvcc). This is
//   the "__host__ __device__ shared core" idiom (docs/PATTERNS.md §2), the same
//   trick flagship 5.01 (Monte Carlo dose) uses.
//
//   The BNCT_HD macro expands to `__host__ __device__` under nvcc and to nothing
//   under the plain host compiler, so the same inline functions compile in both
//   worlds. Production BNCT MC (OpenMC, MCNP, GATE, PHITS) uses continuous-energy
//   ENDF/B cross sections and cuRAND; we use a shared counter-based RNG and a
//   small multi-group cross-section model *specifically* so CPU and GPU
//   histories are bit-identical and the demo is deterministic.
//
// ---------------------------------------------------------------------------
// THE SCIENCE IN ONE PARAGRAPH (full derivation in THEORY.md)
//   Boron Neutron Capture Therapy loads a tumor with a ^10B carrier drug, then
//   irradiates it with an epithermal neutron beam. Neutrons slow to thermal
//   energies in tissue; a thermal neutron captured by ^10B triggers
//   ^10B(n,alpha)^7Li, releasing an alpha (~1.47 MeV) and a Li-7 recoil
//   (~0.84 MeV) that together travel < 10 um -- roughly one cell diameter -- so
//   the killing dose is deposited *inside the boron-loaded cell*. The measured
//   physical dose in tissue is the sum of FOUR components, each with a different
//   biological effectiveness (RBE/CBE):
//     (1) BORON dose   : ^10B(n,alpha)^7Li            -- high-LET, CBE ~ 3.8
//     (2) NITROGEN dose: ^14N(n,p)^14C  (0.626 MeV p) -- high-LET, RBE ~ 3.2
//     (3) HYDROGEN/gamma: ^1H(n,gamma)^2H (2.22 MeV)  -- low-LET,  RBE ~ 1.0
//     (4) FAST-neutron : ^1H(n,n')p recoil protons    -- high-LET, RBE ~ 3.2
//   The clinically relevant quantity is the CBE/RBE-*weighted* biological dose
//   (units "Gy-Eq"): D_bio = sum_i w_i * D_i. This project computes all four
//   physical component doses vs. depth AND the weighted biological dose, then
//   verifies GPU == CPU exactly.
//
// ---------------------------------------------------------------------------
// THE SIMPLIFIED TRANSPORT MODEL (deliberately reduced; THEORY.md §algorithm)
//   Geometry: a 1-D slab of tissue, thickness L (cm), split into n_bins depth
//   bins. A monodirectional beam enters at depth 0. We track TWO neutron energy
//   groups (the classic "multi-group" idea, here just fast + thermal):
//
//     * A neutron is born FAST at depth 0. Each step it travels a free path
//       s = -ln(xi)/Sigma_tot_fast and, on interaction, either (a) MODERATES
//       (elastic scatter off hydrogen) -- with some probability it becomes
//       THERMAL, otherwise it stays fast and continues forward, depositing a
//       recoil-proton FAST-neutron dose quantum at the interaction site -- or
//       (b) leaks out the far side.
//     * Once THERMAL, the neutron random-walks with the thermal total cross
//       section until it is CAPTURED. At capture we sample WHICH nucleus
//       captured it, weighted by that nucleus's macroscopic capture cross
//       section Sigma_a = N * sigma_a:
//           ^10B -> BORON dose quantum   (Q ~ 2.31 MeV shared alpha+Li)
//           ^14N -> NITROGEN dose quantum (Q ~ 0.626 MeV proton)
//           ^1H  -> HYDROGEN/GAMMA quantum (2.22 MeV capture gamma)
//       Boron's capture cross section is huge (sigma_a ~ 3837 barn at 0.025 eV)
//       so even a few tens of ppm of ^10B dominate the thermal capture rate in
//       the tumor -- the whole physical basis of BNCT selectivity.
//
//   Energy is deposited as INTEGER keV quanta so the dose tallies are exact
//   under atomic add (integer adds commute -> GPU order-independent -> matches
//   CPU exactly; docs/PATTERNS.md §3). Physical dose (Gy) is a post-processing
//   scale applied identically on both sides.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh. READ BEFORE:
// reference_cpu.h, kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <cstdint>
#include <cmath>

#ifdef __CUDACC__
#define BNCT_HD __host__ __device__
#else
#define BNCT_HD
#endif

// ---------------------------------------------------------------------------
// Dose-component indices. We tally four physical dose components separately so
// we can apply their different biological weights afterward. Keep these in a
// fixed order: main.cu and the reports rely on it.
// ---------------------------------------------------------------------------
enum DoseComponent {
    DC_BORON   = 0,   // ^10B(n,alpha)^7Li      -- the therapeutic component
    DC_NITROGEN= 1,   // ^14N(n,p)^14C          -- background, high-LET
    DC_GAMMA   = 2,   // ^1H(n,gamma)^2H        -- background, low-LET
    DC_FAST    = 3,   // fast-neutron recoil p  -- background, high-LET
    DC_COUNT   = 4    // number of components (array sizing)
};

// ---------------------------------------------------------------------------
// RNG: a splitmix64 counter-based stream (identical on host and device).
//   We roll our own instead of cuRAND for ONE reason: the exact same integer
//   math must run on the CPU and the GPU so both simulate bit-identical
//   histories and the dose tallies match exactly. splitmix64 is tiny, fast,
//   and has good statistical quality for a teaching MC.
// ---------------------------------------------------------------------------
struct Rng { uint64_t state; };

// One splitmix64 step: advance `x` and return a well-mixed 64-bit value.
BNCT_HD inline uint64_t splitmix64(uint64_t& x) {
    x += 0x9E3779B97F4A7C15ULL;                 // add the golden-ratio odd constant
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; // avalanche the bits...
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Seed an independent stream for neutron history `history`, so each history is
// uncorrelated yet reproducible from (base, history) -- essential for exact
// CPU/GPU agreement regardless of thread scheduling.
BNCT_HD inline Rng rng_seed(uint64_t base, uint64_t history) {
    Rng r;
    r.state = base ^ (history * 0x9E3779B97F4A7C15ULL + 0xD1B54A32D192ED03ULL);
    splitmix64(r.state);     // warm up so nearby seeds diverge immediately
    return r;
}

// Uniform double in [0,1) from 53 random bits (identical math host/device).
BNCT_HD inline double rng_uniform(Rng& r) {
    uint64_t z = splitmix64(r.state);
    return (z >> 11) * (1.0 / 9007199254740992.0);   // multiply by 2^-53
}

// ---------------------------------------------------------------------------
// SimParams: everything that defines a BNCT transport problem. Loaded from the
// one-line sample file (see data/README.md). All cross sections here are
// MACROSCOPIC (Sigma, units 1/cm) = number density N * microscopic sigma; we
// pre-fold the boron concentration into Sigma_a_B so the file stays simple.
// ---------------------------------------------------------------------------
struct SimParams {
    double L;              // slab thickness (cm)
    int    n_bins;         // number of depth bins along the slab

    // --- FAST group (incident/epithermal neutrons slowing down) ---
    double Sig_s_fast;     // fast-neutron macroscopic scatter cross section (1/cm)
    double p_thermalize;   // P(a fast scatter thermalizes the neutron) in [0,1]

    // --- THERMAL group (captured by boron / nitrogen / hydrogen) ---
    double Sig_a_B;        // ^10B macroscopic capture Sigma_a (1/cm) -- carrier-loaded
    double Sig_a_N;        // ^14N macroscopic capture Sigma_a (1/cm)
    double Sig_a_H;        // ^1H  macroscopic capture Sigma_a (1/cm)
    double Sig_s_th;       // thermal-neutron scatter Sigma_s (1/cm) (random walk)

    // --- energy released per reaction, in INTEGER keV quanta ---
    // (Integer so the dose tally is exact under atomic add. Real Q-values are
    //  2.31 MeV effective for boron, 0.626 MeV for nitrogen, 2.224 MeV for the
    //  hydrogen capture gamma, and we credit ~0.5 MeV per fast recoil step.)
    unsigned int Q_boron_keV;   // e.g. 2310
    unsigned int Q_nitro_keV;   // e.g. 626
    unsigned int Q_gamma_keV;   // e.g. 2224
    unsigned int Q_fast_keV;    // e.g. 500 per fast-scatter recoil deposit
};

// ---------------------------------------------------------------------------
// A single energy deposit recorded by a history: which depth bin, which dose
// component, and how many keV quanta. simulate_neutron() fills an array of
// these; the caller applies them to the tally (atomicAdd on GPU, += on CPU).
// ---------------------------------------------------------------------------
struct Deposit {
    int bin;            // depth-bin index [0, n_bins)
    int component;      // one of DoseComponent
    unsigned int keV;   // integer energy quanta deposited here
};

// Max deposits a single neutron history can record. A fast neutron may scatter
// several times (each a fast-dose deposit) before thermalizing and being
// captured (one final capture deposit); 64 is a safe, generous cap.
static const int BNCT_MAX_DEPOSITS = 64;

// ---------------------------------------------------------------------------
// clamp_bin: map a depth z (cm) to a valid depth-bin index. Pulled out so the
// boundary logic is written exactly once and is identical on both sides.
// ---------------------------------------------------------------------------
BNCT_HD inline int clamp_bin(double z, double dz, int n_bins) {
    int bin = static_cast<int>(z / dz);
    if (bin < 0) bin = 0;
    if (bin >= n_bins) bin = n_bins - 1;
    return bin;
}

// ---------------------------------------------------------------------------
// simulate_neutron: run ONE neutron history through the two-group slab model
// and record its energy deposits. Returns the number of deposits written.
//
//   P    : the slab + cross-section parameters (const).
//   rng  : this history's private, reproducible RNG stream (advanced in place).
//   dep  : caller-owned scratch array of length >= BNCT_MAX_DEPOSITS.
//
// The physics is intentionally simple but self-consistent (see THEORY.md):
//   * Born FAST at z=0 moving in +z. Loop:
//       - sample a fast free path s = -ln(xi)/Sig_s_fast; advance z += s.
//       - if z >= L: neutron leaks out the far side -> history ends.
//       - else it scatters: with prob p_thermalize it becomes THERMAL (break to
//         the thermal phase); otherwise it stays fast, deposits ONE fast-recoil
//         quantum (Q_fast) at this bin, and continues forward.
//   * THERMAL phase: random-walk with total thermal Sigma_tot = Sig_s_th +
//     Sig_a_B + Sig_a_N + Sig_a_H. Each step:
//       - sample free path with Sig_tot; take an ISOTROPIC-in-1D step (+/- z with
//         equal probability -- a 1-D diffusion surrogate); clamp to the slab.
//       - decide scatter vs. capture by the ratio Sig_s_th / Sig_tot.
//       - on CAPTURE, sample the capturing nuclide by its Sigma_a share and
//         deposit that reaction's Q into the matching component -> history ends.
// This captures the BNCT essentials: boron's giant capture cross section makes
// DC_BORON dominate wherever ^10B is present, while the background N/H/fast
// components set the healthy-tissue dose floor.
// ---------------------------------------------------------------------------
BNCT_HD inline int simulate_neutron(const SimParams& P, Rng& rng, Deposit* dep) {
    const double dz = P.L / P.n_bins;    // depth-bin thickness (cm)
    double z = 0.0;                       // current depth (cm)
    int nd = 0;                           // number of deposits recorded so far

    // -------- FAST phase: slow down toward thermal ------------------------
    bool thermal = false;
    for (int guard = 0; guard < 100000 && !thermal; ++guard) {
        const double xi = 1.0 - rng_uniform(rng);   // in (0,1], avoids log(0)
        z += -log(xi) / P.Sig_s_fast;               // exponential free path
        if (z >= P.L || z < 0.0) return nd;          // leaked out of the slab
        // A fast scatter happened at depth z. Does it thermalize the neutron?
        if (rng_uniform(rng) < P.p_thermalize) {
            thermal = true;                          // enter the thermal walk
        } else {
            // Still fast: credit a recoil-proton fast-neutron dose quantum here.
            int bin = clamp_bin(z, dz, P.n_bins);
            if (nd < BNCT_MAX_DEPOSITS) { dep[nd].bin = bin; dep[nd].component = DC_FAST; dep[nd].keV = P.Q_fast_keV; ++nd; }
        }
    }
    if (!thermal) return nd;   // never thermalized within the guard -> done

    // -------- THERMAL phase: random-walk until captured -------------------
    const double Sig_tot = P.Sig_s_th + P.Sig_a_B + P.Sig_a_N + P.Sig_a_H;
    if (Sig_tot <= 0.0) return nd;                  // degenerate params guard
    const double p_scatter = P.Sig_s_th / Sig_tot;  // P(this interaction scatters)

    for (int guard = 0; guard < 100000; ++guard) {
        const double xi = 1.0 - rng_uniform(rng);
        const double step = -log(xi) / Sig_tot;     // thermal free path (cm)
        // 1-D isotropic surrogate: go +z or -z with equal probability so the
        // thermal cloud spreads both ways (real thermal neutrons diffuse).
        z += (rng_uniform(rng) < 0.5) ? step : -step;
        if (z >= P.L) return nd;                    // leaked out the far side
        if (z < 0.0)  return nd;                    // back-scattered out the front

        if (rng_uniform(rng) < p_scatter) {
            continue;                                // scatter: keep walking
        }
        // -------- CAPTURE: pick the capturing nuclide by Sigma_a share -----
        int bin = clamp_bin(z, dz, P.n_bins);
        const double Sig_cap = P.Sig_a_B + P.Sig_a_N + P.Sig_a_H;
        double pick = rng_uniform(rng) * Sig_cap;   // uniform over capture space
        int comp; unsigned int q;
        if (pick < P.Sig_a_B) {                     // captured by ^10B (the goal)
            comp = DC_BORON;   q = P.Q_boron_keV;
        } else if (pick < P.Sig_a_B + P.Sig_a_N) {  // captured by ^14N
            comp = DC_NITROGEN; q = P.Q_nitro_keV;
        } else {                                     // captured by ^1H -> gamma
            comp = DC_GAMMA;    q = P.Q_gamma_keV;
        }
        if (nd < BNCT_MAX_DEPOSITS) { dep[nd].bin = bin; dep[nd].component = comp; dep[nd].keV = q; ++nd; }
        return nd;                                   // neutron absorbed: history ends
    }
    return nd;   // ran out of guard steps (should not happen with sane params)
}

// ---------------------------------------------------------------------------
// Biological weights (RBE for the neutron/gamma background, CBE for boron).
// These are representative literature values for BNCT with a BPA carrier; they
// convert physical dose (Gy) per component into biologically weighted dose
// (Gy-Eq). Kept as compile-time constants shared by both sides so the weighted
// dose is computed with identical arithmetic. (THEORY.md §real-world.)
//   D_bio = w_B*D_B + w_N*D_N + w_gamma*D_gamma + w_fast*D_fast
// Weights are stored x1000 as INTEGERS so the biological weighting stays exact
// integer math (no float divergence between CPU and GPU).
// ---------------------------------------------------------------------------
BNCT_HD inline unsigned int bio_weight_milli(int component) {
    switch (component) {
        case DC_BORON:    return 3800;  // CBE ~ 3.8 (BPA in tumor)
        case DC_NITROGEN: return 3200;  // RBE ~ 3.2 (high-LET proton)
        case DC_GAMMA:    return 1000;  // RBE ~ 1.0 (low-LET photon)
        case DC_FAST:     return 3200;  // RBE ~ 3.2 (high-LET recoil proton)
        default:          return 1000;
    }
}
