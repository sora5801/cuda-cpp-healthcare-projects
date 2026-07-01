// ===========================================================================
// src/ts_physics.h  --  Shared (host + device) RNG + track-structure transport
// ---------------------------------------------------------------------------
// Project 5.11 : Microdosimetry & Track-Structure Simulation
//
// WHY THIS HEADER IS SHARED (the load-bearing idea)
//   Microdosimetry is intrinsically STOCHASTIC: two identical particles deposit
//   energy in completely different patterns. The only way to VERIFY a Monte Carlo
//   result is to make the CPU reference and the GPU kernel replay the *identical*
//   particle histories and then check that their tallies agree EXACTLY. That
//   requires both sides to use the same random-number generator and the same
//   transport logic -- so both live here, in ONE header, included by
//   reference_cpu.cpp (host compiler) AND kernels.cu / main.cu (nvcc).
//
//   The TS_HD macro expands to `__host__ __device__` under nvcc and to nothing
//   under the plain host compiler, so the same inline functions compile in both
//   worlds (PATTERNS.md §2, the "HD-macro idiom"). Production GPU track-structure
//   codes (Geant4-DNA, MPEXS-DNA) use cuRAND Philox; we use a shared, reproducible
//   counter-based RNG on purpose, so the CPU and GPU histories are bit-identical
//   and the demo is deterministic -- see THEORY.md.
//
// -------------------------------------------------------------------------
// THE SIMPLIFIED PHYSICS  (a deliberately reduced TEACHING model; the real
// event-by-event physics is described in THEORY.md "Where this sits in the
// real world" -- do NOT read this as validated radiobiology.)
//
//   We follow the "condensed-history + discrete-ionization" idea that underlies
//   every track-structure code, stripped to the essentials a learner can hold in
//   their head:
//
//   * A primary charged particle (electron / proton / alpha, distinguished only
//     by its LET class here) enters a cube of LIQUID WATER of side `box_nm`
//     (nanometres). Water is the standard tissue surrogate: ~70% of a cell is
//     water and DNA sits in it.
//
//   * The particle travels in a nearly-straight line (heavy ions barely deflect;
//     we add small Gaussian angular scatter for electrons). Along the path it
//     produces IONIZATION events. The mean free path between ionizations is
//     `1 / (Sigma)` where Sigma (events per nm) is set by the particle's LET:
//     high-LET tracks (alpha, carbon) ionize densely; low-LET tracks (fast
//     electrons) ionize sparsely. Step length is sampled exponentially,
//     s = -ln(xi)/Sigma  -- the same free-path law as any Monte Carlo transport.
//
//   * Each ionization deposits an INTEGER number of energy quanta (we quantise
//     the local energy loss in units of `quantum_eV`, ~ the mean energy per
//     ionization in water, so the whole simulation stays in integers and the
//     dose/energy tallies are EXACT under atomic add -- PATTERNS.md §3).
//
//   * Each ionization also emits, with some probability, a short-range SECONDARY
//     electron ("delta ray") that deposits a small extra cluster nearby. This is
//     the microscopic origin of clustered damage.
//
//   * DNA is modelled as a set of scoring segments laid along one axis. An
//     ionization within `dna_radius_nm` of the DNA axis is a candidate STRAND
//     BREAK. Following the classic combinatorial DNA-damage model:
//        - a single break in a segment  -> SSB  (single-strand break),
//        - two breaks in the SAME segment on OPPOSITE strands within a short
//          window (10 base pairs ~ 3.4 nm) -> DSB (double-strand break, the
//          lethal lesion). DSBs are what make high-LET radiation biologically
//          more effective (higher RBE): dense tracks create DSBs; sparse tracks
//          mostly make repairable SSBs.
//
//   * MICRODOSIMETRY proper: we also record the total energy imparted to the
//     whole box by each track. Divided by the mean chord length of the box this
//     is the LINEAL ENERGY y (keV/µm) -- the fundamental microdosimetric
//     quantity, the thing a Rossi proportional counter or a TEPC measures. We
//     bin y across many tracks to form the y-spectrum f(y); its dose-mean yD is
//     the summary number radiobiologists quote.
//
//   Every per-track quantity is an INTEGER count (quanta, breaks), so summing
//   across millions of tracks with atomicAdd is order-independent -> the GPU
//   tally equals the CPU tally to the bit. That exactness is the whole reason we
//   quantise energy.
// ===========================================================================
#pragma once

#include <cstdint>
#include <cmath>

#ifdef __CUDACC__
#define TS_HD __host__ __device__
#else
#define TS_HD
#endif

// ---------------------------------------------------------------------------
// RNG: splitmix64, a counter-based stream that is identical on host and device.
//   Counter-based (as opposed to a long stateful Mersenne Twister) is exactly
//   what GPUs want: thread `i` derives its own independent stream from `i`, with
//   no shared mutable state and no correlation between threads. Production codes
//   use cuRAND Philox for the same reason; splitmix64 is the smallest thing that
//   teaches the idea and stays bit-identical across CPU and GPU.
// ---------------------------------------------------------------------------
struct Rng { uint64_t state; };

// One splitmix64 step: advance `x` in place and return a well-mixed 64-bit word.
TS_HD inline uint64_t splitmix64(uint64_t& x) {
    x += 0x9E3779B97F4A7C15ULL;              // odd additive constant (golden ratio)
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;  // avalanche mixing steps
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Seed an INDEPENDENT stream for track `track` from a base seed, so every
// primary particle is uncorrelated yet exactly reproducible from (base, track).
TS_HD inline Rng rng_seed(uint64_t base, uint64_t track) {
    Rng r;
    r.state = base ^ (track * 0x9E3779B97F4A7C15ULL + 0xD1B54A32D192ED03ULL);
    splitmix64(r.state);   // warm-up so nearby seeds diverge immediately
    return r;
}

// Uniform double in [0,1) built from 53 random bits (identical math host/device).
TS_HD inline double rng_uniform(Rng& r) {
    uint64_t z = splitmix64(r.state);
    return (z >> 11) * (1.0 / 9007199254740992.0);   // multiply by 2^-53
}

// Standard-normal sample via Box-Muller. Used for the small transverse scatter
// of the track (electrons wander; heavy ions barely do). Only the first of the
// pair is returned -- simplicity over squeezing out the second value.
TS_HD inline double rng_normal(Rng& r) {
    double u1 = 1.0 - rng_uniform(r);   // in (0,1] so log() is finite
    double u2 = rng_uniform(r);
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

// ---------------------------------------------------------------------------
// Simulation parameters (loaded from the data file). All lengths in nanometres,
// energies in electron-volts, so the numbers stay human-scale for the reader.
// ---------------------------------------------------------------------------
struct TrackParams {
    double box_nm;          // side of the cubic water scoring volume (nm)
    double sigma_ion;       // MEAN ionizations per nm along the track (1/nm)
                            //   -- this number encodes the particle's LET class
    double let_spread;      // per-track LET variability (lognormal sigma). Each
                            //   primary samples its own local density
                            //   sigma_ion * exp(let_spread * N(0,1)), modelling a
                            //   MIXED radiation field / LET straggling so the
                            //   lineal-energy spectrum has a realistic tail.
                            //   0 => every track identical (a pure mono-LET beam).
    double quantum_eV;      // energy per quantum (~ mean energy per ionization)
    int    quanta_per_ion;  // integer quanta deposited at each primary ionization
    double p_delta;         // probability an ionization launches a delta-ray
    int    delta_quanta;    // quanta deposited by a delta-ray cluster
    double dna_radius_nm;   // ionization within this distance of DNA axis => break
    int    n_dna_segments;  // number of DNA scoring segments along the box y-axis
    int    n_y_bins;        // number of lineal-energy (y) histogram bins
    double y_max_keV_um;    // upper edge of the y histogram (keV/µm)
};

// A single track can record at most this many DNA break events; a fixed cap lets
// each thread use a small on-stack scratch array (no dynamic allocation on the
// GPU). Tracks that would exceed it simply stop recording further breaks -- an
// honest, documented truncation for the teaching model.
static const int TS_MAX_BREAKS = 96;

// One recorded strand-break candidate: which DNA segment, which strand (0/1),
// and the position along the DNA axis (nm) so we can test the DSB proximity rule.
struct BreakEvent {
    int    segment;   // DNA segment index [0, n_dna_segments)
    int    strand;    // 0 or 1 (the two complementary strands of the double helix)
    double pos_nm;    // position along the DNA axis (nm)
};

// The per-track OUTCOME the caller will fold into the global tallies. Everything
// here is an integer or a small fixed-point-able count, chosen so the reduction
// across tracks is exact and order-independent (PATTERNS.md §3).
struct TrackResult {
    unsigned long long energy_quanta;  // total quanta imparted to the box by this track
    int ssb;                           // single-strand breaks produced
    int dsb;                           // double-strand breaks produced
    int y_bin;                         // which lineal-energy bin this track lands in
};

// ---------------------------------------------------------------------------
// classify_breaks: apply the combinatorial DNA-damage model to the list of
//   break candidates a track produced. Two breaks in the SAME segment on
//   OPPOSITE strands whose positions are within `dsb_window_nm` form a DSB;
//   the rest are SSBs. Written as an O(k^2) scan over the (small, capped) break
//   list -- clarity over cleverness; k <= TS_MAX_BREAKS.
//
//   dsb_window_nm = 3.4 nm ~ 10 base pairs, the standard proximity threshold in
//   the literature for calling two breaks a DSB. Each break is "consumed" once
//   so we never double-count a break into both a DSB and an SSB.
// ---------------------------------------------------------------------------
TS_HD inline void classify_breaks(const BreakEvent* br, int k,
                                  double dsb_window_nm,
                                  int& out_ssb, int& out_dsb) {
    // `used[j]` marks a break already paired into a DSB. Small fixed array so it
    // lives in registers/local memory on the device -- no allocation.
    bool used[TS_MAX_BREAKS];
    for (int j = 0; j < k; ++j) used[j] = false;

    int dsb = 0;
    for (int a = 0; a < k; ++a) {
        if (used[a]) continue;
        for (int b = a + 1; b < k; ++b) {
            if (used[b]) continue;
            // DSB rule: same segment, opposite strands, close along the axis.
            if (br[a].segment == br[b].segment &&
                br[a].strand  != br[b].strand  &&
                fabs(br[a].pos_nm - br[b].pos_nm) <= dsb_window_nm) {
                used[a] = used[b] = true;   // consume both breaks
                ++dsb;
                break;                       // break `a` is now spent; move on
            }
        }
    }
    // Every break not consumed by a DSB counts as a single-strand break.
    int ssb = 0;
    for (int j = 0; j < k; ++j) if (!used[j]) ++ssb;

    out_ssb = ssb;
    out_dsb = dsb;
}

// ---------------------------------------------------------------------------
// simulate_track: run ONE primary particle through the water box and return its
//   TrackResult. This is the single function that BOTH the CPU reference and the
//   GPU kernel call -- identical math => identical histories => exact verification.
//
//   Geometry: the box is [0,box_nm]^3. The primary enters at (x0, 0, z0) heading
//   along +y (the DNA axis), where (x0,z0) is a random impact point in the x-z
//   plane. DNA is a line of `n_dna_segments` segments along y at the box centre
//   (x = z = box/2). A break candidate is any ionization within dna_radius_nm of
//   that axis; its strand is chosen at random (a stand-in for the real helix
//   geometry). See THEORY.md for the full picture and what is simplified.
//
//   Returns: total energy quanta, SSB/DSB counts, and the lineal-energy bin.
// ---------------------------------------------------------------------------
TS_HD inline TrackResult simulate_track(const TrackParams& P, Rng& rng) {
    const double half = 0.5 * P.box_nm;          // box centre coordinate
    const double seg_len = P.box_nm / P.n_dna_segments;   // length of one DNA segment

    // PER-TRACK LET: sample this primary's local ionization density from a
    // lognormal around the mean sigma_ion. This one line turns a mono-energetic
    // beam into a MIXED FIELD: some tracks are dense (high-LET, DSB-rich, high
    // lineal energy), others sparse -- reproducing the long tail of a real
    // microdosimetric y-spectrum. let_spread = 0 disables it (identical tracks).
    const double sigma_track =
        (P.let_spread > 0.0) ? P.sigma_ion * exp(P.let_spread * rng_normal(rng))
                             : P.sigma_ion;

    // Random impact point of the primary in the entry face (x-z plane).
    double x = rng_uniform(rng) * P.box_nm;      // entry x
    double z = rng_uniform(rng) * P.box_nm;      // entry z
    double y = 0.0;                              // enters at y = 0, travels +y

    // Direction: mostly +y with a tiny transverse tilt (electrons scatter, ions
    // barely). We keep it a straight ray after this small initial tilt -- the
    // condensed-history approximation for a short traversal.
    double dx = P.dna_radius_nm > 0.0 ? rng_normal(rng) * 0.02 : 0.0;  // small slope
    double dz = rng_normal(rng) * 0.02;

    unsigned long long quanta = 0;   // integer energy tally for this track
    BreakEvent breaks[TS_MAX_BREAKS];
    int nbreak = 0;

    // March the primary in exponential free-flight steps between ionizations.
    for (int guard = 0; guard < 200000; ++guard) {
        const double xi = 1.0 - rng_uniform(rng);     // in (0,1], avoids log(0)
        const double s  = -log(xi) / sigma_track;     // exponential free path (nm)
        x += dx * s; y += s; z += dz * s;             // advance along the ray
        if (y >= P.box_nm) break;                     // primary exits the box

        // --- primary IONIZATION at (x,y,z): deposit integer quanta ---
        quanta += static_cast<unsigned long long>(P.quanta_per_ion);

        // Distance from the DNA axis (the line x=half, z=half, varying y).
        const double rx = x - half;
        const double rz = z - half;
        const double r_dna = sqrt(rx * rx + rz * rz);
        if (r_dna <= P.dna_radius_nm && nbreak < TS_MAX_BREAKS) {
            int seg = static_cast<int>(y / seg_len);
            if (seg >= P.n_dna_segments) seg = P.n_dna_segments - 1;
            breaks[nbreak].segment = seg;
            breaks[nbreak].strand  = (rng_uniform(rng) < 0.5) ? 0 : 1;
            breaks[nbreak].pos_nm  = y;
            ++nbreak;
        }

        // --- optional DELTA-RAY: a short secondary electron cluster nearby ---
        // This is where clustered damage comes from: a delta ray dropped near
        // the DNA can add the SECOND break that turns an SSB into a DSB.
        if (rng_uniform(rng) < P.p_delta) {
            quanta += static_cast<unsigned long long>(P.delta_quanta);
            // The delta lands a short random offset from the primary ionization.
            const double off = rng_normal(rng) * P.dna_radius_nm;   // ~nm-scale
            const double dr  = fabs(r_dna + off);
            if (dr <= P.dna_radius_nm && nbreak < TS_MAX_BREAKS) {
                int seg = static_cast<int>(y / seg_len);
                if (seg >= P.n_dna_segments) seg = P.n_dna_segments - 1;
                breaks[nbreak].segment = seg;
                breaks[nbreak].strand  = (rng_uniform(rng) < 0.5) ? 0 : 1;
                breaks[nbreak].pos_nm  = y + off;   // slightly displaced along axis
                ++nbreak;
            }
        }
    }

    // Turn the raw break list into SSB/DSB counts via the combinatorial rule.
    int ssb = 0, dsb = 0;
    const double dsb_window_nm = 3.4;   // ~10 base pairs
    classify_breaks(breaks, nbreak, dsb_window_nm, ssb, dsb);

    // MICRODOSIMETRY: lineal energy y = energy imparted / mean chord length.
    //   Energy imparted (keV) = quanta * quantum_eV / 1000.
    //   Mean chord of a convex body (Cauchy) = 4V/S; for a cube of side L that is
    //   (4 L^3)/(6 L^2) = 2L/3. With L in nm, convert chord to µm (/1000).
    //   => y in keV/µm.
    const double E_keV     = quanta * P.quantum_eV / 1000.0;
    const double chord_um  = (2.0 / 3.0) * P.box_nm / 1000.0;   // mean chord (µm)
    const double y_keV_um  = (chord_um > 0.0) ? E_keV / chord_um : 0.0;

    // Bin the lineal energy. Clamp into [0, n_y_bins) so the last bin catches the
    // tail; a track with zero energy lands in bin 0.
    int y_bin = static_cast<int>(y_keV_um / P.y_max_keV_um * P.n_y_bins);
    if (y_bin < 0) y_bin = 0;
    if (y_bin >= P.n_y_bins) y_bin = P.n_y_bins - 1;

    TrackResult out;
    out.energy_quanta = quanta;
    out.ssb = ssb;
    out.dsb = dsb;
    out.y_bin = y_bin;
    return out;
}
