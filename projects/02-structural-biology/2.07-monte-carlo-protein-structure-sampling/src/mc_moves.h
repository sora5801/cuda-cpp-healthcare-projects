// ===========================================================================
// src/mc_moves.h  --  Shared (host + device) RNG + lattice-protein MC engine
// ---------------------------------------------------------------------------
// Project 2.7 : Monte Carlo Protein Structure Sampling (reduced-scope, the
//               2-D HP lattice-protein model of Lau & Dill, 1989).
//
// WHY THIS HEADER IS SHARED (the single most important idea in this project)
//   Monte Carlo verification only works if the CPU reference and the GPU kernel
//   simulate the *identical* random walk, so their results must match EXACTLY.
//   That only happens if both use the same RNG and the same accept/reject logic.
//   So both live here, in ONE header, included by reference_cpu.cpp (the plain
//   host C++ compiler) AND by kernels.cu / main.cu (nvcc). The MC_HD macro is
//   `__host__ __device__` under nvcc and empty under the host compiler, so the
//   same inline functions compile and run in both worlds. (PATTERNS.md §2.)
//
//   Production GPU MC uses cuRAND for device randomness; we deliberately use a
//   shared, counter-based splitmix64 stream instead, *specifically* so the CPU
//   and GPU draw bit-identical random numbers and the demo is reproducible. See
//   THEORY.md "How we verify correctness".
//
// WHY THE RESULTS ARE BIT-IDENTICAL (not just "close") -- read this carefully
//   The Metropolis accept test is a HARD BRANCH: a single accepted-vs-rejected
//   flip sends two trajectories to completely different conformations. Floating
//   point that diverges by even one ULP between host and device would therefore
//   ruin an exact comparison. We avoid that two ways:
//     1. The HP model's energy is an INTEGER (minus the count of H-H contacts),
//        so the energy change dE of any move is a small integer.
//     2. The accept probability exp(beta*dE) is therefore one of only a few
//        discrete values. We precompute those in a small table of doubles ONCE
//        (host side, in build_boltzmann_table) and BOTH the CPU and the GPU look
//        up the SAME table entry. No transcendental is evaluated inside the walk
//        on either side, so the accept comparison `u < table[dE]` uses identical
//        bits everywhere -> identical accept decisions -> identical trajectories.
//   The result: CPU and GPU final energies match with tolerance EXACTLY 0
//   (PATTERNS.md §4, the "exact" row). That is the lesson this project teaches.
//
// THE SIMPLIFIED SCIENCE (a deliberately reduced teaching model; THEORY.md has
// the full picture and how Rosetta/real folding differs)
//   A protein is modeled as a self-avoiding chain of N residues placed on the
//   integer lattice Z^2. Each residue is H (hydrophobic) or P (polar). The free
//   energy is  E = -eps * (number of non-bonded H-H contacts), eps>0: the chain
//   wants to bury its H residues next to each other (the "hydrophobic collapse"
//   that drives real folding). MC samples conformations: propose a local move,
//   accept with the Metropolis rule, repeat. Many independent replicas at a
//   ladder of temperatures explore the landscape in parallel (the GPU's job).
// ===========================================================================
#pragma once

#include <cstdint>
#include <cmath>

// MC_HD: expands to the CUDA decorators under nvcc, to nothing under the host
// compiler -- so every function below is callable from BOTH a CPU loop and a
// GPU thread. (Keep CUDA-only constructs like __global__ OUT of this header so
// the plain host compiler can include it; PATTERNS.md §2.)
#ifdef __CUDACC__
#define MC_HD __host__ __device__
#else
#define MC_HD
#endif

// Hard caps so every buffer is a fixed-size stack array (no device malloc, and
// the same sizes on host and device). A teaching model stays small on purpose.
static const int MC_MAX_N    = 64;   // max residues in the chain
static const int MC_DE_RANGE = 16;   // Boltzmann table covers dE in [-R, +R]

// ---------------------------------------------------------------------------
// RNG: a splitmix64 counter-based stream, identical on host and device.
//   We need a generator that (a) gives each replica an independent, reproducible
//   stream from just its index, and (b) computes the SAME bits on CPU and GPU.
//   splitmix64 is a tiny, well-known mixer that satisfies both; it needs no
//   per-thread state table (unlike a Mersenne twister) -- perfect for one
//   lightweight stream per GPU thread.
// ---------------------------------------------------------------------------
struct Rng { uint64_t state; };   // the whole generator is one 64-bit word

// One splitmix64 step: advance `x` and return a well-mixed 64-bit value.
MC_HD inline uint64_t splitmix64(uint64_t& x) {
    x += 0x9E3779B97F4A7C15ULL;                 // add the golden-ratio constant
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; // avalanche the high/low bits
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Seed an independent stream for replica `replica` from a global base seed, so
// replicas are uncorrelated yet each is reproducible from (base, replica).
MC_HD inline Rng rng_seed(uint64_t base, uint64_t replica) {
    Rng r;
    r.state = base ^ (replica * 0x9E3779B97F4A7C15ULL + 0xD1B54A32D192ED03ULL);
    splitmix64(r.state);   // warm up so nearby seeds don't start correlated
    return r;
}

// Uniform double in [0,1) from 53 random bits (identical math host/device).
MC_HD inline double rng_uniform(Rng& r) {
    uint64_t z = splitmix64(r.state);
    return (z >> 11) * (1.0 / 9007199254740992.0);   // multiply by 2^-53
}

// Uniform integer in [0, n) via the classic multiply-high trick (no modulo
// bias for our small n; identical on host and device).
MC_HD inline int rng_below(Rng& r, int n) {
    uint64_t z = splitmix64(r.state);
    // (z * n) >> 32 maps the top 32 bits of z uniformly onto [0,n).
    return (int)(((z >> 32) * (uint64_t)n) >> 32);
}

// ---------------------------------------------------------------------------
// The MC problem: a fixed HP sequence plus the run parameters. The conformation
// itself (the lattice coordinates) is per-replica scratch, NOT stored here.
// ---------------------------------------------------------------------------
struct McProblem {
    int     n;                 // chain length (residues), 2 <= n <= MC_MAX_N
    int     hp[MC_MAX_N];      // residue types: 1 = H (hydrophobic), 0 = P (polar)
    int     sweeps;            // MC sweeps per replica (1 sweep = n move attempts)
    int     n_replicas;        // independent walkers (= GPU threads)
    double  t_min;             // lowest replica temperature (in eps units)
    double  t_max;             // highest replica temperature (replica ladder)
    uint64_t seed;             // base RNG seed (replica r uses stream (seed,r))
};

// The result of one replica's walk: the best (lowest) energy it ever reached
// and the energy of its final conformation. Energies are INTEGERS (= -contacts)
// so they compare exactly between CPU and GPU.
struct McResult {
    int best_energy;    // most negative energy seen during the walk (>= ... <= 0)
    int final_energy;   // energy of the last accepted conformation
};

// Temperature assigned to replica r: a geometric ladder from t_min to t_max.
// Hot replicas (large T) cross barriers; cold replicas (small T) refine minima.
// Geometric spacing gives roughly equal acceptance gaps between neighbours --
// the standard choice for parallel tempering (THEORY.md "real world").
MC_HD inline double replica_temperature(const McProblem& P, int r) {
    if (P.n_replicas <= 1) return P.t_min;
    double f = (double)r / (double)(P.n_replicas - 1);   // 0..1 across replicas
    // Geometric interpolation: T(r) = t_min * (t_max/t_min)^f.
    return P.t_min * pow(P.t_max / P.t_min, f);
}

// ---------------------------------------------------------------------------
// build_boltzmann_table: precompute min(1, exp(-dE / T)) for every integer dE
// in [-MC_DE_RANGE, +MC_DE_RANGE], for a given temperature T. Called ONCE PER
// REPLICA on the host (in main.cu) before the walk; the resulting table is read
// (never recomputed) inside the walk by both CPU and GPU.
//   * tbl must point to (2*MC_DE_RANGE + 1) doubles; tbl[dE + MC_DE_RANGE] holds
//     the acceptance probability for an energy change of exactly dE.
//   * Computing exp() here (and NOWHERE inside the hot loop) is the trick that
//     makes the accept comparison bit-identical on CPU and GPU (see header top).
// This function is host-only on purpose (it builds a host array); it is plain
// C++ so reference_cpu.cpp and main.cu can both call it.
// ---------------------------------------------------------------------------
inline void build_boltzmann_table(double T, double* tbl) {
    for (int dE = -MC_DE_RANGE; dE <= MC_DE_RANGE; ++dE) {
        double p = exp(-(double)dE / T);     // Metropolis weight for this dE
        if (p > 1.0) p = 1.0;                // downhill / flat moves: always accept
        tbl[dE + MC_DE_RANGE] = p;
    }
}

// ---------------------------------------------------------------------------
// count_contacts: the energy function. Given the lattice coordinates of all n
// residues, return the number of NON-BONDED H-H contacts -- pairs of H residues
// that are lattice neighbours (distance 1) but are NOT adjacent along the chain.
// Energy is E = -count (we maximize contacts = minimize energy). O(n^2); n is
// tiny so this is fine, and it keeps the teaching code obvious. (A real engine
// updates energy incrementally per move; we recompute for clarity -- THEORY.)
//   x, y : integer lattice coordinates, length n.
// ---------------------------------------------------------------------------
MC_HD inline int count_contacts(const McProblem& P, const int* x, const int* y) {
    int c = 0;
    for (int i = 0; i < P.n; ++i) {
        if (!P.hp[i]) continue;                 // only H residues form contacts
        for (int j = i + 2; j < P.n; ++j) {     // j>=i+2 => skip chain neighbours
            if (!P.hp[j]) continue;
            int dx = x[i] - x[j], dy = y[i] - y[j];
            if (dx * dx + dy * dy == 1) ++c;    // lattice-adjacent H-H pair
        }
    }
    return c;
}

// is_occupied: is any residue OTHER THAN `except` sitting on lattice cell (px,py)?
// Used to enforce SELF-AVOIDANCE: two residues may never share a lattice site.
MC_HD inline bool is_occupied(const McProblem& P, const int* x, const int* y,
                              int px, int py, int except) {
    for (int k = 0; k < P.n; ++k) {
        if (k == except) continue;
        if (x[k] == px && y[k] == py) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// run_replica: simulate ONE Metropolis Monte Carlo walker end to end. This is
// the function the CPU loops over (once per replica) and the GPU runs (one
// thread per replica) -- identical code, identical RNG, identical Boltzmann
// table => identical result. Returns best & final energy (both integers).
//
//   P    : the problem (sequence, sweeps, ...).
//   r    : this replica's index (selects its RNG stream and temperature).
//   tbl  : this replica's Boltzmann table (2*MC_DE_RANGE+1 doubles), prebuilt
//          on the host for temperature replica_temperature(P, r).
//
// THE MOVE SET (single-residue "pull"/end + corner-flip, the textbook local
// moves for lattice proteins): we attempt, for a random interior residue i, to
// move it to a random lattice neighbour of residue i-1 (a "crankshaft"-like
// local rewiring), provided the new position keeps the chain connected and
// self-avoiding. End residues (i=0 or n-1) get a free end move. This small,
// ergodic move set is enough to fold the toy sequences in data/sample.
//
// THE METROPOLIS RULE: compute the energy change dE = E_new - E_old. If dE <= 0
// the move lowers (or keeps) energy -> accept. If dE > 0 accept with probability
// exp(-dE/T): draw u ~ U[0,1) and accept iff u < tbl[dE]. The table already
// holds min(1, exp(-dE/T)), so a single comparison decides both cases.
// ---------------------------------------------------------------------------
MC_HD inline McResult run_replica(const McProblem& P, int r, const double* tbl) {
    // --- Per-replica conformation: start as a straight horizontal rod ---------
    // A straight chain is always valid (self-avoiding) and gives a deterministic
    // starting point, so replica r's entire trajectory depends only on (seed,r).
    int x[MC_MAX_N];
    int y[MC_MAX_N];
    for (int i = 0; i < P.n; ++i) { x[i] = i; y[i] = 0; }

    Rng rng = rng_seed(P.seed, (uint64_t)r);     // this replica's RNG stream
    int energy = -count_contacts(P, x, y);       // current energy (integer)
    int best   = energy;                         // best (lowest) seen so far

    // The four unit lattice directions, used to propose neighbour cells.
    const int dirx[4] = { 1, -1, 0, 0 };
    const int diry[4] = { 0, 0, 1, -1 };

    // One "sweep" attempts n moves, so total attempts = sweeps * n. This is the
    // conventional MC time unit (give every residue ~one chance to move).
    const long long attempts = (long long)P.sweeps * (long long)P.n;
    for (long long step = 0; step < attempts; ++step) {
        int i = rng_below(rng, P.n);             // pick a residue to move

        // Propose a new position for residue i.
        int nx, ny;
        if (i == 0) {
            // End move: place residue 0 at a random neighbour of residue 1.
            int d = rng_below(rng, 4);
            nx = x[1] + dirx[d]; ny = y[1] + diry[d];
        } else if (i == P.n - 1) {
            // End move at the other terminus: neighbour of residue n-2.
            int d = rng_below(rng, 4);
            nx = x[P.n - 2] + dirx[d]; ny = y[P.n - 2] + diry[d];
        } else {
            // Interior corner/crankshaft move: a candidate cell that is a lattice
            // neighbour of BOTH i-1 and i+1 keeps the chain connected. We pick a
            // random neighbour of i-1 and let the connectivity check below reject
            // it if it is not also adjacent to i+1.
            int d = rng_below(rng, 4);
            nx = x[i - 1] + dirx[d]; ny = y[i - 1] + diry[d];
        }

        // --- Validity: keep the chain CONNECTED and SELF-AVOIDING -------------
        // Connectivity: the new site must be lattice-adjacent to the residue(s)
        // that remain bonded to i.
        bool connected = true;
        if (i > 0) {
            int ddx = nx - x[i - 1], ddy = ny - y[i - 1];
            if (ddx * ddx + ddy * ddy != 1) connected = false;   // broke i-1 bond
        }
        if (i < P.n - 1) {
            int ddx = nx - x[i + 1], ddy = ny - y[i + 1];
            if (ddx * ddx + ddy * ddy != 1) connected = false;   // broke i+1 bond
        }
        // Self-avoidance: the target cell must be empty (ignoring i itself).
        if (connected && !is_occupied(P, x, y, nx, ny, i)) {
            // Tentatively apply the move, score it, then accept or roll back.
            int ox = x[i], oy = y[i];
            x[i] = nx; y[i] = ny;
            int new_energy = -count_contacts(P, x, y);
            int dE = new_energy - energy;        // small integer in practice

            // Clamp dE into the table range purely for safety; for these tiny
            // chains |dE| never approaches MC_DE_RANGE, but a guard keeps the
            // index in bounds on host and device alike.
            int idx = dE; if (idx >  MC_DE_RANGE) idx =  MC_DE_RANGE;
                          if (idx < -MC_DE_RANGE) idx = -MC_DE_RANGE;

            double u = rng_uniform(rng);         // one draw decides the move
            if (u < tbl[idx + MC_DE_RANGE]) {
                energy = new_energy;             // ACCEPT: keep the new position
                if (energy < best) best = energy;
            } else {
                x[i] = ox; y[i] = oy;            // REJECT: restore old position
            }
        } else {
            // Even a rejected-by-geometry attempt consumes one uniform draw, so
            // that CPU and GPU stay in RNG lockstep regardless of the branch.
            (void)rng_uniform(rng);
        }
    }

    McResult res;
    res.best_energy  = best;
    res.final_energy = energy;
    return res;
}
