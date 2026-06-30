// ===========================================================================
// src/reference_cpu.cpp  --  Loader, ladder builder, serial REST2 sampler
// ---------------------------------------------------------------------------
// Project 2.28 : Replica Exchange Solute Tempering (REST2) on GPU
//
// Compiled by the HOST compiler only (cl.exe / g++). It contains no CUDA, but it
// calls the SAME REST2 physics (rest2.h) that the GPU kernel calls, so the CPU
// trajectory is the trusted, bit-identical baseline for the GPU one. The actual
// MC sweep + energy math live in rest2.h; this file is just the serial driver
// and the I/O around it. The exchange step lives in main.cu (shared by both
// paths). See ../THEORY.md "How we verify correctness".
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>       // std::pow
#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error

// A fixed 64-bit base mixed into each replica's seed. Any constant works; this
// one ("REST2" leetspelled into hex) keeps the seeds reproducible run to run.
static constexpr uint64_t SEED_BASE = 0x0000'2E57'2000'0000ULL;

// ---------------------------------------------------------------------------
// load_config: read the 10 numbers that define a REST2 run (see data/README.md).
//   We validate aggressively: a bad ladder (n_replicas < 2) or a non-positive
//   temperature would make the geometric spacing or the lambda ratio undefined.
// ---------------------------------------------------------------------------
SimConfig load_config(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open config file: " + path);

    SimConfig c;
    if (!(in >> c.n_replicas >> c.sweeps_per_round >> c.n_rounds
             >> c.barrier_h >> c.tilt >> c.k_bond >> c.k_pw >> c.x_solvent >> c.step_size
             >> c.T0 >> c.Tmax))
        throw std::runtime_error(
            "bad config (expected 'n_replicas sweeps_per_round n_rounds  "
            "barrier_h tilt k_bond k_pw x_solvent step_size  T0 Tmax') in " + path);

    // Physical sanity: at least two replicas (you need a ladder to exchange on),
    // a non-degenerate, hot-end-warmer-than-cold-end temperature range, and a
    // positive move size. These guard against silently-meaningless runs.
    if (c.n_replicas < 2)
        throw std::runtime_error("need n_replicas >= 2 (a ladder) in " + path);
    if (c.sweeps_per_round <= 0 || c.n_rounds <= 0)
        throw std::runtime_error("sweeps_per_round and n_rounds must be > 0 in " + path);
    if (c.T0 <= 0.0 || c.Tmax < c.T0)
        throw std::runtime_error("need 0 < T0 <= Tmax in " + path);
    if (c.step_size <= 0.0)
        throw std::runtime_error("step_size must be > 0 in " + path);
    return c;
}

// ---------------------------------------------------------------------------
// build_ladder: geometric temperature spacing -> per-replica (lambda, seed).
//   T_m = T0 * (Tmax/T0)^(m/(M-1))  for m = 0..M-1.
//   lambda_m = T0 / T_m  in (0,1]; replica 0 has T_m = T0 -> lambda = 1 (cold).
//   Geometric (constant ratio) spacing is the textbook choice because exchange
//   acceptance depends on temperature RATIOS, so equal ratios -> roughly equal
//   acceptance between every neighbouring pair (THEORY.md "GPU mapping").
//   Seeds are derived deterministically from the replica index via the shared
//   hash so every replica gets an INDEPENDENT, reproducible RNG stream.
// ---------------------------------------------------------------------------
std::vector<ReplicaParams> build_ladder(const SimConfig& cfg) {
    std::vector<ReplicaParams> reps(cfg.n_replicas);
    const int M = cfg.n_replicas;
    for (int m = 0; m < M; ++m) {
        // Fraction along the ladder in [0,1]; m=0 -> 0 (cold), m=M-1 -> 1 (hot).
        const double frac = (M > 1) ? static_cast<double>(m) / (M - 1) : 0.0;
        const double Tm = cfg.T0 * std::pow(cfg.Tmax / cfg.T0, frac);  // geometric
        reps[m].lambda = cfg.T0 / Tm;                                  // in (0,1]
        // Distinct, deterministic per-replica seed (hash of a fixed base XOR m).
        reps[m].seed = rng_hash64(SEED_BASE ^ (static_cast<uint64_t>(m) * 0x100000001B3ULL));
    }
    return reps;
}

// ---------------------------------------------------------------------------
// init_state: every replica starts with all beads in the LEFT well (x = -1).
//   A deterministic, identical start on CPU and GPU is essential: enhanced
//   sampling is judged by whether the cold replica ESCAPES this well to find the
//   right one, so we must all begin trapped in the same place.
// ---------------------------------------------------------------------------
void init_state(const SimConfig& cfg, std::vector<double>& coords) {
    coords.assign(static_cast<std::size_t>(cfg.n_replicas) * N_SOLUTE, -1.0);
}

// ---------------------------------------------------------------------------
// cpu_sample_round: serially advance EVERY replica by cfg.sweeps_per_round MC
//   sweeps. This is the exact serial counterpart of the GPU kernel
//   (sample_round_kernel in kernels.cu): one replica == one independent job.
//   Per replica we thread its rng_counter so successive rounds keep drawing
//   fresh, non-repeating randoms; we accumulate accepted moves for diagnostics.
// ---------------------------------------------------------------------------
void cpu_sample_round(const SimConfig& cfg,
                      const std::vector<ReplicaParams>& reps,
                      std::vector<double>& coords,
                      std::vector<long>& accepted,
                      std::vector<uint64_t>& rng_counters) {
    for (int r = 0; r < cfg.n_replicas; ++r) {
        double* x = &coords[static_cast<std::size_t>(r) * N_SOLUTE];  // this replica's beads
        uint64_t ctr = rng_counters[r];                              // local copy of the stream cursor
        long acc = 0;
        for (int s = 0; s < cfg.sweeps_per_round; ++s)
            acc += mc_sweep(x, reps[r], cfg, ctr);    // one Metropolis sweep (rest2.h)
        accepted[r]    += acc;     // running accept count for the acceptance ratio
        rng_counters[r] = ctr;     // persist the advanced stream cursor
    }
}
