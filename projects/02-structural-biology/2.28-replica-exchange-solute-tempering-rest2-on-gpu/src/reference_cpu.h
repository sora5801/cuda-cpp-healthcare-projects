// ===========================================================================
// src/reference_cpu.h  --  Config loading + CPU reference REST2 driver
// ---------------------------------------------------------------------------
// Project 2.28 : Replica Exchange Solute Tempering (REST2) on GPU
//
// This header declares:
//   * the input format + loader for a REST2 run (load_config),
//   * build_ladder(): turn (T0, Tmax, n_replicas) into the lambda ladder + seeds,
//   * ReplicaResult: the per-replica diagnostics we report and verify,
//   * the CPU reference driver (init_state + cpu_sample_round): the SERIAL twin
//     of the GPU kernel. main.cu does the EXCHANGES identically on both paths and
//     compares the per-replica results.
//
// It is PURE C++ (no CUDA constructs) because reference_cpu.cpp is built by the
// host compiler -- but it includes rest2.h, whose physics is shared with the GPU
// via the REST2_HD macro. kernels.cu also includes this header to reuse the
// config/ladder/result types (so there is exactly one definition of each).
//
// READ THIS AFTER: rest2.h.   READ BEFORE: reference_cpu.cpp, kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <cstdint>   // uint64_t (RNG counters)
#include <string>
#include <vector>

#include "rest2.h"   // SimConfig, ReplicaParams, N_SOLUTE, the shared physics

// ---------------------------------------------------------------------------
// Per-replica diagnostics we compute, report (deterministically), and VERIFY.
//   These are integer / exactly-reproducible quantities on purpose: the GPU and
//   CPU run identical math, so they MUST agree exactly (tolerance 0). We avoid
//   reporting floating averages as the headline check; coordinates and accept
//   counts come out of identical operation sequences and so match bit-for-bit.
// ---------------------------------------------------------------------------
struct ReplicaResult {
    double x[N_SOLUTE];      // final solute coordinates of this replica
    long   accepted_moves;   // total accepted MC moves over the whole run
    long   total_moves;      // total attempted MC moves (for the acceptance %)
    int    well_right;       // how many beads ended in the right well (x > 0)
};

// ---------------------------------------------------------------------------
// load_config: parse the tiny text input (see data/README.md).
//   Format (whitespace-separated, in this order):
//     n_replicas  sweeps_per_round  n_rounds
//     barrier_h   k_bond  k_pw  x_solvent  step_size
//     T0  Tmax
//   Throws std::runtime_error on a missing file or malformed/invalid numbers so
//   the demo fails loudly instead of silently running on garbage.
// ---------------------------------------------------------------------------
SimConfig load_config(const std::string& path);

// ---------------------------------------------------------------------------
// build_ladder: construct the per-replica (lambda, seed) list from the config.
//   Effective solute temperatures are spaced GEOMETRICALLY from T0 to Tmax (the
//   standard choice: it makes neighbouring exchange acceptance roughly uniform
//   across the ladder). lambda_m = T0 / T_m, so replica 0 is the cold, physical
//   replica (lambda = 1) and the last is the hottest (smallest lambda). Each
//   replica gets a distinct deterministic seed -> independent MC streams.
// ---------------------------------------------------------------------------
std::vector<ReplicaParams> build_ladder(const SimConfig& cfg);

// ---------------------------------------------------------------------------
// CPU reference driver. main.cu calls these so the CPU and GPU share the SAME
// exchange bookkeeping and only the per-replica SAMPLING differs (serial loop vs
// kernel). Splitting "sample" from "exchange" lets the exchange logic live once
// in main.cu and run identically on both paths -- the cleanest way to guarantee
// the two REST2 simulations are the same algorithm.
//
//   init_state: place every replica's beads at the SAME deterministic start
//               (left well, x = -1) so CPU and GPU begin identically.
//   cpu_sample_round: advance EVERY replica by cfg.sweeps_per_round MC sweeps,
//               serially. Updates coordinates and accumulates accept counts.
//               rng_counters[r] is threaded per replica so the next round
//               continues the same non-repeating deterministic stream.
// ---------------------------------------------------------------------------
void init_state(const SimConfig& cfg, std::vector<double>& coords /* n_replicas*N_SOLUTE */);

void cpu_sample_round(const SimConfig& cfg,
                      const std::vector<ReplicaParams>& reps,
                      std::vector<double>& coords,           // [n_replicas*N_SOLUTE], in/out
                      std::vector<long>& accepted,           // [n_replicas], in/out
                      std::vector<uint64_t>& rng_counters);  // [n_replicas], in/out
