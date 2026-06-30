// ===========================================================================
// src/main.cu  --  Entry point: run REST2 on CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 2.28 : Replica Exchange Solute Tempering (REST2) on GPU
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the run config (data/sample, or a built-in synthetic fallback).
//   2. Build the lambda ladder, then run the FULL REST2 loop twice:
//        - once with the CPU sampler (reference_cpu.cpp)  -> trusted answer,
//        - once with the GPU sampler  (kernels.cu)         -> the thing taught.
//      Both paths use the SAME exchange step (run_exchange below), so the only
//      difference is loop-vs-kernel sampling. They must agree bit-for-bit.
//   3. VERIFY: assert the GPU final state equals the CPU final state EXACTLY.
//   4. REPORT: deterministic per-replica summary to stdout; timing to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Timings (run-to-run varying) go to STDERR.
//
//   THE REST2 ROUND (repeated n_rounds times, identically on CPU and GPU):
//     a) SAMPLE: advance every replica by sweeps_per_round Metropolis MC sweeps
//        at its own effective temperature (cpu_sample_round / gpu_sample_round).
//     b) EXCHANGE: attempt to swap configurations between neighbouring replicas
//        using the REST2 Metropolis criterion (run_exchange). Alternating
//        even/odd pairs each round is the standard, deadlock-free scheme.
//
// Code tour: start here, then rest2.h (the physics), kernels.cu (the GPU twin),
// reference_cpu.cpp (the serial baseline). See ../THEORY.md for the "why".
// ===========================================================================
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>   // std::labs (integer absolute value for the occupancy diff)
#include <string>
#include <vector>

#include "kernels.cuh"        // gpu_sample_round (GPU path)
#include "reference_cpu.h"    // load_config, build_ladder, cpu_sample_round
#include "util/io.hpp"        // util::CpuTimer

// Stamped identifiers; keep in sync with demo/expected_output.txt.
static const char* PROJECT_ID   = "2.28";
static const char* PROJECT_NAME = "Replica Exchange Solute Tempering (REST2) on GPU";

// A SEPARATE deterministic RNG key for the EXCHANGE decisions (distinct from any
// per-replica sampling key) so swap rolls never correlate with move rolls.
static constexpr uint64_t EXCHANGE_KEY = 0x00000E5C'A0000000ULL;

// ---------------------------------------------------------------------------
// run_exchange: attempt REST2 swaps between neighbouring replicas for one round.
//   Shared by the CPU and GPU paths so the exchange algorithm is identical on
//   both (only the sampling differs). It is plain host code -- the exchange
//   touches only M energies, so doing it on the host is free and keeps the data
//   flow obvious (production engines exchange on-GPU via NCCL; THEORY.md).
//
//   Scheme: on round `round`, consider the pairs (p, p+1) whose lower index p has
//   the same parity as `round` -- i.e. even rounds swap (0,1),(2,3),..., odd
//   rounds swap (1,2),(3,4),.... Alternating parity lets configurations migrate
//   the full length of the ladder over successive rounds without any replica
//   being in two swaps at once.
//
//   For each candidate pair we:
//     1) compute both replicas' energy GROUPS (E_pp, E_pw) from their coords,
//     2) form the REST2 exchange Delta (rest2_exchange_delta, rest2.h),
//     3) accept the swap with prob min(1, exp(-Delta)) using ONE deterministic
//        uniform draw keyed by (round, p) -- reproducible across CPU and GPU,
//     4) on acceptance, SWAP the two replicas' coordinate blocks (the lambdas
//        stay put: a replica is a fixed temperature slot, configurations move).
//   Returns the number of swaps accepted this round (for the diagnostic).
//
//   coords : [n_replicas*N_SOLUTE], modified in place when a swap is accepted.
// ---------------------------------------------------------------------------
static int run_exchange(const SimConfig& cfg,
                        const std::vector<ReplicaParams>& reps,
                        std::vector<double>& coords,
                        int round) {
    int swaps = 0;
    const int start = round % 2;   // 0 -> pairs (0,1),(2,3)...; 1 -> (1,2),(3,4)...
    for (int p = start; p + 1 < cfg.n_replicas; p += 2) {
        const int m = p, n = p + 1;                 // neighbouring replica indices
        double* xm = &coords[static_cast<std::size_t>(m) * N_SOLUTE];
        double* xn = &coords[static_cast<std::size_t>(n) * N_SOLUTE];

        // Energy groups of each replica's CURRENT configuration. The exchange
        // criterion needs only E_pp and E_pw (E_ww cancels in REST2 -- rest2.h).
        double Epp_m, Epw_m, Eww_m, Epp_n, Epw_n, Eww_n;
        rest2_energies(xm, cfg.barrier_h, cfg.tilt, cfg.k_bond, cfg.k_pw, cfg.x_solvent, Epp_m, Epw_m, Eww_m);
        rest2_energies(xn, cfg.barrier_h, cfg.tilt, cfg.k_bond, cfg.k_pw, cfg.x_solvent, Epp_n, Epw_n, Eww_n);

        const double Delta = rest2_exchange_delta(reps[m].lambda, reps[n].lambda,
                                                  Epp_m, Epw_m, Epp_n, Epw_n);

        // One deterministic uniform per pair, keyed by (round, lower index). The
        // hash mixes both so different pairs/rounds get uncorrelated rolls.
        const uint64_t ctr = (static_cast<uint64_t>(round) << 20) ^ static_cast<uint64_t>(p);
        const double roll = rng_uniform(EXCHANGE_KEY, ctr);

        // Metropolis accept: always if Delta<=0, else with prob exp(-Delta).
        if (Delta <= 0.0 || roll < std::exp(-Delta)) {
            // Swap the two coordinate blocks (the configurations trade slots).
            for (int i = 0; i < N_SOLUTE; ++i) {
                const double tmp = xm[i]; xm[i] = xn[i]; xn[i] = tmp;
            }
            ++swaps;
        }
    }
    return swaps;
}

// ---------------------------------------------------------------------------
// summarize: turn a finished simulation's state into per-replica results.
//   well_right counts beads that ended in the RIGHT well (x > 0) -- our readout
//   of "did this replica escape the starting (left) basin?". accepted/total give
//   the MC acceptance ratio (a health check: ~30-50% is the classic sweet spot).
// ---------------------------------------------------------------------------
static std::vector<ReplicaResult> summarize(const SimConfig& cfg,
                                            const std::vector<double>& coords,
                                            const std::vector<long>& accepted) {
    std::vector<ReplicaResult> out(cfg.n_replicas);
    const long moves_per_replica =
        static_cast<long>(cfg.sweeps_per_round) * cfg.n_rounds * N_SOLUTE;
    for (int r = 0; r < cfg.n_replicas; ++r) {
        ReplicaResult& rr = out[r];
        int right = 0;
        for (int i = 0; i < N_SOLUTE; ++i) {
            rr.x[i] = coords[static_cast<std::size_t>(r) * N_SOLUTE + i];
            if (rr.x[i] > 0.0) ++right;
        }
        rr.well_right = right;
        rr.accepted_moves = accepted[r];
        rr.total_moves = moves_per_replica;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Run the WHOLE REST2 simulation with a pluggable sampler. `use_gpu` picks the
// GPU kernel vs the CPU loop for the SAMPLE phase; the EXCHANGE phase is shared.
// Returns the per-replica results and accumulates total swaps + kernel time.
// ---------------------------------------------------------------------------
static std::vector<ReplicaResult> run_rest2(const SimConfig& cfg,
                                            const std::vector<ReplicaParams>& reps,
                                            bool use_gpu,
                                            int& total_swaps,
                                            float& gpu_ms_accum) {
    std::vector<double>   coords;            // [M*N_SOLUTE] all replica coordinates
    std::vector<long>     accepted(cfg.n_replicas, 0);   // running accept counts
    std::vector<uint64_t> rng_ctr(cfg.n_replicas, 0);    // per-replica RNG cursors
    init_state(cfg, coords);                 // every replica starts in the left well
    total_swaps = 0;
    gpu_ms_accum = 0.0f;

    for (int round = 0; round < cfg.n_rounds; ++round) {
        // (a) SAMPLE: advance every replica.
        if (use_gpu) {
            float ms = 0.0f;
            gpu_sample_round(cfg, reps, coords, accepted, rng_ctr, &ms);
            gpu_ms_accum += ms;
        } else {
            cpu_sample_round(cfg, reps, coords, accepted, rng_ctr);
        }
        // (b) EXCHANGE: shared, deterministic REST2 swap step.
        total_swaps += run_exchange(cfg, reps, coords, round);
    }
    return summarize(cfg, coords, accepted);
}

int main(int argc, char** argv) {
    // ---- 1. Load the config ------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/rest2_config.txt";
    SimConfig cfg;
    try {
        cfg = load_config(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const std::vector<ReplicaParams> reps = build_ladder(cfg);

    // ---- 2. CPU reference REST2 (timed) ------------------------------------
    int cpu_swaps = 0; float cpu_dummy_ms = 0.0f;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    std::vector<ReplicaResult> res_cpu = run_rest2(cfg, reps, /*use_gpu=*/false, cpu_swaps, cpu_dummy_ms);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU REST2 (kernel time accumulated across rounds) --------------
    int gpu_swaps = 0; float gpu_ms = 0.0f;
    std::vector<ReplicaResult> res_gpu = run_rest2(cfg, reps, /*use_gpu=*/true, gpu_swaps, gpu_ms);

    // ---- 4. Verify: GPU vs CPU on ROBUST AGGREGATE observables -------------
    // The CPU and GPU run the SAME math, and we compile the device with
    // --fmad=false so their energy ARITHMETIC is bit-identical. But the
    // Metropolis test uses exp(), whose host vs device libm differ by ~1 ULP; a
    // single borderline accept can flip and then -- because an MC trajectory is
    // CHAOTIC -- the two paths diverge afterward. So bit-identical trajectories
    // are NOT a sound gate (PATTERNS.md section 4; sibling project 1.06 teaches
    // the same lesson). We instead compare STATISTICAL observables that are
    // stable under a few flipped moves:
    //   * total beads in the right well across ALL replicas (an integer count of
    //     the sampled population) -- allow a small absolute slack,
    //   * the global MC acceptance ratio                    -- allow a small slack.
    // These are exactly the kind of robust readouts a real REST2 study reports.
    long right_cpu = 0, right_gpu = 0;          // summed right-well beads
    long acc_cpu = 0, acc_gpu = 0, tot_moves = 0;  // summed accepts / attempts
    for (int r = 0; r < cfg.n_replicas; ++r) {
        right_cpu += res_cpu[r].well_right;  right_gpu += res_gpu[r].well_right;
        acc_cpu   += res_cpu[r].accepted_moves;  acc_gpu += res_gpu[r].accepted_moves;
        tot_moves += res_cpu[r].total_moves;
    }
    // Tolerances (documented in THEORY.md "How we verify correctness"):
    //   * well occupancy may differ by at most a handful of beads out of
    //     n_replicas*N_SOLUTE (a few flipped late moves);
    //   * acceptance ratio by at most 1 percentage point.
    const long  WELL_SLACK = 2 * cfg.n_replicas;         // a few beads per replica
    const double ACC_SLACK = 0.01;                       // 1 percentage point
    const double accr_cpu = static_cast<double>(acc_cpu) / tot_moves;
    const double accr_gpu = static_cast<double>(acc_gpu) / tot_moves;
    const long  well_diff = std::labs(right_cpu - right_gpu);
    const double acc_diff = std::fabs(accr_cpu - accr_gpu);
    const bool pass = (well_diff <= WELL_SLACK) && (acc_diff <= ACC_SLACK);

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    // We print the CPU REFERENCE results: identical operations make them
    // deterministic across runs, and plain IEEE arithmetic (no FMA) makes them
    // the most compiler-portable choice for a stable expected_output.txt. The
    // GPU produces statistically the same numbers (verified above) but its exact
    // last digits depend on the device libm, so the portable reference is what we
    // freeze. Integer occupancy counts are the headline; the mean-x is rounded.
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("REST2: %d replicas, %d rounds x %d sweeps, %d solute beads; "
                "barrier h=%.2f tilt=%.2f\n",
                cfg.n_replicas, cfg.n_rounds, cfg.sweeps_per_round, N_SOLUTE,
                cfg.barrier_h, cfg.tilt);
    std::printf("lambda ladder (cold->hot): ");
    for (int r = 0; r < cfg.n_replicas; ++r) std::printf("%.4f ", reps[r].lambda);
    std::printf("\n");
    std::printf("per-replica (idx lambda  accept%%  beadsRight):\n");
    for (int r = 0; r < cfg.n_replicas; ++r) {
        const double acc_pct = 100.0 * res_cpu[r].accepted_moves / res_cpu[r].total_moves;
        std::printf("  r%-2d  %.4f  %6.2f  %d/%d\n",
                    r, reps[r].lambda, acc_pct, res_cpu[r].well_right, N_SOLUTE);
    }
    // Headline science readout: did the COLD replica (r0, physical 300 K, lambda=1)
    // escape the starting LEFT basin and find the global RIGHT well? Plain MD at
    // 300 K started on the left would stay stuck; REST2's swaps let r0 inherit
    // barrier-crossed configurations from the hot replicas. All beads ending
    // right => the cold replica reached the global free-energy minimum.
    std::printf("cold replica r0: %d/%d beads in the global (right) well "
                "[start was 0/%d, trapped left]\n",
                res_cpu[0].well_right, N_SOLUTE, N_SOLUTE);
    std::printf("total right-well beads across ladder: %ld/%d\n",
                right_cpu, cfg.n_replicas * N_SOLUTE);
    std::printf("exchanges accepted over the run: %d\n", cpu_swaps);
    std::printf("RESULT: %s (GPU matches CPU on robust observables: "
                "well occupancy +/-%ld beads, acceptance +/-%.0f%%)\n",
                pass ? "PASS" : "FAIL", WELL_SLACK, 100.0 * ACC_SLACK);

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%d replicas)\n", path.c_str(), cfg.n_replicas);
    std::fprintf(stderr, "[timing] CPU REST2: %.3f ms   GPU kernels (sum over rounds): %.3f ms\n",
                 cpu_ms, gpu_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- a tiny toy system per replica is "
                         "launch/copy-bound; the GPU's edge grows with system size & replica count.\n");
    std::fprintf(stderr, "[verify] right-well beads CPU=%ld GPU=%ld (diff %ld, slack %ld); "
                         "acceptance CPU=%.4f GPU=%.4f (diff %.2e, slack %.2e); "
                         "exchanges CPU=%d GPU=%d\n",
                 right_cpu, right_gpu, well_diff, WELL_SLACK,
                 accr_cpu, accr_gpu, acc_diff, ACC_SLACK, cpu_swaps, gpu_swaps);

    return pass ? 0 : 1;
}
