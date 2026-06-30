// ===========================================================================
// src/main.cu  --  Entry point: load problem, run CPU + GPU MC, verify, report
// ---------------------------------------------------------------------------
// Project 2.7 : Monte Carlo Protein Structure Sampling (HP lattice model)
//
// WHAT THIS FILE DOES  (the 5-step shape every project in this repo follows)
//   1. Load the HP problem (data/sample, or a built-in synthetic fallback).
//   2. Precompute the per-replica Boltzmann acceptance tables on the host (the
//      ONLY place exp() is evaluated -- see mc_moves.h header for why).
//   3. CPU reference: run every replica serially          -> trusted answer.
//   4. GPU: run every replica in parallel, one thread each -> the thing taught.
//   5. VERIFY (exact integer match) and REPORT a deterministic summary.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Run-to-run varying numbers (timings) go to STDERR,
//   which the demo shows but does not diff.
//
// Code tour: start here, then mc_moves.h (RNG + the walk), kernels.cu (the GPU
// twin), reference_cpu.cpp (the serial baseline). The science / GPU mapping is
// in ../THEORY.md.
// ===========================================================================
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "kernels.cuh"        // sample_gpu (GPU path), McProblem, McResult
#include "reference_cpu.h"    // load_mc_problem, sample_cpu, boltzmann_table_size
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "2.7";
static const char* PROJECT_NAME = "Monte Carlo Protein Structure Sampling";

// ---------------------------------------------------------------------------
// make_synthetic: the built-in problem used when no data file is supplied. It
// mirrors data/sample/hp_problem.txt exactly so the program is runnable even
// without the sample on disk. The sequence below is the classic HPHPPHHPHH...
// benchmark-style chain (clearly SYNTHETIC; data/README.md documents it). The
// known ground truth for this sequence is a folded state with several H-H
// contacts -- the demo reports how many the ensemble recovers.
// ---------------------------------------------------------------------------
static McProblem make_synthetic() {
    McProblem P{};
    const char* seq = "HPHPPHHPHHPHHPPHPH";   // 18-residue synthetic HP chain
    P.n          = (int)std::strlen(seq);
    for (int i = 0; i < P.n; ++i) P.hp[i] = (seq[i] == 'H') ? 1 : 0;
    P.sweeps     = 600;     // MC sweeps per replica (1 sweep = n attempts)
    P.n_replicas = 256;     // independent walkers -> 256 GPU threads
    P.t_min      = 0.30;    // coldest replica (refines minima)
    P.t_max      = 3.00;    // hottest replica (crosses energy barriers)
    P.seed       = 20260628ULL;   // fixed seed -> deterministic demo
    return P;
}

int main(int argc, char** argv) {
    // ---- 1. Load the problem (file arg, else built-in synthetic) -----------
    McProblem prob;
    const char* source = "synthetic (built-in)";
    if (argc > 1) {
        try {
            prob = load_mc_problem(argv[1]);
            source = argv[1];
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[error] %s\n", e.what());
            return 2;
        }
    } else {
        prob = make_synthetic();
    }

    // ---- 2. Precompute the Boltzmann acceptance tables (host, once) --------
    // One table per replica (it depends on that replica's temperature). This is
    // the ONLY place exp() runs; the walk just indexes the table. Computing it
    // here -- identically for the CPU and GPU paths -- is what makes the accept
    // decisions, and therefore the whole trajectory, bit-identical (mc_moves.h).
    const int stride = boltzmann_table_size();
    std::vector<double> tables((std::size_t)prob.n_replicas * stride);
    for (int r = 0; r < prob.n_replicas; ++r) {
        double T = replica_temperature(prob, r);
        build_boltzmann_table(T, tables.data() + (std::size_t)r * stride);
    }

    // ---- 3. CPU reference (timed) ------------------------------------------
    std::vector<McResult> res_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    sample_cpu(prob, tables, res_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 4. GPU MC (kernel timed inside the wrapper) -----------------------
    std::vector<McResult> res_gpu;
    float gpu_kernel_ms = 0.0f;
    sample_gpu(prob, tables, res_gpu, &gpu_kernel_ms);

    // ---- 5. Verify: EXACT integer match, then summarize --------------------
    // Energies are integers (= -contacts) computed by the same code on both
    // sides, so the correct tolerance is ZERO (PATTERNS.md §4 "exact" row).
    int mismatches = 0;
    int best_overall = 0;          // most negative best_energy across replicas
    int best_replica = 0;          // which replica found it
    long long sum_best = 0;        // for the mean (integer sum -> deterministic)
    for (int r = 0; r < prob.n_replicas; ++r) {
        if (res_cpu[r].best_energy  != res_gpu[r].best_energy ||
            res_cpu[r].final_energy != res_gpu[r].final_energy) ++mismatches;
        sum_best += res_gpu[r].best_energy;
        if (res_gpu[r].best_energy < best_overall) {
            best_overall = res_gpu[r].best_energy;
            best_replica = r;
        }
    }
    const bool pass = (mismatches == 0);

    // Count how many H residues the sequence has (context for "max possible").
    int n_h = 0; for (int i = 0; i < prob.n; ++i) n_h += prob.hp[i];

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("[reduced-scope teaching model: 2-D HP lattice protein, Metropolis MC]\n");
    // Echo the sequence so the output is self-describing.
    std::printf("sequence (n=%d, %d H): ", prob.n, n_h);
    for (int i = 0; i < prob.n; ++i) std::putchar(prob.hp[i] ? 'H' : 'P');
    std::putchar('\n');
    std::printf("replicas = %d, sweeps = %d, T in [%.2f, %.2f]\n",
                prob.n_replicas, prob.sweeps, prob.t_min, prob.t_max);
    // Lowest energy found = MOST H-H contacts buried = best fold the ensemble saw.
    std::printf("best energy found = %d (%d H-H contacts) by replica %d\n",
                best_overall, -best_overall, best_replica);
    // A deterministic ensemble statistic: the integer-summed mean best energy,
    // printed as a fraction so it stays exact and reproducible.
    std::printf("ensemble mean best energy = %lld/%d\n", sum_best, prob.n_replicas);
    std::printf("RESULT: %s (GPU per-replica energies match CPU exactly)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s\n", source);
    std::fprintf(stderr, "[timing] CPU MC: %.3f ms   GPU MC: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- the GPU's edge grows with the "
                         "replica count; production runs use thousands of replicas.\n");
    std::fprintf(stderr, "[verify] replica mismatches = %d (integer energy => exact match)\n",
                 mismatches);

    return pass ? 0 : 1;
}
