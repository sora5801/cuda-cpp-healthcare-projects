// ===========================================================================
// src/main.cu  --  Entry point: load alignment, run CPU + GPU BQSR, verify, report
// ---------------------------------------------------------------------------
// Project 3.25 : Base Quality Score Recalibration (BQSR)
//
// THE 5-STEP SHAPE (every project in this repo follows it)
//   1. Load the alignment (data/sample): reference + reads + known-variant mask.
//   2. CPU reference: build the covariate table, then recalibrate (reference_cpu).
//   3. GPU path: the same two steps as kernels (accumulate + recalibrate).
//   4. VERIFY: the integer covariate tables and the recalibrated qualities are
//      IDENTICAL (exact -- integer atomics commute), tolerance 0.
//   5. REPORT: a deterministic per-Q recalibration summary to stdout; timing to
//      stderr (so demo/run_demo can diff stdout against expected_output.txt).
//
// Code tour: start here, then bqsr.h (the covariate model + math), kernels.cuh ->
// kernels.cu (the GPU twin), reference_cpu.cpp (the baseline). See ../THEORY.md.
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // bqsr_gpu (GPU path), Dataset, NUM_BINS
#include "reference_cpu.h"    // load_dataset, build_table_cpu, recalibrate_cpu
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "3.25";
static const char* PROJECT_NAME = "Base Quality Score Recalibration (BQSR)";

// ---------------------------------------------------------------------------
// per_q_summary: collapse the full (Q, cycle, context) table down to a per-Q
//   row so the report is small and human-readable. For each reported quality Q we
//   sum observations and errors over all cycles and contexts, then derive the
//   empirical quality of that aggregate. This is the headline BQSR story:
//   "you reported Q, but the bases actually erred at empirical-Q." We print only
//   Q rows that were actually observed, in ascending Q -> deterministic output.
// ---------------------------------------------------------------------------
static void per_q_summary(const std::vector<unsigned int>& obs,
                          const std::vector<unsigned int>& err) {
    for (int q = 0; q < NUM_Q; ++q) {
        unsigned long long o = 0, e = 0;
        for (int cyc = 0; cyc < MAX_CYCLE; ++cyc)
            for (int ctx = 0; ctx < NUM_CONTEXT; ++ctx) {
                const int b = covariate_index(q, cyc, ctx);
                o += obs[static_cast<std::size_t>(b)];
                e += err[static_cast<std::size_t>(b)];
            }
        if (o == 0) continue;                       // never observed -> omit row
        // Empirical Q of the aggregate, same +1-corrected formula as the bins.
        const int qe = empirical_q(static_cast<unsigned int>(o),
                                   static_cast<unsigned int>(e));
        std::printf("  Q=%2d  obs=%6llu  err=%5llu  ->  Q_emp=%2d\n",
                    q, o, e, qe);
    }
}

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/bqsr_sample.txt";
    Dataset d;
    try {
        d = load_dataset(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed): build table, then recalibrate -----------
    std::vector<unsigned int> obs_cpu, err_cpu;
    std::vector<int> newq_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    build_table_cpu(d, obs_cpu, err_cpu);
    recalibrate_cpu(d, obs_cpu, err_cpu, newq_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU path (kernels timed inside the wrapper) --------------------
    std::vector<unsigned int> obs_gpu, err_gpu;
    std::vector<int> newq_gpu;
    float gpu_kernel_ms = 0.0f;
    bqsr_gpu(d, obs_gpu, err_gpu, newq_gpu, &gpu_kernel_ms);

    // ---- 4. Verify: tables + recalibrated qualities EXACTLY equal ----------
    int table_mismatch = 0;
    for (int b = 0; b < NUM_BINS; ++b)
        if (obs_cpu[b] != obs_gpu[b] || err_cpu[b] != err_gpu[b]) ++table_mismatch;
    int qual_mismatch = 0;
    for (int g = 0; g < d.total_bases(); ++g)
        if (newq_cpu[g] != newq_gpu[g]) ++qual_mismatch;
    const bool pass = (table_mismatch == 0) && (qual_mismatch == 0);

    // Aggregate stats for the report (computed from the GPU result; identical to
    // CPU when pass==true). How many bases were tallied vs masked, and how many
    // qualities actually changed.
    unsigned long long tallied = 0, errors = 0;
    for (int b = 0; b < NUM_BINS; ++b) { tallied += obs_gpu[b]; errors += err_gpu[b]; }
    int changed = 0;
    for (int g = 0; g < d.total_bases(); ++g)
        if (newq_gpu[g] != d.read_quals[g]) ++changed;

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("alignment: %d reads x %d bp, reference %zu bp, %d covariate bins\n",
                d.num_reads, d.read_len, d.reference.size(), NUM_BINS);
    std::printf("bases tallied = %llu (of %d; rest masked/skipped), observed errors = %llu\n",
                tallied, d.total_bases(), errors);
    std::printf("per-reported-Q recalibration (aggregated over cycle & context):\n");
    per_q_summary(obs_gpu, err_gpu);
    std::printf("recalibrated qualities changed = %d / %d bases\n", changed, d.total_bases());
    std::printf("RESULT: %s (GPU table + recalibrated Q match CPU exactly)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s\n", path.c_str());
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernels: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- this tile is tiny and launch-bound; "
                         "the GPU's edge appears at WGS scale (~1e11 bases).\n");
    std::fprintf(stderr, "[verify] table mismatches = %d, quality mismatches = %d "
                         "(both 0 => integer atomics reproduce the CPU exactly)\n",
                 table_mismatch, qual_mismatch);

    return pass ? 0 : 1;
}
