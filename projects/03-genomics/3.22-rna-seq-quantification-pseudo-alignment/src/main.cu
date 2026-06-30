// ===========================================================================
// src/main.cu  --  Entry point: quantify transcripts by EM, verify, report
// ---------------------------------------------------------------------------
// Project 3.22 : RNA-seq Quantification / Pseudo-alignment
//
// 5-step shape (mirrors every flagship):
//   1. Load the equivalence-class problem (data/sample): T transcripts, M ecs.
//   2. CPU reference EM (reference_cpu.cpp) -> trusted abundances.
//   3. GPU EM (kernels.cu): per-ec E-step + fixed-point atomic M-step.
//   4. VERIFY: GPU rho matches CPU rho exactly (integer atomics commute).
//   5. REPORT: deterministic per-transcript counts / TPM, recovery vs. the
//      synthetic ground truth, and the GPU-vs-CPU verdict.
//
// Code tour: start here, then pseudoalign.h (the E-step math), reference_cpu.cpp
// (loader + serial EM), kernels.cu (the GPU twin).
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // em_gpu, EcDataset
#include "reference_cpu.h"    // load_dataset, em_cpu, tpm_from_rho
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "3.22";
static const char* PROJECT_NAME = "RNA-seq Quantification / Pseudo-alignment";

// Fixed number of EM iterations -> deterministic, and CPU/GPU run lockstep.
static constexpr int    ITERS     = 100;
// rho is built from identical integer fixed-point sums on both sides, so the only
// possible difference is the tiny rounding inside counts_to_rho; in practice the
// max abs difference is 0. We allow a hair of slack and report the real number.
static constexpr double TOLERANCE = 1.0e-12;

int main(int argc, char** argv) {
    // ---- 1. Load ----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1]
                                        : "data/sample/rnaseq_ec_sample.txt";
    EcDataset d;
    try {
        d = load_dataset(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference EM (timed) -------------------------------------
    std::vector<double> rho_cpu, counts_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    const double delta_cpu = em_cpu(d, ITERS, rho_cpu, counts_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU EM (kernel-timed) ----------------------------------------
    std::vector<double> rho_gpu, counts_gpu;
    float gpu_kernel_ms = 0.0f;
    const double delta_gpu = em_gpu(d, ITERS, rho_gpu, counts_gpu, &gpu_kernel_ms);

    // ---- 4. Verify (GPU rho vs CPU rho) ----------------------------------
    double rho_diff = 0.0;
    for (int t = 0; t < d.T; ++t)
        rho_diff = std::fmax(rho_diff, std::fabs(rho_cpu[t] - rho_gpu[t]));
    const bool pass = (rho_diff <= TOLERANCE);

    // Convert the GPU abundances to TPM for the report (the standard unit).
    std::vector<double> tpm_gpu;
    tpm_from_rho(d, rho_gpu, tpm_gpu);

    // ---- 5a. Deterministic report -> STDOUT ------------------------------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("EM quantification: %d transcripts, %d equivalence classes, "
                "%.0f reads, %d iterations\n",
                d.T, d.M, d.total_reads, ITERS);
    std::printf("  %-4s %12s %12s %12s", "id", "est_counts", "rho", "TPM");
    if (!d.truth_rho.empty()) std::printf(" %12s", "truth_rho");
    std::printf("\n");
    for (int t = 0; t < d.T; ++t) {
        std::printf("  t%-3d %12.4f %12.6f %12.2f",
                    t, counts_gpu[t], rho_gpu[t], tpm_gpu[t]);
        if (!d.truth_rho.empty()) std::printf(" %12.6f", d.truth_rho[t]);
        std::printf("\n");
    }

    // If a ground truth is present, report how well the EM recovered it: the L1
    // distance between estimated and true abundance vectors (0 = perfect). This
    // validates the SCIENCE (did we recover the right answer?), not just CPU==GPU.
    if (!d.truth_rho.empty()) {
        double l1 = 0.0;
        for (int t = 0; t < d.T; ++t) l1 += std::fabs(rho_gpu[t] - d.truth_rho[t]);
        std::printf("recovery: L1(estimated rho, truth rho) = %.4f\n", l1);
    }
    std::printf("RESULT: %s (GPU abundances match CPU reference)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR ------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d transcripts, %d ecs)\n",
                 path.c_str(), d.T, d.M);
    std::fprintf(stderr, "[timing] CPU EM: %.3f ms   GPU EM loop: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- this sample is tiny, so the "
                         "per-iteration launch + copy overhead dominates; the GPU's "
                         "edge appears at 10^5-10^7 ecs (real RNA-seq).\n");
    std::fprintf(stderr, "[verify] max |rho_cpu - rho_gpu| = %.3e (tol %.1e); "
                         "final L1 step delta cpu/gpu = %.3e / %.3e\n",
                 rho_diff, TOLERANCE, delta_cpu, delta_gpu);

    return pass ? 0 : 1;
}
