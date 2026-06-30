// ===========================================================================
// src/main.cu  --  Entry point: load data, run CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 3.24 : Methylation / Modified-Base Calling
//
// 5-step shape (the same skeleton every project in this repo follows):
//   1. Load the demo instance (reference pore models + nanopore reads + jobs).
//   2. CPU reference: per-job LLRs by banded event-alignment DP (reference_cpu).
//   3. GPU: the same per-job LLRs, one thread per job (kernels.cu).
//   4. VERIFY: GPU LLRs match the CPU LLRs within a documented tolerance.
//   5. REPORT: deterministic per-site methylation calls + accuracy vs ground
//      truth, to stdout; timings + the max error to stderr.
//
// WHY SPLIT STREAMS: demo/run_demo diffs STDOUT against expected_output.txt, so
// stdout must be byte-identical every run -> only deterministic results go there.
// Timings (which vary run to run) go to STDERR, shown but not diffed (PATTERNS §3).
//
// Code tour: start here, then meth_core.h (the physics), reference_cpu.h/.cpp
// (the trusted baseline), then kernels.cuh -> kernels.cu (the GPU twin).
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // score_jobs_gpu, MethData, Job
#include "reference_cpu.h"    // load_meth_data, score_jobs_cpu, call_sites
#include "util/io.hpp"        // util::CpuTimer, util::max_abs_err

static const char* PROJECT_ID   = "3.24";
static const char* PROJECT_NAME = "Methylation / Modified-Base Calling";

// Verification tolerance. The CPU and GPU run the SAME banded DP (meth_core.h)
// over the SAME inputs; the only divergence is the GPU's fused multiply-add (FMA)
// contracting `a*b + c` differently from the host in the double-precision
// emission/transition sums. Over a 10-event DP that is well under 1e-4 in the LLR
// (a difference of two ~tens-of-units log-likelihoods). We verify the float LLRs
// agree to 1e-3, which is far tighter than the LLR magnitudes that drive a call
// (PATTERNS.md §4: a small physical tolerance for an FMA-sensitive computation).
static constexpr double TOLERANCE = 1.0e-3;

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/methylation_sample.txt";
    MethData d;
    try {
        d = load_meth_data(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int num_jobs = static_cast<int>(d.jobs.size());

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<float> llr_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    score_jobs_cpu(d, llr_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU (kernel timed) --------------------------------------------
    std::vector<float> llr_gpu;
    float gpu_kernel_ms = 0.0f;
    score_jobs_gpu(d, llr_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    const double err = util::max_abs_err(llr_cpu, llr_gpu);
    const bool pass = err <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    // Aggregate the GPU LLRs into per-site mean LLRs and 0/1 calls (the same
    // deterministic decision the CPU would make; call_sites is shared).
    std::vector<float> mean_llr;
    std::vector<int>   call;
    call_sites(d, llr_gpu, mean_llr, call);

    // Compare the calls to the synthetic ground truth to report accuracy. (This
    // validates the SCIENCE, not just CPU==GPU agreement: a correct DP + LLR
    // should recover which sites we built as methylated.)
    int correct = 0;
    for (int s = 0; s < d.num_sites; ++s)
        if (call[s] == d.truth[s]) ++correct;

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("nanopore methylation calling: %d sites x %d reads = %d alignment jobs\n",
                d.num_sites, d.coverage, num_jobs);
    std::printf("per-site 5mC calls (CpG):\n");
    std::printf("  site  ref_pos  mean_LLR   call    truth\n");
    for (int s = 0; s < d.num_sites; ++s) {
        std::printf("  %3d   %6d   %+8.3f   %-5s   %-5s\n",
                    s, d.site_pos[s], mean_llr[s],
                    call[s]    ? "5mC" : "C",
                    d.truth[s] ? "5mC" : "C");
    }
    std::printf("calls matching ground truth: %d of %d\n", correct, d.num_sites);
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-03)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d sites, %d reads, coverage %d, %d jobs)\n",
                 path.c_str(), d.num_sites, d.num_reads, d.coverage, num_jobs);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernel: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- this tiny instance is launch-bound; the GPU's "
                         "edge grows with reads (real 30x WGS = billions of signal samples).\n");
    std::fprintf(stderr, "[verify] max_abs_err(LLR) = %.3e  (tolerance %.1e)\n", err, TOLERANCE);

    // Exit code: 0 only if the GPU reproduced the CPU within tolerance.
    return pass ? 0 : 1;
}
