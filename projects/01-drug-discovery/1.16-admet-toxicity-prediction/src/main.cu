// ===========================================================================
// src/main.cu  --  Entry point: load data, run CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 1.16 : ADMET / Toxicity Prediction  (reduced-scope teaching version)
//
// WHAT THIS FILE DOES  (the 5-step shape EVERY project in this repo follows)
//   1. Load the screening problem (N molecules + M endpoint models) from
//      data/sample (or a path given on the command line).
//   2. Compute the CPU reference (reference_cpu.cpp)         -> trusted answer.
//   3. Compute the GPU result    (kernels.cu)                -> the thing taught.
//   4. VERIFY: GPU probability matrix agrees with CPU within tolerance AND the
//      GPU per-endpoint flag counts equal the CPU's exactly                -> ok.
//   5. REPORT: a deterministic triage summary to stdout; timing to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Anything that varies run-to-run (timings) goes to
//   STDERR, which the demo shows but does not diff (PATTERNS.md sec.3).
//
// READ THIS FIRST in the code tour, then kernels.cuh -> kernels.cu, and
// reference_cpu.cpp for the baseline. See ../THEORY.md for the "why".
// ===========================================================================
#include <cmath>      // std::fabs
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // admet_screen_gpu (GPU path), ADMET_D/M
#include "reference_cpu.h"    // load_admet, admet_predict_cpu, admet_reduce
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "1.16";
static const char* PROJECT_NAME = "ADMET / Toxicity Prediction";

// Verification tolerance for the probability MATRIX. CPU and GPU run the exact
// same double-precision operation sequence (shared admet_core.h, no FMA), so
// they agree to ~1e-13 in practice; 1e-9 is a documented, comfortable margin
// (the machine-precision class, PATTERNS.md sec.4). The per-endpoint flag COUNTS
// are integers and must match EXACTLY (== 0 difference).
static constexpr double TOLERANCE = 1.0e-9;

// max_abs_err over two double arrays (returns +inf on a length mismatch so a
// shape bug can never masquerade as agreement).
static double max_abs_err_d(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return 1e300;
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double d = std::fabs(a[i] - b[i]);
        if (d > worst) worst = d;
    }
    return worst;
}

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/admet_sample.txt";
    AdmetData data;
    try {
        data = load_admet(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<double> probs_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    admet_predict_cpu(data, probs_cpu);                 // [n*M] probabilities
    const AdmetResult res_cpu = admet_reduce(data, probs_cpu);  // triage summary
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU result (kernels timed inside the wrapper) ------------------
    std::vector<double> probs_gpu;
    AdmetResult res_gpu;
    float gpu_kernel_ms = 0.0f;
    admet_screen_gpu(data, probs_gpu, res_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    // (a) probability matrix within tolerance ...
    const double err = max_abs_err_d(probs_cpu, probs_gpu);
    bool pass = err <= TOLERANCE;
    // (b) ... and the GPU's integer flag counts match the CPU's EXACTLY.
    for (int t = 0; t < ADMET_M; ++t)
        if (res_gpu.flagged_per_endpoint[t] != res_cpu.flagged_per_endpoint[t])
            pass = false;
    // (c) ... and the worst-molecule pick agrees.
    if (res_gpu.worst_mol != res_cpu.worst_mol) pass = false;

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("Multi-task ADMET screen: %d molecules x %d toxicity endpoints (descriptor D=%d)\n",
                data.n, ADMET_M, ADMET_D);
    std::printf("flagged molecules per endpoint (p >= %.2f):\n", ADMET_THRESHOLD);
    for (int t = 0; t < ADMET_M; ++t)
        std::printf("  %-18s %3d / %d\n",
                    data.endpoint_names[t].c_str(),
                    res_gpu.flagged_per_endpoint[t], data.n);
    const int wm = res_gpu.worst_mol;
    std::printf("worst molecule: %s  (flags %d/%d, summed risk %.6f)\n",
                data.mol_names[wm].c_str(),
                res_gpu.total_flags[wm], ADMET_M, res_gpu.worst_mol_score);
    std::printf("RESULT: %s (GPU matches CPU: probs within tol=1.0e-09, flag counts exact)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (n=%d, D=%d, M=%d; SYNTHETIC)\n",
                 path.c_str(), data.n, ADMET_D, ADMET_M);
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernels: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny sample is dominated by "
                         "launch/copy overhead; the GPU wins screening millions of molecules.\n");
    std::fprintf(stderr, "[verify] max_abs_err(probs) = %.3e  (tolerance %.1e); flag counts exact.\n",
                 err, TOLERANCE);

    return pass ? 0 : 1;
}
