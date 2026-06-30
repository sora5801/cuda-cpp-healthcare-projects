// ===========================================================================
// src/main.cu  --  Entry point: balance a Hi-C matrix, find TADs, verify, report
// ---------------------------------------------------------------------------
// Project 3.15 : Hi-C / 3D Genome Contact Analysis
//
// 5-step shape (every project in this repo follows it):
//   1. Load the sparse contact matrix (data/sample, or a path argument).
//   2. CPU reference ICE balancing (reference_cpu.cpp)        -> trusted bias.
//   3. GPU ICE balancing (kernels.cu): parallel fixed-point row-sum reduction.
//   4. VERIFY: GPU bias == CPU bias (fixed-point atomics => exact agreement),
//      to a documented tolerance.
//   5. REPORT (deterministic -> stdout): bias summary, insulation score, and the
//      called TAD boundaries. Timing/detail -> stderr.
//
// The downstream insulation score + boundary calling run on the BALANCED matrix
// using the GPU's bias, so the reported biology depends on the GPU result.
//
// Code tour: start here, then hic.h (shared math), reference_cpu.cpp (baseline),
// kernels.cu (GPU twin). See ../THEORY.md for the science and the GPU mapping.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // ice_balance_gpu (also pulls in reference_cpu.h)
#include "reference_cpu.h"    // load_matrix, ice_balance_cpu, insulation_score, ...
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "3.15";
static const char* PROJECT_NAME = "Hi-C / 3D Genome Contact Analysis";

// ICE iteration count: fixed so the run is deterministic. ~30 iterations is
// plenty to balance the tiny sample (real runs use ~20-50 with a tolerance stop).
static constexpr int    ICE_ITERS = 30;

// Insulation diamond half-size (in bins) and local-minimum search radius (bins).
// The sample is engineered (data/README.md) so two TAD borders sit at known bins;
// these settings recover them. Both are deterministic constants.
static constexpr int    INSULATION_WINDOW = 3;
// Radius 1 = compare each bin against its immediate neighbours. The insulation
// window already blanks the 3 bins at each matrix edge, so a wider radius would
// disqualify the bin-4 border (its radius-2 neighbour, bin 2, is an edge n/a).
static constexpr int    BOUNDARY_RADIUS   = 1;

// Verification tolerance on the per-bin bias. The reduction is fixed-point exact
// (CPU and GPU sum identical integers), but the host bias update does ~30 rounds
// of double-precision multiply/divide; FMA contraction differences between the
// host compiler and nvcc can drift the bias by ~1e-12 over those rounds. We
// verify to 1e-9 -- far below any biological significance (docs/PATTERNS.md §4).
static constexpr double BIAS_TOLERANCE = 1.0e-9;

int main(int argc, char** argv) {
    // ---- 1. Load ----------------------------------------------------------
    const std::string path =
        (argc > 1) ? argv[1] : "data/sample/hic_sample.txt";
    HicMatrix m;
    try {
        m = load_matrix(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference ICE (timed) -------------------------------------
    std::vector<double> bias_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    const double var_cpu = ice_balance_cpu(m, ICE_ITERS, bias_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU ICE (kernel time via CUDA events) -------------------------
    std::vector<double> bias_gpu;
    float gpu_kernel_ms = 0.0f;
    const double var_gpu = ice_balance_gpu(m, ICE_ITERS, bias_gpu, &gpu_kernel_ms);

    // ---- 4. Verify GPU bias vs CPU bias -----------------------------------
    double max_bias_diff = 0.0;
    for (int k = 0; k < m.n; ++k)
        max_bias_diff = std::fmax(max_bias_diff,
                                  std::fabs(bias_cpu[k] - bias_gpu[k]));
    const bool pass = (max_bias_diff <= BIAS_TOLERANCE);

    // ---- 5. Downstream biology on the BALANCED matrix (GPU bias) ----------
    // Insulation score then TAD boundaries -- the headline result a Hi-C analyst
    // cares about. We use the GPU bias so the reported biology reflects the GPU.
    const std::vector<double> score = insulation_score(m, bias_gpu, INSULATION_WINDOW);
    const std::vector<int> boundaries = call_boundaries(score, BOUNDARY_RADIUS);

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("matrix: %d bins, %zu stored contacts (upper triangle)\n",
                m.n, m.entries.size());
    std::printf("ICE: %d iterations, balanced bias per bin (occupied bins):\n", ICE_ITERS);
    for (int k = 0; k < m.n; ++k) {
        // Print biases at 6 decimals -- deterministic across CPU/GPU at our tol.
        std::printf("  bin %2d: bias = %.6f\n", k, bias_gpu[k]);
    }
    std::printf("insulation score (window=%d):\n", INSULATION_WINDOW);
    for (int k = 0; k < m.n; ++k) {
        if (score[k] < 0.0) std::printf("  bin %2d:   n/a (edge)\n", k);
        else                std::printf("  bin %2d: %.6f\n", k, score[k]);
    }
    std::printf("TAD boundaries (local minima, radius=%d): %zu found\n",
                BOUNDARY_RADIUS, boundaries.size());
    for (int b : boundaries) std::printf("  boundary at bin %d\n", b);
    std::printf("RESULT: %s (GPU bias matches CPU reference within %.0e)\n",
                pass ? "PASS" : "FAIL", BIAS_TOLERANCE);

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d bins, %zu contacts)\n",
                 path.c_str(), m.n, m.entries.size());
    std::fprintf(stderr, "[timing] CPU ICE: %.3f ms   GPU ICE (kernels): %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- the GPU edge grows with matrix size; "
                         "real Hi-C has 10^6-10^9 nonzeros.\n");
    std::fprintf(stderr, "[verify] max |bias_cpu - bias_gpu| = %.3e (tol %.0e); "
                         "convergence var cpu/gpu = %.3e / %.3e\n",
                 max_bias_diff, BIAS_TOLERANCE, var_cpu, var_gpu);

    return pass ? 0 : 1;
}
