// ===========================================================================
// src/main.cu  --  Entry point: load plan, optimize on CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 5.2 : Radiotherapy Treatment-Plan Optimization
//
// WHAT THIS FILE DOES  (the 5-step shape every project in this repo follows)
//   1. Load the FMO problem (sparse dose-influence matrix D + objective specs).
//   2. CPU reference: projected-gradient optimize the fluence (reference_cpu.cpp).
//   3. GPU: the SAME optimizer with cuSPARSE doing the two SpMVs (kernels.cu).
//   4. VERIFY: the GPU fluence and the resulting dose match the CPU within a
//      documented tolerance (long iterative solver -> small physical tolerance;
//      see THEORY.md section 5).
//   5. REPORT: a DETERMINISTIC plan-quality summary (DVH-style stats derived
//      from the CPU dose) to stdout; timings + the measured error to stderr.
//
//   The reported stats come from the CPU dose so stdout is byte-for-byte stable
//   across runs (the GPU dose can differ in the last digits due to float
//   summation order); the CPU-vs-GPU agreement is asserted separately and its
//   numeric size is printed to stderr.
//
// Code tour: start in fmo.h (the shared per-voxel math), then reference_cpu.h /
//   .cpp (the CSR problem + CPU optimizer), then kernels.cuh / kernels.cu (the
//   cuSPARSE GPU optimizer), then back here for how it is all driven.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // optimize_gpu
#include "reference_cpu.h"    // Problem, PlanStats, load_problem, optimize_cpu, ...
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "5.2";
static const char* PROJECT_NAME = "Radiotherapy Treatment-Plan Optimization";

// Verification tolerances. This is a LONG iterative solver (hundreds of gradient
// steps), and GPU fused-multiply-add + cuSPARSE's parallel summation order
// diverge from the host compiler's serial float order by a tiny amount that
// accumulates. We therefore verify to a small PHYSICAL tolerance, not bit
// equality (PATTERNS.md section 4). DOSE_TOL is in Gray; FLU_TOL is on fluence.
static constexpr double DOSE_TOL = 1.0e-2;   // Gy : negligible vs a ~60 Gy Rx
static constexpr double FLU_TOL  = 1.0e-2;   // fluence units

int main(int argc, char** argv) {
    // ---- 1. Load the problem -----------------------------------------------
    const std::string path = (argc > 1) ? argv[1]
                                        : "data/sample/plan_sample.txt";
    Problem p;
    try {
        p = load_problem(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference optimize (timed) ---------------------------------
    std::vector<float> x_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    optimize_cpu(p, x_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU optimize (loop timed with CUDA events) ---------------------
    std::vector<float> x_gpu;
    float gpu_ms = 0.0f;
    optimize_gpu(p, x_gpu, &gpu_ms);

    // ---- 4. Verify ---------------------------------------------------------
    // (a) fluence agreement, beamlet by beamlet.
    double flu_err = 0.0;
    for (int j = 0; j < p.n_beam; ++j)
        flu_err = std::fmax(flu_err, std::fabs((double)x_cpu[j] - (double)x_gpu[j]));

    // (b) the physically meaningful check: do the two fluences deposit the same
    //     dose? Recompute each dose from its fluence and compare voxelwise.
    std::vector<float> dose_cpu, dose_gpu;
    csr_spmv_cpu(p, x_cpu, dose_cpu);
    csr_spmv_cpu(p, x_gpu, dose_gpu);
    double dose_err = 0.0;
    for (int v = 0; v < p.n_vox; ++v)
        dose_err = std::fmax(dose_err, std::fabs((double)dose_cpu[v] - (double)dose_gpu[v]));

    const bool pass = (flu_err <= FLU_TOL) && (dose_err <= DOSE_TOL);

    // Stats for the report come from the CPU dose -> deterministic stdout.
    const PlanStats st = compute_stats(p, dose_cpu);

    // Count voxels per structure for the header (deterministic).
    int n_ptv = 0, n_oar = 0, n_body = 0;
    for (const VoxelSpec& s : p.voxels) {
        if (s.kind == STRUCT_PTV) ++n_ptv;
        else if (s.kind == STRUCT_OAR) ++n_oar;
        else ++n_body;
    }

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("Fluence-map optimization: %d voxels (PTV %d, OAR %d, BODY %d), "
                "%d beamlets, nnz=%d\n",
                p.n_vox, n_ptv, n_oar, n_body, p.n_beam, p.nnz());
    std::printf("optimizer: projected gradient, %d iters, step=%.3f, Rx=%.1f Gy\n",
                p.iters, p.step, p.d_rx);
    std::printf("final objective F(x) = %.4f\n", st.objective);
    std::printf("PTV dose (Gy): mean %.3f  min %.3f  max %.3f  homogeneity %.4f\n",
                st.ptv_mean, st.ptv_min, st.ptv_max, st.homogeneity);
    std::printf("OAR dose (Gy): mean %.3f  max %.3f  (tolerance-limited sparing)\n",
                st.oar_mean, st.oar_max);
    std::printf("RESULT: %s (GPU plan matches CPU within dose tol=%.1e Gy)\n",
                pass ? "PASS" : "FAIL", DOSE_TOL);

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%d voxels, %d beamlets, nnz=%d)\n",
                 path.c_str(), p.n_vox, p.n_beam, p.nnz());
    std::fprintf(stderr, "[timing] CPU optimize: %.3f ms   GPU optimize (cuSPARSE): %.3f ms\n",
                 cpu_ms, gpu_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- the GPU's edge grows with "
                         "matrix size; tiny samples are launch/setup bound.\n");
    std::fprintf(stderr, "[verify] max fluence diff = %.3e (tol %.1e)   "
                         "max dose diff = %.3e Gy (tol %.1e)\n",
                 flu_err, FLU_TOL, dose_err, DOSE_TOL);

    return pass ? 0 : 1;
}
