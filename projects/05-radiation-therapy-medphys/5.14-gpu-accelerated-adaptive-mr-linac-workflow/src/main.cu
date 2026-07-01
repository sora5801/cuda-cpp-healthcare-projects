// ===========================================================================
// src/main.cu  --  Entry point: load case, run CPU + GPU oART, verify, report
// ---------------------------------------------------------------------------
// Project 5.14 : GPU-Accelerated Adaptive MR-Linac Workflow (reduced-scope)
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the oART case (data/sample, else fail loudly).
//   2. Run the CPU reference workflow (reference_cpu.cpp)   -> trusted answer.
//   3. Run the GPU workflow            (kernels.cu)          -> the thing taught.
//   4. VERIFY: GPU displacement field + warped dose + metrics agree with the CPU
//      within a documented tolerance                         -> correctness.
//   5. REPORT: deterministic plan-approval numbers to stdout; timing to stderr.
//
//   THE STORY THE OUTPUT TELLS. The synthetic daily MR is the planning MR with
//   the tumour shifted (a full bladder pushed it). Before registration the images
//   mismatch (high MSE) and the planned dose, if delivered blindly, would MISS the
//   moved tumour (low coverage). After Demons registration the MSE collapses and
//   the dose, warped onto the daily anatomy, again covers the target -- exactly
//   what online adaptation buys you. We print both so the learner sees the gain.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Run-varying numbers (timings) go to STDERR.
//
// READ THIS FIRST in the code tour, then mrl_registration.h (the physics),
//   kernels.cuh -> kernels.cu (GPU), reference_cpu.cpp (baseline). Why: THEORY.md.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // oart_gpu (GPU path)
#include "reference_cpu.h"    // oart_cpu (CPU baseline), OartCase/OartResult
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "5.14";
static const char* PROJECT_NAME = "GPU-Accelerated Adaptive MR-Linac Workflow";

// Correctness tolerance. Registration is an ITERATIVE double-precision solver with
// bilinear gathers and Gaussian smoothing; over many iterations the GPU's fused
// multiply-add and the host compiler diverge by ~1e-11 even though the arithmetic
// is "the same" (PATTERNS.md section 4, the long-iterative case). We verify the
// displacement field and warped dose to a physically-negligible tolerance and say
// so, rather than pretending the two are bit-identical.
static constexpr double TOLERANCE = 1.0e-6;

// max_abs_err over two double vectors (util/io.hpp's helper is float-only).
static double max_abs_err_d(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return 1e300;      // shape bug -> never "agree"
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs(a[i] - b[i]);
        if (d > worst) worst = d;
    }
    return worst;
}

int main(int argc, char** argv) {
    // ---- 1. Load ------------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/oart_case.txt";
    OartCase c;
    try {
        c = load_case(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) ------------------------------------------
    OartResult cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    oart_cpu(c, cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU workflow (kernels timed via CUDA events inside oart_gpu) ----
    OartResult gpu;
    float gpu_kernel_ms = 0.0f;
    oart_gpu(c, gpu, &gpu_kernel_ms);

    // ---- 4. Verify (field + warped dose agree; metrics agree) --------------
    const double err_u    = max_abs_err_d(cpu.u,           gpu.u);
    const double err_v    = max_abs_err_d(cpu.v,           gpu.v);
    const double err_dose = max_abs_err_d(cpu.warped_dose, gpu.warped_dose);
    const double err = std::fmax(std::fmax(err_u, err_v), err_dose);
    const bool pass = (err <= TOLERANCE)
                   && (std::fabs(cpu.mean_gtv_dose - gpu.mean_gtv_dose) <= TOLERANCE)
                   && (std::fabs(cpu.d95           - gpu.d95)           <= TOLERANCE);

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    // We report the GPU numbers (verified equal to the CPU) at fixed precision.
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("[reduced-scope teaching version: 2-D Demons registration + dose warp; "
                "synthetic data]\n");
    std::printf("case: %dx%d voxels, %d Demons iters, sigma=%.2f, K=%.2f, dose_thresh=%.2f Gy\n",
                c.nx, c.ny, c.iters, c.sigma, c.k_norm, c.dose_thresh);
    std::printf("registration MSE(moving vs fixed): before=%.6f  after=%.6f\n",
                gpu.mse_before, gpu.mse_after);
    // A stable summary of the displacement field: its peak magnitude (voxels).
    double umax = 0.0;
    for (std::size_t i = 0; i < gpu.u.size(); ++i) {
        const double mag = std::sqrt(gpu.u[i]*gpu.u[i] + gpu.v[i]*gpu.v[i]);
        if (mag > umax) umax = mag;
    }
    std::printf("peak displacement magnitude: %.6f voxels\n", umax);
    std::printf("GTV plan metrics on WARPED dose:  mean=%.6f Gy  D95=%.6f Gy  coverage(>=%.2f Gy)=%.6f\n",
                gpu.mean_gtv_dose, gpu.d95, c.dose_thresh, gpu.gtv_coverage);
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-06)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%dx%d, %d iters)\n",
                 path.c_str(), c.nx, c.ny, c.iters);
    std::fprintf(stderr, "[timing] CPU workflow: %.3f ms   GPU kernels: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- a real oART slab is 3-D and far "
                         "larger; per-iteration kernel launches dominate at this tiny size.\n");
    std::fprintf(stderr, "[verify] max|GPU-CPU|: field=%.3e dose=%.3e  (tolerance %.1e)\n",
                 std::fmax(err_u, err_v), err_dose, TOLERANCE);

    return pass ? 0 : 1;
}
