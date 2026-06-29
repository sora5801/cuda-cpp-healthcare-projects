// ===========================================================================
// src/main.cu  --  Entry point: load trajectory, run CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 1.30 : Trajectory RMSD, Clustering & Contact Analysis
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the trajectory (from data/sample, or a CLI path).
//   2. Compute the CPU reference (reference_cpu.cpp)         -> trusted answer.
//   3. Compute the GPU result    (kernels.cu)                -> the thing taught.
//   4. VERIFY: assert GPU agrees with CPU within a tolerance -> correctness.
//   5. REPORT: deterministic result to stdout; timing to stderr.
//
//   STDOUT is kept byte-for-byte deterministic so demo/run_demo can diff it
//   against demo/expected_output.txt. Anything that varies run-to-run (timings)
//   goes to STDERR, which the demo shows but does not diff. The clustering
//   histogram is computed on the CPU from the per-frame RMSDs (a deterministic
//   reduction) and printed to stdout -- it is the "clustering" in the title.
//
// READ THIS FIRST in the code tour, then kernels.cuh -> kernels.cu, then
//   reference_cpu.* and rmsd_core.h. See ../THEORY.md for the "why".
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // analyze_trajectory_gpu (GPU path), Trajectory
#include "reference_cpu.h"    // load_trajectory, analyze_trajectory_cpu, cluster_by_rmsd
#include "util/io.hpp"        // util::CpuTimer

// Program identity (kept in sync with demo/expected_output.txt).
static const char* PROJECT_ID   = "1.30";
static const char* PROJECT_NAME = "Trajectory RMSD, Clustering & Contact Analysis";

// Correctness tolerance. Both sides run the SAME double-precision rmsd_core.h
// math, so they agree to ~machine epsilon; we verify to 1e-9 (machine-precision
// class, PATTERNS.md sec 4 -- a short FP64 computation, not a long iterative
// solver). The tiny residual comes only from the order of FP adds in the
// covariance/eigenvalue, which is identical here -> err is effectively 0.
static constexpr double TOLERANCE = 1.0e-9;

// RMSD clustering bin width (length units). Frames are grouped into RMSD shells
// of this width (see cluster_by_rmsd in reference_cpu.cpp / THEORY).
static constexpr double CLUSTER_WIDTH = 1.0;

// max_abs_err over two double arrays (our headline correctness metric). Returns
// +inf on a length mismatch so a shape bug can never masquerade as agreement.
static double max_abs_err(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return 1.0e300;   // sentinel "infinite" error
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = (a[i] > b[i]) ? (a[i] - b[i]) : (b[i] - a[i]);
        if (d > worst) worst = d;
    }
    return worst;
}

int main(int argc, char** argv) {
    // ---- 1. Load the trajectory --------------------------------------------
    const std::string path = (argc > 1) ? argv[1]
                                        : "data/sample/trajectory_sample.txt";
    Trajectory traj;
    try {
        traj = load_trajectory(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) ------------------------------------------
    FrameMetrics cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    analyze_trajectory_cpu(traj, cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU result (kernel timed inside the wrapper) -------------------
    FrameMetrics gpu;
    float gpu_kernel_ms = 0.0f;
    analyze_trajectory_gpu(traj, gpu, &gpu_kernel_ms);

    // ---- 4. Verify (both metrics) ------------------------------------------
    const double err_rmsd = max_abs_err(cpu.rmsd, gpu.rmsd);
    const double err_qnc  = max_abs_err(cpu.qnc,  gpu.qnc);
    const double err = (err_rmsd > err_qnc) ? err_rmsd : err_qnc;
    const bool pass = err <= TOLERANCE;

    // Clustering: a deterministic RMSD-shell histogram from the GPU RMSDs.
    std::vector<int> clusters;
    cluster_by_rmsd(gpu, CLUSTER_WIDTH, clusters);

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("frames=%d  atoms=%d  reference=frame %d\n",
                traj.n_frames, N_ATOMS, traj.ref);
    std::printf("per-frame RMSD (to reference, optimal superposition) and Q (native contacts):\n");
    for (int f = 0; f < traj.n_frames; ++f)
        std::printf("  frame %2d   RMSD = %8.4f   Q = %6.4f\n",
                    f, gpu.rmsd[static_cast<std::size_t>(f)],
                    gpu.qnc[static_cast<std::size_t>(f)]);

    // RMSD-shell clustering histogram (width = CLUSTER_WIDTH).
    std::printf("RMSD clustering (shell width = %.1f):\n", CLUSTER_WIDTH);
    for (std::size_t b = 0; b < clusters.size(); ++b)
        std::printf("  shell %zu  [%.1f, %.1f):  %d frame(s)\n",
                    b, static_cast<double>(b) * CLUSTER_WIDTH,
                    (static_cast<double>(b) + 1.0) * CLUSTER_WIDTH, clusters[b]);

    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-09)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%d frames x %d atoms)\n",
                 path.c_str(), traj.n_frames, N_ATOMS);
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny sample is dominated "
                         "by launch/copy overhead; the GPU wins at trajectory scale "
                         "(millions of frames).\n");
    std::fprintf(stderr, "[verify] max_abs_err: rmsd=%.3e  Q=%.3e  (tolerance %.1e)\n",
                 err_rmsd, err_qnc, TOLERANCE);

    return pass ? 0 : 1;
}
