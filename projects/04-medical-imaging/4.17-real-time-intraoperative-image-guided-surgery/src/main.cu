// ===========================================================================
// src/main.cu  --  Entry point: load clouds, run CPU + GPU ICP, verify, report
// ---------------------------------------------------------------------------
// Project 4.17 : Real-Time Intraoperative / Image-Guided Surgery
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the two point clouds (moving pre-op P, fixed intra-op Q) from
//      data/sample, or fall back to a built-in synthetic pair.
//   2. Run the CPU reference ICP (reference_cpu.cpp)   -> trusted transform.
//   3. Run the GPU ICP        (kernels.cu)             -> the thing being taught.
//   4. VERIFY: the GPU transform equals the CPU transform (bit-for-bit, thanks
//      to the fixed-point reduction) and both convergence curves match.
//   5. REPORT: deterministic result to STDOUT; timing to STDERR.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Timings (which vary run to run) go to STDERR,
//   which the demo shows but does not diff.
//
// CODE TOUR: read this first, then icp.h (the shared math), kernels.cuh ->
//   kernels.cu (the GPU twin), and reference_cpu.cpp (the serial baseline).
//   See ../THEORY.md for the science, the math, and the GPU mapping.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // icp_gpu (GPU path), Vec3, Rigid
#include "reference_cpu.h"    // load_clouds, icp_cpu, rms_error, Clouds
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "4.17";
static const char* PROJECT_NAME = "Real-Time Intraoperative / Image-Guided Surgery";

// Number of ICP iterations. FIXED (no early stopping) so the run is perfectly
// deterministic and the CPU/GPU comparison is apples-to-apples. Twelve is plenty
// for the well-conditioned synthetic sample to converge to sub-micron RMS.
static constexpr int ICP_ITERS = 12;

// Verification tolerances (documented; see ../THEORY.md "verification"):
//   * The recovered TRANSFORMS are built from an INTEGER fixed-point reduction
//     plus the identical host 3x3 SVD, so CPU and GPU agree to the last bit.
//     We still allow a whisper of slack (1e-9) purely as defensive slack.
//   * The RMS convergence curves are computed by the same shared rms_error(),
//     so they also match to ~1e-9.
static constexpr double TRANSFORM_TOL = 1.0e-9;   // max |R,t| difference CPU vs GPU
static constexpr double HISTORY_TOL   = 1.0e-9;   // max per-iteration RMS difference

// ---------------------------------------------------------------------------
// make_synthetic: a built-in fallback pair used when no data file is given.
//   Q = a small fixed "surface": 20 points on a tilted plane patch (a stand-in
//       for a digitized organ surface). P = a rigid-transformed copy of Q by a
//       known small rotation+translation (the "misalignment" ICP must undo).
//   Deterministic and self-checking: ICP should drive RMS to ~0.
// ---------------------------------------------------------------------------
static Clouds make_synthetic() {
    Clouds c;
    // Build Q: a 5x4 grid of points on a gently tilted plane, spacing 10 mm.
    for (int iy = 0; iy < 4; ++iy) {
        for (int ix = 0; ix < 5; ++ix) {
            const float X = static_cast<float>(ix) * 10.0f - 20.0f;   // mm, centred
            const float Y = static_cast<float>(iy) * 10.0f - 15.0f;   // mm, centred
            const float Z = 0.05f * X + 0.03f * Y;                    // gentle tilt
            c.Q.push_back(Vec3{ X, Y, Z });
        }
    }
    // Ground-truth misalignment: rotate ~10 deg about Z, translate a few mm.
    const double ang = 10.0 * 3.14159265358979323846 / 180.0;
    Rigid gt = rigid_identity();
    gt.R[0][0] =  std::cos(ang); gt.R[0][1] = -std::sin(ang);
    gt.R[1][0] =  std::sin(ang); gt.R[1][1] =  std::cos(ang);
    gt.t[0] = 5.0; gt.t[1] = -3.0; gt.t[2] = 2.0;
    c.gt = gt; c.has_gt = true;
    // P = gt applied to Q (so P is Q, misaligned; ICP recovers gt^{-1}).
    for (const Vec3& q : c.Q) c.P.push_back(rigid_apply(gt, q));
    return c;
}

// Pretty-print a 3x4 rigid transform [R | t] to stdout at fixed precision so the
// output is deterministic and diffable.
static void print_transform(const char* label, const Rigid& g) {
    std::printf("%s:\n", label);
    for (int r = 0; r < 3; ++r)
        std::printf("  [ %8.5f %8.5f %8.5f | %9.5f ]\n",
                    g.R[r][0], g.R[r][1], g.R[r][2], g.t[r]);
}

int main(int argc, char** argv) {
    // ---- 1. Load ------------------------------------------------------------
    Clouds c;
    const char* source = "synthetic (built-in)";
    if (argc > 1) {
        try {
            c = load_clouds(argv[1]);
            source = argv[1];
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[error] %s\n", e.what());
            return 2;
        }
    } else {
        c = make_synthetic();
    }

    // ---- 2. CPU reference ICP (timed) --------------------------------------
    std::vector<double> hist_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    const Rigid g_cpu = icp_cpu(c, ICP_ITERS, hist_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU ICP (kernels timed inside the driver) ----------------------
    std::vector<double> hist_gpu;
    float gpu_kernel_ms = 0.0f;
    const Rigid g_gpu = icp_gpu(c.P, c.Q, ICP_ITERS, hist_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ----------------------------------------------------------
    // Max absolute difference between the CPU and GPU transforms (R entries + t).
    double transform_diff = 0.0;
    for (int r = 0; r < 3; ++r) {
        for (int col = 0; col < 3; ++col)
            transform_diff = std::fmax(transform_diff, std::fabs(g_cpu.R[r][col] - g_gpu.R[r][col]));
        transform_diff = std::fmax(transform_diff, std::fabs(g_cpu.t[r] - g_gpu.t[r]));
    }
    // Max difference between the two convergence curves.
    double history_diff = 0.0;
    for (int i = 0; i < ICP_ITERS; ++i)
        history_diff = std::fmax(history_diff, std::fabs(hist_cpu[i] - hist_gpu[i]));
    const bool pass = (transform_diff <= TRANSFORM_TOL) && (history_diff <= HISTORY_TOL);

    // Final alignment quality (from the GPU transform; identical to the CPU's).
    const double final_rms = rms_error(c.P, c.Q, g_gpu);

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("ICP rigid registration: %zu moving pts -> %zu fixed pts, %d iterations\n",
                c.P.size(), c.Q.size(), ICP_ITERS);
    std::printf("RMS alignment error per iteration (mm):\n");
    for (int i = 0; i < ICP_ITERS; ++i)
        std::printf("  iter %2d: %10.6f\n", i + 1, hist_gpu[i]);
    print_transform("recovered transform [R | t] (maps pre-op onto intra-op)", g_gpu);
    std::printf("final RMS error = %.6f mm\n", final_rms);
    std::printf("RESULT: %s (GPU transform matches CPU reference)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%zu moving, %zu fixed points)\n",
                 source, c.P.size(), c.Q.size());
    std::fprintf(stderr, "[timing] CPU ICP: %.3f ms   GPU kernels (sum over %d iters): %.3f ms\n",
                 cpu_ms, ICP_ITERS, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- on this tiny cloud the GPU is "
                         "launch-bound; the O(|P||Q|) search wins as clouds grow to 10^4-10^6 pts.\n");
    std::fprintf(stderr, "[verify] max transform diff = %.3e (tol %.1e), max history diff = %.3e (tol %.1e)\n",
                 transform_diff, TRANSFORM_TOL, history_diff, HISTORY_TOL);

    return pass ? 0 : 1;
}
