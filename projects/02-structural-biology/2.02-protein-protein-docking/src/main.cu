// ===========================================================================
// src/main.cu  --  Entry point: load proteins, dock CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 2.2 : Protein-Protein Docking  (catalog id 2.2)
//
// WHAT THIS FILE DOES  (the 5-step shape EVERY project in this repo follows)
//   1. Load two proteins (receptor + ligand) from data/sample.
//   2. Voxelize both onto a shared 3D grid (the Katchalski-Katzir shape model).
//   3. Score every rigid translation t by cross-correlation S(t), TWO ways:
//        CPU: brute-force O(Ng^2) direct sum   (reference_cpu.cpp) -> trusted.
//        GPU: cuFFT O(Ng log Ng) correlation    (kernels.cu)        -> taught.
//   4. VERIFY: the GPU score grid matches the CPU one within tolerance AND the
//      best-scoring translation (the predicted docking pose) is identical. When
//      the sample carries a known answer, also check we recovered it (science).
//   5. REPORT: deterministic best pose + score to stdout; timings to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Run-varying numbers (timings) go to STDERR, which
//   the demo shows but does not diff.
//
// READ THIS FIRST in the code tour, then kernels.cuh -> kernels.cu (the cuFFT
// correlation), then reference_cpu.cpp (the baseline). See ../THEORY.md for why.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // dock_gpu (cuFFT correlation)
#include "reference_cpu.h"    // load_dock, voxelize_*, correlate_cpu, argmax_grid
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "2.2";
static const char* PROJECT_NAME = "Protein-Protein Docking";

// Verification tolerance (ABSOLUTE, on the correlation score). The CPU computes
// S(t) as an exact sum of small-integer products; cuFFT works in SINGLE
// precision and its forward+inverse round-trip accumulates round-off that grows
// with the grid size Ng and the score magnitude. For the committed 32^3 sample
// the peak score is ~7.7e4, and the worst voxel diff we observe is ~5e-2 -- a
// RELATIVE error of ~6e-7, i.e. right at float epsilon scaled by the transform
// size. We set the absolute floor to 0.5 (still ~1e-5 relative, utterly
// negligible next to the integer-spaced scores) so the check is robust across
// machines without hiding a real bug. The ARGMAX (the actual docking answer) is
// far more robust than any single voxel and matches the CPU EXACTLY -- that is
// the result we truly care about. (See THEORY section 5 "Numerical considerations".)
static constexpr double SCORE_TOL = 0.5;

int main(int argc, char** argv) {
    // ---- 1. Load the two proteins ------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/dock_sample.txt";
    DockData d;
    try {
        d = load_dock(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. Voxelize both onto the shared grid -----------------------------
    // The receptor defines the frame (origin); the ligand uses the same origin
    // so a translation in voxels maps the ligand directly onto the receptor.
    std::vector<float> R, L;
    double origin[3];
    voxelize_receptor(d, R, origin);
    voxelize_ligand(d, origin, L);

    // ---- 3a. CPU reference correlation (timed) -----------------------------
    std::vector<float> score_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    correlate_cpu(d.N, R, L, score_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3b. GPU cuFFT correlation (kernel timed inside the wrapper) -------
    std::vector<float> score_gpu;
    float gpu_kernel_ms = 0.0f;
    dock_gpu(d.N, R, L, score_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ----------------------------------------------------------
    // (a) grid agreement: largest absolute score difference over all voxels.
    double worst = 0.0;
    for (std::size_t i = 0; i < score_cpu.size(); ++i) {
        double diff = std::fabs(static_cast<double>(score_cpu[i]) -
                                static_cast<double>(score_gpu[i]));
        if (diff > worst) worst = diff;
    }
    const bool grid_ok = (worst <= SCORE_TOL);

    // (b) the headline answer: the best-scoring translation must be identical.
    //     argmax returns the voxel INDEX in [0, N); a translation t and t+N are
    //     the same circular shift, so we also map indices > N/2 to negative
    //     values for a human-readable signed translation (e.g. index N-1 -> -1).
    int cx, cy, cz, gx, gy, gz;
    argmax_grid(d.N, score_cpu, cx, cy, cz);
    argmax_grid(d.N, score_gpu, gx, gy, gz);
    const bool pose_ok = (cx == gx && cy == gy && cz == gz);
    auto to_signed = [N = d.N](int i) { return (i <= N / 2) ? i : i - N; };
    const int sgx = to_signed(gx), sgy = to_signed(gy), sgz = to_signed(gz);

    // (c) science check (only when the sample carries the known answer): did we
    //     recover the translation the synthetic complex was built with? The known
    //     T is signed (e.g. -1); compare it to the recovered SIGNED translation.
    const bool have_truth = (d.true_tx != DockData::NO_TRUTH);   // sample carries a known answer?
    const bool truth_ok = !have_truth ||
        (sgx == d.true_tx && sgy == d.true_ty && sgz == d.true_tz);

    const bool pass = grid_ok && pose_ok && truth_ok;
    // Report the score from the CPU reference at the winning voxel: it is the
    // exact integer-valued sum (the GPU's single-precision FFT value differs by
    // ~1e-5 relative and could vary by GPU arch). Printing the exact CPU value
    // keeps stdout byte-identical across machines for the demo diff.
    const float best_score = score_cpu[static_cast<std::size_t>(flat3(gx, gy, gz, d.N))];

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("FFT rigid-body docking (Katchalski-Katzir shape correlation via cuFFT)\n");
    std::printf("grid: %dx%dx%d voxels @ %.2f A/voxel   receptor atoms: %d   ligand atoms: %d\n",
                d.N, d.N, d.N, d.spacing, d.n_recv, d.n_lig);
    std::printf("best translation (voxels): t = (%d, %d, %d)\n", sgx, sgy, sgz);
    std::printf("best shape-complementarity score: %.4f\n", best_score);
    if (have_truth)
        std::printf("known-answer translation:  t = (%d, %d, %d)  -> %s\n",
                    d.true_tx, d.true_ty, d.true_tz, truth_ok ? "RECOVERED" : "MISSED");
    std::printf("RESULT: %s (cuFFT score grid matches CPU within %.0e; best pose identical)\n",
                pass ? "PASS" : "FAIL", SCORE_TOL);

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (N=%d, %d+%d atoms)\n",
                 path.c_str(), d.N, d.n_recv, d.n_lig);
    std::fprintf(stderr, "[timing] CPU brute-force O(Ng^2): %.3f ms   GPU cuFFT O(Ng log Ng): %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- the brute-force correlation is O(Ng^2); the "
                         "FFT route is O(Ng log Ng). The gap explodes with grid size.\n");
    std::fprintf(stderr, "[verify] worst |score_cpu - score_gpu| = %.6e (tol %.1e)  grid=%s pose=%s\n",
                 worst, SCORE_TOL, grid_ok ? "ok" : "FAIL", pose_ok ? "ok" : "FAIL");

    return pass ? 0 : 1;
}
