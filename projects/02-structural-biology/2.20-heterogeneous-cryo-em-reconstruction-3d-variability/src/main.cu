// ===========================================================================
// src/main.cu  --  Entry point: load volumes, run 3DVA on CPU + GPU, verify
// ---------------------------------------------------------------------------
// Project 2.20 : Heterogeneous Cryo-EM Reconstruction (3D Variability)
//
// WHAT THIS FILE DOES  (the 5-step shape EVERY project in this repo follows)
//   1. Load the N particle volumes (data/sample) + their ground-truth coords.
//   2. CPU reference 3DVA: mean, Gram matrix, Jacobi eigendecomp, PC1, latents.
//   3. GPU 3DVA: the same pipeline via per-element kernels + cuSOLVER.
//   4. VERIFY: GPU eigenvalues / PC1 / latent coords match the CPU within tol.
//   5. REPORT: deterministic 3DVA result to stdout; timing to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Run-to-run timings go to STDERR (shown, not diffed).
//
//   THE SCIENCE PAYOFF: the synthetic dataset hides a known continuous motion
//   (a density blob sliding along z). We report how much of the variance PC1
//   captures and how well the recovered latent coordinate z[p] correlates with
//   the hidden ground-truth coordinate -- i.e. did 3DVA recover the motion?
//
// Code tour: start HERE, then reference_cpu.h (data model + shared math),
//   reference_cpu.cpp (CPU 3DVA + Jacobi), kernels.cuh/.cu (GPU + cuSOLVER).
//   The "why" is in ../THEORY.md.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <limits>     // std::numeric_limits (used by max_abs_diff)
#include <string>
#include <vector>

#include "kernels.cuh"        // run_3dva_gpu (GPU path), GpuTimings
#include "reference_cpu.h"    // VolumeSet, CPU 3DVA, shared HD math
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "2.20";
static const char* PROJECT_NAME = "Heterogeneous Cryo-EM Reconstruction (3D Variability)";

// Verification tolerances. The Gram matrix and projections are short
// double-precision computations using the SAME shared math on both sides, so we
// expect agreement near machine precision. cuSOLVER's divide-and-conquer
// eigensolver and our Jacobi sweep are DIFFERENT algorithms, so their
// eigenvalues agree only to a few ulps -- 1e-9 is a safe, honest floor
// (PATTERNS.md §4: ~machine precision for short double-precision work).
static constexpr double TOL_EIG = 1.0e-9;   // eigenvalues (different algorithms)
static constexpr double TOL_VEC = 1.0e-9;   // PC1 voxels + latent coordinates

// pearson: correlation between two equal-length series, in [-1, 1].
//   We use it to ask "does the recovered latent coordinate z track the hidden
//   ground-truth conformational coordinate?" A magnitude near 1.0 means 3DVA
//   recovered the motion (sign is arbitrary -- PCs are defined up to sign).
static double pearson(const std::vector<double>& a, const std::vector<double>& b) {
    const int n = (int)a.size();
    double ma = 0.0, mb = 0.0;
    for (int i = 0; i < n; ++i) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    double sab = 0.0, saa = 0.0, sbb = 0.0;
    for (int i = 0; i < n; ++i) {
        const double da = a[i] - ma, db = b[i] - mb;
        sab += da * db; saa += da * da; sbb += db * db;
    }
    const double den = std::sqrt(saa * sbb);
    return (den > 0.0) ? (sab / den) : 0.0;
}

// max_abs_diff: largest |a[i]-b[i]| over two equal-length double arrays. Our
//   headline GPU-vs-CPU agreement metric (returns +inf on a length mismatch).
static double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        worst = std::fmax(worst, std::fabs(a[i] - b[i]));
    return worst;
}

int main(int argc, char** argv) {
    // ---- 1. Load the particle volumes -------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/volumes.txt";
    VolumeSet vs;
    try {
        vs = load_volumes(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int N = vs.N, D = vs.D;

    // ---- 2. CPU reference 3DVA (timed) ------------------------------------
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    std::vector<double> mean_cpu;
    compute_mean(vs, mean_cpu);
    std::vector<double> gram_cpu;
    build_gram_cpu(vs, mean_cpu, gram_cpu);
    std::vector<double> eval_cpu, gevec_cpu;
    jacobi_eigen_symmetric(gram_cpu, N, eval_cpu, gevec_cpu);
    std::vector<double> pc1_cpu;
    lift_to_volume_pc(vs, mean_cpu, gevec_cpu, N - 1, pc1_cpu);  // last col = largest eigenvalue
    std::vector<double> z_cpu;
    project_all_cpu(vs, mean_cpu, pc1_cpu, z_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // Variance explained by PC1 on the CPU side (for the report + a cross-check).
    double total_cpu = 0.0;
    for (int i = 0; i < N; ++i) total_cpu += (eval_cpu[i] > 0.0 ? eval_cpu[i] : 0.0);
    const double var_pc1_cpu = (total_cpu > 0.0) ? (eval_cpu[N - 1] / total_cpu) : 0.0;

    // ---- 3. GPU 3DVA (per-stage GPU timings inside the wrapper) -----------
    std::vector<double> mean_gpu, eval_gpu, pc1_gpu, z_gpu;
    double var_pc1_gpu = 0.0;
    GpuTimings gt;
    run_3dva_gpu(vs, mean_gpu, eval_gpu, pc1_gpu, z_gpu, var_pc1_gpu, gt);

    // ---- 4. Verify GPU vs CPU ---------------------------------------------
    const double err_eig = max_abs_diff(eval_cpu, eval_gpu);   // eigenvalues
    const double err_pc1 = max_abs_diff(pc1_cpu, pc1_gpu);     // PC1 voxels
    const double err_z   = max_abs_diff(z_cpu, z_gpu);         // latent coords
    const bool pass = (err_eig <= TOL_EIG) && (err_pc1 <= TOL_VEC) && (err_z <= TOL_VEC);

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) ----------
    // Correlation of the recovered latent with the hidden ground truth: did we
    // recover the motion? Reported as |corr| since a PC's sign is arbitrary.
    const double corr = std::fabs(pearson(z_gpu, vs.truth));

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("3DVA (PCA on volumes): N=%d particles, G=%d, D=%d voxels/volume\n", N, vs.G, D);
    std::printf("top-3 mode variances (eigenvalues, descending):");
    for (int m = 0; m < 3 && m < N; ++m) std::printf(" %.5f", eval_gpu[N - 1 - m]);
    std::printf("\n");
    std::printf("PC1 variance explained: %.4f\n", var_pc1_gpu);
    std::printf("latent z along PC1 (8 sampled particles):");
    for (int s = 0; s < 8 && N > 0; ++s) {
        const int p = (N > 1) ? (s * (N - 1)) / 7 : 0;
        std::printf(" %+.4f", z_gpu[p]);
    }
    std::printf("\n");
    std::printf("PC1 vs ground-truth conformation |corr|: %.4f\n", corr);
    std::printf("RESULT: %s (GPU 3DVA matches CPU reference within tol)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) -----------------
    std::fprintf(stderr, "[data]   source: %s  (N=%d, G=%d, D=%d)\n", path.c_str(), N, vs.G, D);
    std::fprintf(stderr, "[timing] CPU reference (full 3DVA): %.3f ms\n", cpu_ms);
    std::fprintf(stderr, "[timing] GPU stages -- mean: %.3f  gram: %.3f  eigen(cuSOLVER): %.3f  "
                         "lift: %.3f  project: %.3f ms\n",
                 gt.mean_ms, gt.gram_ms, gt.eigen_ms, gt.lift_ms, gt.proj_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- the tiny sample is launch-bound; the "
                         "GPU's edge grows with N (the Gram step) and D (the lift step).\n");
    std::fprintf(stderr, "[verify] max|dEigenvalue|=%.3e (tol %.1e)  max|dPC1|=%.3e (tol %.1e)  "
                         "max|dLatent|=%.3e (tol %.1e)\n",
                 err_eig, TOL_EIG, err_pc1, TOL_VEC, err_z, TOL_VEC);
    std::fprintf(stderr, "[verify] CPU PC1 variance explained = %.4f (cross-check vs GPU %.4f)\n",
                 var_pc1_cpu, var_pc1_gpu);

    return pass ? 0 : 1;
}
