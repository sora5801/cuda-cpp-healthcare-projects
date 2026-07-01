// ===========================================================================
// src/main.cu  --  Entry point: load ASL study, fit CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 4.23 : Arterial Spin Labeling & Perfusion Imaging
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the multi-delay ASL study (data/sample/asl_sample.txt).
//   2. CPU reference: fit every voxel serially (reference_cpu.cpp) -> trusted.
//   3. GPU: one thread per voxel, same Gauss-Newton solver (kernels.cu).
//   4. VERIFY: assert the GPU per-voxel (CBF, ATT) match the CPU to round-off,
//      AND (a science check) that the fit RECOVERS the known ground-truth
//      physiology used to synthesize the noise-free curves.
//   5. REPORT: deterministic per-voxel + summary result to stdout; timing stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Anything that varies run-to-run (timings) goes to
//   STDERR, which the demo shows but does not diff (docs/PATTERNS.md §3).
//
// READ THIS FIRST in the code tour, then asl.h (the model + fit), kernels.cuh ->
// kernels.cu (the GPU mapping), reference_cpu.cpp (the baseline + loader).
// See ../THEORY.md for the "why".
// ===========================================================================
#include <cmath>      // std::fabs, std::fmax
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // fit_gpu (GPU path)
#include "reference_cpu.h"    // HostDataset, load_asl, fit_cpu (CPU baseline)
#include "util/io.hpp"        // util::CpuTimer

// Program identity (matches demo/expected_output.txt).
static const char* PROJECT_ID   = "4.23";
static const char* PROJECT_NAME = "Arterial Spin Labeling & Perfusion Imaging";

// Correctness tolerances (docs/PATTERNS.md §4):
//  * GPU-vs-CPU: both sides run the IDENTICAL double-precision asl_fit_voxel(),
//    so they should agree to a few ULPs; 1e-9 is a safe round-off ceiling. (The
//    only permitted divergence is FMA-contraction differences between the host
//    and device compilers, which are ~1e-12 over this short computation.)
static constexpr double TOL_GPU_CPU = 1.0e-9;
//  * Ground-truth recovery: the sample curves are NOISE-FREE, so a converged
//    Gauss-Newton fit should recover the true CBF/ATT to near round-off too. We
//    allow a slightly looser 1e-4 to absorb the tiny residual of a finite
//    iteration cap -- still a strong "the science works" check, not a fudge.
static constexpr double TOL_RECOVER = 1.0e-4;

int main(int argc, char** argv) {
    // ---- 1. Load the ASL study ---------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/asl_sample.txt";
    HostDataset ds;
    try {
        ds = load_asl(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) ------------------------------------------
    std::vector<AslFit> fit_cpu_res;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    fit_cpu(ds, fit_cpu_res);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU fit (kernel timed inside the wrapper) ----------------------
    std::vector<AslFit> fit_gpu_res;
    float gpu_kernel_ms = 0.0f;
    fit_gpu(ds, fit_gpu_res, &gpu_kernel_ms);

    // ---- 4. Verify ----------------------------------------------------------
    // (a) GPU vs CPU: worst per-voxel disagreement in CBF and ATT.
    double worst_gpu_cpu = 0.0;
    // (b) recovery: worst |fit - truth| across voxels (the science check).
    double worst_recover = 0.0;
    for (int v = 0; v < ds.n_voxels; ++v) {
        worst_gpu_cpu = std::fmax(worst_gpu_cpu,
                                  std::fabs(fit_cpu_res[v].cbf - fit_gpu_res[v].cbf));
        worst_gpu_cpu = std::fmax(worst_gpu_cpu,
                                  std::fabs(fit_cpu_res[v].att - fit_gpu_res[v].att));
        worst_recover = std::fmax(worst_recover,
                                  std::fabs(fit_gpu_res[v].cbf - ds.true_cbf[v]));
        worst_recover = std::fmax(worst_recover,
                                  std::fabs(fit_gpu_res[v].att - ds.true_att[v]));
    }
    const bool pass_gpu_cpu = worst_gpu_cpu <= TOL_GPU_CPU;
    const bool pass_recover = worst_recover <= TOL_RECOVER;
    const bool pass = pass_gpu_cpu && pass_recover;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("multi-delay ASL Buxton fit: %d voxels x %d PLDs, Levenberg-Marquardt (<=%d it)\n",
                ds.n_voxels, ds.n_plds, ds.max_iters);
    std::printf("PLDs (s):");
    for (int j = 0; j < ds.n_plds; ++j) std::printf(" %.2f", ds.pld[j]);
    std::printf("\n");
    std::printf("per-voxel fit (true -> recovered):\n");
    std::printf("  vox   CBF_true CBF_fit   ATT_true ATT_fit   iters\n");
    for (int v = 0; v < ds.n_voxels; ++v) {
        // %.3f keeps the printed digits deterministic (independent of tiny ULP
        // noise below the 4th decimal) so stdout is byte-stable across machines.
        std::printf("  v%-3d  %8.3f %7.3f   %8.3f %7.3f   %5d\n",
                    v, ds.true_cbf[v], fit_gpu_res[v].cbf,
                    ds.true_att[v], fit_gpu_res[v].att, fit_gpu_res[v].iters);
    }
    // Population-style summary (mean recovered CBF/ATT) -- the kind of number an
    // ASL perfusion map is ultimately reduced to.
    double mean_cbf = 0.0, mean_att = 0.0;
    for (int v = 0; v < ds.n_voxels; ++v) {
        mean_cbf += fit_gpu_res[v].cbf;
        mean_att += fit_gpu_res[v].att;
    }
    mean_cbf /= ds.n_voxels; mean_att /= ds.n_voxels;
    std::printf("mean recovered: CBF = %.3f mL/100g/min   ATT = %.3f s\n",
                mean_cbf, mean_att);
    std::printf("RESULT: %s (GPU==CPU within 1e-09; fit recovers ground truth within 1e-04)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%d voxels, %d PLDs)\n",
                 path.c_str(), ds.n_voxels, ds.n_plds);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernel: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- with a few voxels the GPU is "
                         "launch-bound; the win grows toward whole-brain (~10^5-10^6 voxels).\n");
    std::fprintf(stderr, "[verify] worst |GPU-CPU| = %.3e (tol %.1e)   worst |fit-truth| = %.3e (tol %.1e)\n",
                 worst_gpu_cpu, TOL_GPU_CPU, worst_recover, TOL_RECOVER);

    // Exit code feeds the demo's pass/fail gate.
    return pass ? 0 : 1;
}
