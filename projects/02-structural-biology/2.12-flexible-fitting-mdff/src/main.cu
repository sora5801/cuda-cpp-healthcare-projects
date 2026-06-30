// ===========================================================================
// src/main.cu  --  Entry point: load problem, fit on CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 2.12 : Flexible Fitting / MDFF
//
// WHAT THIS FILE DOES  (the 5-step shape every project in this repo follows)
//   1. Load the problem (data/sample/*.txt, or a built-in synthetic fallback):
//      a cryo-EM-style density map + an atomic model misfitted from its target.
//   2. Fit on the CPU  (reference_cpu.cpp)  -> trusted baseline positions.
//   3. Fit on the GPU  (kernels.cu)          -> the thing being taught.
//   4. VERIFY: assert the GPU fitted positions match the CPU within a tolerance.
//   5. REPORT: deterministic fit quality (RMSD-to-target, cross-correlation,
//      sample atom positions) to stdout; timing to stderr.
//
//   STDOUT is kept byte-for-byte deterministic so demo/run_demo can diff it
//   against demo/expected_output.txt. Anything that varies run-to-run (timings)
//   goes to STDERR, which the demo shows but does not diff.
//
//   The headline story for the learner: the model starts MISFITTED (high
//   RMSD-to-target, low cross-correlation) and the density-derived force pulls it
//   onto the map (RMSD drops, cross-correlation rises), with the GPU and CPU
//   reaching the same answer.
//
// Code tour: start here, then mdff.h (the physics + trilinear sampler),
//   kernels.cu (the GPU twin), reference_cpu.cpp (the baseline). See ../THEORY.md.
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // fit_gpu, MdffParams, Vec3
#include "reference_cpu.h"    // load_problem, make_synthetic, fit_cpu, metrics
#include "util/io.hpp"        // util::CpuTimer

// Self-identification tokens (kept in sync with demo/expected_output.txt).
static const char* PROJECT_ID   = "2.12";
static const char* PROJECT_NAME = "Flexible Fitting / MDFF";

// Verification tolerance on the final atom positions.
//   The fit is a 200-iteration double-precision steepest descent. Over that many
//   iterations the GPU's fused-multiply-add (FMA) contraction and the host
//   compiler's diverge at the ~1e-6 level even though the math is identical (the
//   same lesson taught in projects 10.02 and 14.02; see THEORY "Numerical
//   considerations"). We therefore verify GPU==CPU to a physically-negligible
//   1e-4 on positions of magnitude ~10 -- NOT to bit-exactness, and we say so.
static constexpr double TOLERANCE = 1.0e-4;

// Largest per-atom position discrepancy between two atom sets (our GPU-vs-CPU
// agreement metric). Returns +inf on a length mismatch so a shape bug cannot be
// mistaken for agreement.
static double worst_atom_diff(const std::vector<Vec3>& a, const std::vector<Vec3>& b) {
    if (a.size() != b.size()) return 1e300;
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        worst = std::fmax(worst, length(a[i] - b[i]));
    return worst;
}

int main(int argc, char** argv) {
    // ---- 1. Load the problem -----------------------------------------------
    MdffProblem prob;
    const char* source = "synthetic (built-in)";
    if (argc > 1) {
        try {
            prob = load_problem(argv[1]);
            source = argv[1];
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[error] %s\n", e.what());
            return 2;
        }
    } else {
        prob = make_synthetic();
    }
    const MdffParams& P = prob.params;

    // ---- 2. CPU reference fit (timed) --------------------------------------
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    std::vector<Vec3> x_cpu = fit_cpu(prob);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU fit (kernel loop timed inside the wrapper) -----------------
    std::vector<Vec3> x_gpu;
    float gpu_kernel_ms = 0.0f;
    fit_gpu(P, prob.rho, prob.x0, prob.x_ref, x_gpu, &gpu_kernel_ms);

    // ---- 4. Verify (GPU fitted positions match the CPU baseline) -----------
    const double worst = worst_atom_diff(x_cpu, x_gpu);
    const bool pass = worst <= TOLERANCE;

    // Quality metrics: before vs after, scored against the GROUND-TRUTH targets
    // and via the density cross-correlation. (Computed on the GPU result; the
    // CPU result is identical to within tolerance.)
    const double rmsd0 = rmsd(prob.x0,  prob.x_target);   // start RMSD-to-target
    const double rmsd1 = rmsd(x_gpu,    prob.x_target);   // final RMSD-to-target
    const double cc0   = cross_correlation(prob.x0,  prob);
    const double cc1   = cross_correlation(x_gpu,    prob);

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("density map: %dx%dx%d voxels @ %.2f u/voxel | atoms: %d | "
                "iters: %d\n", P.nx, P.ny, P.nz, P.vox, P.natoms, P.iters);
    std::printf("weights: w_dens=%.2f  k_rest=%.2f  step=%.3f\n",
                P.w_dens, P.k_rest, P.step);
    std::printf("fit quality (vs ground-truth target):\n");
    std::printf("  RMSD start -> final : %.6f -> %.6f\n", rmsd0, rmsd1);
    std::printf("  cross-corr start -> final : %.6f -> %.6f\n", cc0, cc1);
    // Show a few representative atoms so the learner sees concrete coordinates.
    const int show = P.natoms < 4 ? P.natoms : 4;
    for (int i = 0; i < show; ++i) {
        const int idx = (P.natoms <= 1) ? 0 : i * (P.natoms - 1) / (show - 1);
        std::printf("  atom %2d fitted = (%.6f, %.6f, %.6f)\n",
                    idx, x_gpu[idx].x, x_gpu[idx].y, x_gpu[idx].z);
    }
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-04)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%d atoms, %lld voxels)\n",
                 source, P.natoms,
                 static_cast<long long>(prob.rho.size()));
    std::fprintf(stderr, "[timing] CPU fit: %.3f ms   GPU fit: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- with tens of atoms "
                         "this is launch-bound; the GPU's edge appears at the "
                         "10^5-10^6 atoms of real ribosome/capsid fits.\n");
    std::fprintf(stderr, "[verify] worst per-atom GPU-vs-CPU diff = %.3e  "
                         "(tolerance %.1e)\n", worst, TOLERANCE);

    // Exit code feeds the demo's pass/fail gate.
    return pass ? 0 : 1;
}
