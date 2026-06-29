// ===========================================================================
// src/main.cu  --  Entry point: load problem, dock on CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 1.3 : Molecular Docking Engine  (reduced-scope teaching version)
//
// THE 5-STEP SHAPE EVERY PROJECT IN THIS REPO FOLLOWS
//   1. Load the problem (energy grid + ligand + pose search space from
//      data/sample, via load_problem in reference_cpu.cpp).
//   2. CPU reference (dock_cpu)  -> trusted best pose (index + energy).
//   3. GPU dock      (dock_gpu)  -> the thing being taught.
//   4. VERIFY: the GPU's winning pose INDEX matches the CPU's, and the energies
//      agree within tolerance.
//   5. REPORT: deterministic best-pose summary to stdout; timing to stderr.
//
//   STDOUT is byte-for-byte deterministic (diffed by demo/run_demo against
//   demo/expected_output.txt); run-to-run timings go to STDERR (shown, not diffed).
//
// CODE TOUR: start here, then docking_core.h (the physics), kernels.cuh ->
//   kernels.cu (the GPU path), and reference_cpu.cpp (the baseline + loader).
//   See ../THEORY.md for the science and the GPU mapping.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>

#include "kernels.cuh"        // dock_gpu (GPU path); pulls in reference_cpu.h
#include "reference_cpu.h"    // load_problem, dock_cpu, unrank_pose, data model
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "1.3";
static const char* PROJECT_NAME = "Molecular Docking Engine";

// ---------------------------------------------------------------------------
// Verification tolerance on the BEST-POSE ENERGY.
//   The headline correctness check is that the GPU and CPU pick the SAME pose
//   index (an exact integer match -- the reduction is deterministic by design,
//   PATTERNS.md S3). As a second check we compare the energies: both come from
//   the identical docking_core.h::score_pose on the identical pose, so they are
//   bit-identical in principle; 1e-9 is a generous margin that also guards the
//   degenerate "different index, equal energy" tie case. (PATTERNS.md S4: exact
//   integer key for the winner, tiny tolerance for the floating-point energy.)
// ---------------------------------------------------------------------------
static constexpr double TOLERANCE = 1.0e-9;

int main(int argc, char** argv) {
    // ---- 1. Load the problem -----------------------------------------------
    const std::string path = (argc > 1) ? argv[1]
                                        : "data/sample/receptor_ligand_sample.txt";
    DockingProblem prob;
    try {
        prob = load_problem(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const long long n_poses = prob.space.n_poses();

    // ---- 2. CPU reference (timed) ------------------------------------------
    double    cpu_energy = 0.0;
    long long cpu_index  = 0;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    dock_cpu(prob, &cpu_energy, &cpu_index);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU dock (kernel timed inside the wrapper) ---------------------
    double    gpu_energy = 0.0;
    long long gpu_index  = 0;
    float     gpu_kernel_ms = 0.0f;
    dock_gpu(prob, &gpu_energy, &gpu_index, &gpu_kernel_ms);

    // ---- 4. Verify ----------------------------------------------------------
    // Primary: same winning pose (exact integer). Secondary: energies agree.
    const bool index_match = (gpu_index == cpu_index);
    const double energy_err = std::fabs(gpu_energy - cpu_energy);
    const bool pass = index_match && (energy_err <= TOLERANCE);

    // Decode the winning pose so we can print its concrete parameters. Both sides
    // agree, so we report the GPU's (== CPU's) winner.
    const Pose best = unrank_pose(prob.space, gpu_index);

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("Rigid-body docking: scored %lld poses "
                "(%d^3 translations x %d^3 rotations) over a %dx%dx%d energy grid\n",
                n_poses, prob.space.n_trans, prob.space.n_rot,
                prob.dims.nx, prob.dims.ny, prob.dims.nz);
    std::printf("ligand atoms: %d\n", prob.ligand.n_atoms);
    std::printf("best pose index: %lld\n", gpu_index);
    std::printf("best pose translation (A): tx=%.4f ty=%.4f tz=%.4f\n",
                best.tx, best.ty, best.tz);
    std::printf("best pose rotation (rad):  a=%.4f b=%.4f c=%.4f\n",
                best.a, best.b, best.c);
    std::printf("best score (kcal/mol): %.6f\n", gpu_energy);
    std::printf("RESULT: %s (GPU best pose matches CPU; energies agree within tol=1.0e-09)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (grid %dx%dx%d, spacing %.3f A, %lld poses)\n",
                 path.c_str(), prob.dims.nx, prob.dims.ny, prob.dims.nz,
                 prob.dims.spacing, n_poses);
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny sample is dominated by "
                         "launch/copy overhead; the GPU wins at campaign scale (millions of "
                         "poses x millions of ligands).\n");
    std::fprintf(stderr, "[verify] index match: %s   CPU idx=%lld  GPU idx=%lld   "
                         "energy_err=%.3e (tol %.1e)\n",
                 index_match ? "yes" : "NO", cpu_index, gpu_index, energy_err, TOLERANCE);

    return pass ? 0 : 1;
}
