// ===========================================================================
// src/main.cu  --  Entry point: load problem, dock on CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 1.28 : Covalent Docking
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the docking problem (data/sample, or a built-in synthetic fallback).
//   2. CPU reference: score every torsion-grid conformation serially.
//   3. GPU: score every conformation, one thread each (kernels.cu).
//   4. VERIFY: the GPU energy array matches the CPU array (same math -> exact),
//      then both argmin to the docked pose.
//   5. REPORT: the deterministic docked pose + a few diagnostics to stdout;
//      timing to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Anything that varies run-to-run (timings) goes to
//   STDERR, which the demo shows but does not diff.
//
//   NOT FOR CLINICAL USE -- synthetic, didactic geometry and force field.
//
// READ THIS FIRST in the code tour, then docking.h (the physics), kernels.cuh ->
// kernels.cu, and reference_cpu.cpp for the baseline. See ../THEORY.md.
// ===========================================================================
#include <cmath>      // std::fabs
#include <cstdio>
#include <string>
#include <vector>

#include "docking.h"          // DockProblem, build_conformation, score helpers
#include "kernels.cuh"        // score_all_gpu (GPU path)
#include "reference_cpu.h"    // load_problem, score_all_cpu, argmin_energy (CPU)
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "1.28";
static const char* PROJECT_NAME = "Covalent Docking";

// Correctness tolerance. The CPU and GPU run the SAME double-precision operations
// in the SAME order (the shared score_conformation() in docking.h), so the only
// possible difference is the GPU's fused-multiply-add (FMA) contracting a*b+c
// into one rounding step where the host does two. That perturbs energies by, at
// most, a few ULP (~1e-9 kcal/mol on these ~10 kcal/mol numbers). 1e-6 is a
// comfortable, honest margin above that. (docs/PATTERNS.md section 4.)
static constexpr double TOLERANCE = 1.0e-6;

// ---------------------------------------------------------------------------
// max_abs_err_d: largest |a[i]-b[i]| over two equal-length double arrays. The
// headline correctness metric; returns +inf on a length mismatch so a shape bug
// cannot masquerade as agreement. (util::max_abs_err is float-only; docking is
// done in double, so we provide the double version here.)
// ---------------------------------------------------------------------------
static double max_abs_err_d(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return HUGE_VAL;
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        worst = std::fmax(worst, std::fabs(a[i] - b[i]));
    return worst;
}

// ---------------------------------------------------------------------------
// built_in_problem: the synthetic docking problem used when no data file is
// supplied. It mirrors the committed sample (data/sample/covalent_sample.txt)
// EXACTLY so the program prints the same result with or without the file -- the
// numbers below and the file are two copies of the same synthetic system (a
// warhead bonded to a cysteine sulfur, a 3-torsion ligand chain, a 6-atom
// pocket). See data/README.md for the field meanings.
// ---------------------------------------------------------------------------
static DockProblem built_in_problem() {
    DockProblem p{};
    p.anchor = Vec3{0.0, 0.0, 0.0};        // warhead carbon at the origin
    p.sg     = Vec3{1.81, 0.0, 0.0};       // cysteine S-gamma 1.81 A away (ideal)
    p.bond_len_ideal = 1.81;               // ideal C-S covalent length, A
    p.angle_ideal    = 1.9106;             // ~109.47 deg (sp3), in radians
    p.k_bond         = 300.0;              // stiff bond-length spring
    p.k_angle        = 100.0;              // bond-angle spring
    p.seg_len        = 1.50;               // ligand C-C bond length, A
    p.bond_angle     = 1.9106;             // sp3 valence angle between segments
    // first_dir makes the S-gamma -- anchor -- first-atom angle exactly 109.47
    // deg (tetrahedral): first_dir . (+x toward sulfur) = -1/3. This zeroes the
    // covalent ANGLE penalty so the SCORE that varies is the ligand-pocket fit.
    p.first_dir      = Vec3{-1.0/3.0, 0.94280904158206336, 0.0};
    p.lig_sigma      = 3.40;               // ligand-atom LJ sigma (carbon-like), A
    p.lig_epsilon    = 0.10;               // ligand-atom LJ epsilon, kcal/mol
    p.lig_charge     = -0.10;              // ligand-atom partial charge, e
    // Six fixed pocket atoms, each placed ~3.82 A (the LJ energy minimum,
    // 2^(1/6)*sigma) from the closest point the flexible ligand tip can reach.
    // So every pocket atom is ATTRACTIVE-but-never-clashing: the landscape is
    // smooth (no r^-12 blow-up) with a clear, deep, negative minimum. Positions
    // come from scanning the ligand's reachable shell (see scripts/make_synthetic
    // .py and THEORY.md). pos(x,y,z), sigma, epsilon, charge.
    const PocketAtom pk[N_POCKET] = {
        {Vec3{ 0.382, -3.497,  4.476}, 3.40, 0.20,  0.10},
        {Vec3{ 5.194,  6.384,  0.686}, 3.40, 0.20, -0.10},
        {Vec3{-3.293, -0.819,  6.642}, 3.40, 0.20,  0.10},
        {Vec3{ 0.170,  0.809, -7.523}, 3.40, 0.20, -0.10},
        {Vec3{-7.918,  1.226, -1.574}, 3.40, 0.20,  0.05},
        {Vec3{-2.702,  7.121, -4.190}, 3.40, 0.20, -0.05},
    };
    for (int q = 0; q < N_POCKET; ++q) p.pocket[q] = pk[q];
    return p;
}

int main(int argc, char** argv) {
    // ---- 1. Load the problem ------------------------------------------------
    DockProblem p;
    const char* source = "synthetic (built-in)";
    if (argc > 1) {
        try {
            p = load_problem(argv[1]);     // parse the committed sample
            source = argv[1];
        } catch (const std::exception& e) {
            // Fall back to the identical built-in problem so the demo always runs.
            std::fprintf(stderr, "[warn] %s -- using built-in synthetic problem\n", e.what());
            p = built_in_problem();
        }
    } else {
        p = built_in_problem();
    }
    const long long M = n_conformations();

    // ---- 2. CPU reference (timed) ------------------------------------------
    std::vector<double> e_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    score_all_cpu(p, e_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU scoring (kernel timed inside the wrapper) ------------------
    std::vector<double> e_gpu;
    float gpu_kernel_ms = 0.0f;
    score_all_gpu(p, e_gpu, &gpu_kernel_ms);

    // ---- 4. Verify, then reduce to the docked pose --------------------------
    const double err = max_abs_err_d(e_cpu, e_gpu);
    const bool pass = err <= TOLERANCE;
    const DockResult best = argmin_energy(e_gpu);   // GPU energies -> best pose

    // Rebuild the winning pose's coordinates so we can report the warhead-sulfur
    // bond length actually achieved (a chemistry-meaningful diagnostic).
    int angle_idx[N_TORSIONS];
    map_conformation_index(best.best_id, angle_idx);
    double torsion[N_TORSIONS];
    for (int j = 0; j < N_TORSIONS; ++j) torsion[j] = sample_to_angle(angle_idx[j]);
    Vec3 pose[N_LIG_ATOMS];
    build_conformation(p, torsion, pose);
    const double warhead_S = vnorm(vsub(p.anchor, p.sg));  // fixed covalent length

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("covalent docking: %lld conformations (%d torsions x %d samples)\n",
                M, N_TORSIONS, GRID_PER_DOF);
    std::printf("best pose: id=%lld  energy=%.6f kcal/mol\n",
                best.best_id, best.best_energy);
    std::printf("best torsions (deg):");
    for (int j = 0; j < N_TORSIONS; ++j)
        std::printf(" %.1f", torsion[j] * 180.0 / 3.14159265358979323846);
    std::printf("\n");
    std::printf("warhead-Sgamma bond = %.3f A (ideal %.3f)\n",
                warhead_S, p.bond_len_ideal);
    std::printf("ligand atom[0] = (%.3f, %.3f, %.3f) A\n",
                pose[0].x, pose[0].y, pose[0].z);
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-06)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%lld conformations)\n", source, M);
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- the GPU's edge grows "
                         "as torsions (and thus conformations) increase.\n");
    std::fprintf(stderr, "[verify] max_abs_err = %.3e kcal/mol  (tolerance %.1e)\n",
                 err, TOLERANCE);

    return pass ? 0 : 1;
}
