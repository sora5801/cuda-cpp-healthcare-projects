// ===========================================================================
// src/main.cu  --  Entry point: build the PBE problem, solve, verify, report
// ---------------------------------------------------------------------------
// Project 2.9 : Solvent-Accessible Surface & Poisson-Boltzmann Electrostatics
//
// 5-step shape (the pattern shared by every project in this repo):
//   1. Load the atoms + grid parameters (reference_cpu.cpp) and build the
//      eps / kappa^2 / rho grids (build_problem).
//   2. CPU reference solve: serial red-black Gauss-Seidel (solve_cpu).
//   3. GPU solve: the same relaxation, red-black parallel (solve_gpu, kernels.cu).
//   4. VERIFY: the GPU potential field matches the CPU field within tolerance.
//   5. REPORT: deterministic electrostatics summary + SASA to STDOUT; timing to
//      STDERR (so stdout is byte-identical every run and the demo can diff it).
//
// Code tour: start here, then pbe.h (the shared per-cell math), kernels.cu (the
// red-black GPU sweep), reference_cpu.cpp (the trusted serial baseline).
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // solve_gpu, PbeProblem, GridParams
#include "reference_cpu.h"    // load_atoms, build_problem, solve_cpu, compute_sasa
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "2.9";
static const char* PROJECT_NAME = "Solvent-Accessible Surface & Poisson-Boltzmann Electrostatics";

// Verification tolerance. CPU and GPU run the IDENTICAL red-black Gauss-Seidel
// arithmetic in the IDENTICAL colour order, all in double precision, so they
// agree to near machine precision. We allow a small margin (1e-9) because over
// thousands of sweeps the GPU's fused multiply-add can differ from the host
// compiler in the last bits (PATTERNS.md sec.4). The potentials are O(1-10), so
// 1e-9 is "the same field to ~9 significant digits" -- documented, not pretended.
static constexpr double TOLERANCE = 1.0e-9;

// SASA sampling: water probe radius 1.4 A, 256 Fibonacci-sphere points/atom.
// Fixed constants so the reported surface area is fully deterministic.
static constexpr double PROBE_RADIUS = 1.4;
static constexpr int    SASA_POINTS  = 256;

int main(int argc, char** argv) {
    // ---- 1. Load atoms + build the grids -----------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/molecule.pqr";
    GridParams P{};
    std::vector<Atom> atoms;
    try {
        atoms = load_atoms(path, P);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const PbeProblem prob = build_problem(atoms, P);
    const int n = prob.P.n;
    const size_t Ncells = static_cast<size_t>(n) * n * n;

    // ---- 2. CPU reference solve (timed) ------------------------------------
    std::vector<double> phi_cpu(Ncells, 0.0);   // grounded box: zero everywhere
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    solve_cpu(prob, phi_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU solve (loop timed inside solve_gpu) ------------------------
    std::vector<double> phi_gpu(Ncells, 0.0);
    float gpu_kernel_ms = 0.0f;
    solve_gpu(prob, phi_gpu, &gpu_kernel_ms);

    // ---- 4. Verify (fields agree) ------------------------------------------
    double worst = 0.0;
    for (size_t i = 0; i < Ncells; ++i)
        worst = std::fmax(worst, std::fabs(phi_cpu[i] - phi_gpu[i]));
    const bool pass = worst <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    // Electrostatics summary from the GPU field: the extreme potentials (where
    // positive/negative charge piles up), and the potential at the grid centre.
    double phi_min = phi_gpu[0], phi_max = phi_gpu[0], sum_abs = 0.0;
    for (size_t i = 0; i < Ncells; ++i) {
        phi_min = std::fmin(phi_min, phi_gpu[i]);
        phi_max = std::fmax(phi_max, phi_gpu[i]);
        sum_abs += std::fabs(phi_gpu[i]);
    }
    const int cc = pbe_idx(n / 2, n / 2, n / 2, n);   // centre cell
    const double phi_center = phi_gpu[cc];

    // The SASA -- the geometric "surface" half of the project (CPU, deterministic).
    const double sasa = compute_sasa(atoms, PROBE_RADIUS, SASA_POINTS);

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("grid: %dx%dx%d cells, h=%.2f A, eps_in=%.1f eps_out=%.1f kappa^2=%.4f, sweeps=%d\n",
                n, n, n, prob.P.h, prob.P.eps_in, prob.P.eps_out, prob.P.kappa2, prob.P.iters);
    std::printf("atoms: %zu  | SASA (probe=1.4 A) = %.2f A^2\n", atoms.size(), sasa);
    std::printf("potential (kT/e): min=%.6f  max=%.6f  center=%.6f  sum|phi|=%.4f\n",
                phi_min, phi_max, phi_center, sum_abs);
    // Potential profile along the central x-axis line (8 samples): shows the
    // charge well and its decay into the screened solvent.
    std::printf("phi along center x-line (8 samples):");
    const int yc = n / 2, zc = n / 2;
    for (int s = 0; s < 8; ++s) {
        const int x = (s * (n - 1)) / 7;
        std::printf(" %.4f", phi_gpu[pbe_idx(x, yc, zc, n)]);
    }
    std::printf("\n");
    std::printf("RESULT: %s (GPU field matches CPU within tol=1.0e-09)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Run-varying detail -> STDERR ----------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%zu atoms, %zu grid cells)\n",
                 path.c_str(), atoms.size(), Ncells);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- the GPU's edge over the serial sweep "
                         "grows with grid size; tiny grids are launch-bound.\n");
    std::fprintf(stderr, "[verify] worst |phi_cpu - phi_gpu| = %.3e  (tolerance %.1e)\n",
                 worst, TOLERANCE);

    return pass ? 0 : 1;
}
