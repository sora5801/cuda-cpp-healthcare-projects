// ===========================================================================
// src/main.cu  --  Entry point: load batch, run CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 1.8 : Semi-Empirical & Tight-Binding Quantum Methods
//
// The 5-step shape every project in this repo follows:
//   1. Load the problem (a batch of conjugated molecules from data/sample).
//   2. CPU reference  (reference_cpu.cpp): build each Huckel matrix, diagonalise
//      with Jacobi, fill electrons -> trusted MO energies / HOMO-LUMO gaps.
//   3. GPU batch      (kernels.cu): build ALL matrices on-device, diagonalise
//      the WHOLE batch in one cuSOLVER call -> the thing being taught.
//   4. VERIFY: per-molecule, the GPU eigenvalues match the CPU within tolerance.
//   5. REPORT: a deterministic per-molecule table to stdout; timing to stderr.
//
// STDOUT is byte-for-byte deterministic (diffed by demo/run_demo against
// demo/expected_output.txt); run-to-run timings go to STDERR (PATTERNS.md §3).
//
// Code tour: start here, then tight_binding.h (the shared physics), then
// reference_cpu.cpp (CPU pipeline) and kernels.cu (GPU pipeline).
// ===========================================================================
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // tb_solve_batch_gpu
#include "reference_cpu.h"    // load_batch, build_hamiltonian, jacobi_eigen, analyze_molecule
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "1.8";
static const char* PROJECT_NAME = "Semi-Empirical & Tight-Binding Quantum Methods";

// Tolerance: both sides diagonalise the SAME symmetric matrix with Jacobi-family
// solvers in double precision, so eigenvalues agree to ~machine precision. 1e-9
// is a comfortable margin above the ~1e-13 we actually observe (PATTERNS.md §4).
static constexpr double TOLERANCE = 1.0e-9;

// clean_zero: snap values within PRINT_EPS of zero to +0.0 before printing.
//   WHY (determinism, PATTERNS.md §3): several molecules here have eigenvalues
//   that are EXACTLY zero by symmetry (allyl's non-bonding MO, cyclobutadiene's
//   degenerate pair). Two different eigensolvers can return that zero as +1e-16
//   or -1e-16, which "%.6f" would render as "0.000000" vs "-0.000000" -- a
//   spurious stdout difference. Snapping tiny magnitudes to +0 makes the printed
//   table byte-identical regardless of the sign of the floating-point noise. The
//   snap is far below any physically meaningful energy, so it changes nothing
//   chemical; it only stabilises the display.
static double clean_zero(double x) {
    const double PRINT_EPS = 5.0e-7;   // half a ulp of the printed 6th decimal
    return (std::fabs(x) < PRINT_EPS) ? 0.0 : x;
}

int main(int argc, char** argv) {
    // ---- 1. Load the batch -------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/molecules_sample.txt";
    MoleculeBatch batch;
    try {
        batch = load_batch(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int M = batch.num_mol;     // molecules in the batch
    const int N = batch.max_n;       // padded matrix dimension

    // ---- 2. CPU reference: per-molecule build + Jacobi + analyse (timed) --
    std::vector<MoleculeResult> res_cpu(M);
    std::vector<std::vector<double>> eval_cpu(M);   // keep eigenvalues for verify
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    for (int m = 0; m < M; ++m) {
        std::vector<double> H, evec;
        build_hamiltonian(batch, m, H);
        jacobi_eigen(H, N, eval_cpu[m], evec);
        res_cpu[m] = analyze_molecule(eval_cpu[m], batch.n[m]);
    }
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU: build all matrices + batched eigensolve ------------------
    std::vector<double> eval_gpu_flat;               // [M*N] ascending per block
    float build_ms = 0.0f, solve_ms = 0.0f;
    tb_solve_batch_gpu(batch.adj, batch.n, M, N, eval_gpu_flat, &build_ms, &solve_ms);

    // Run the SAME chemistry post-processing on the GPU eigenvalues so the two
    // reports are produced by identical code paths above the eigensolver.
    std::vector<MoleculeResult> res_gpu(M);
    for (int m = 0; m < M; ++m) {
        std::vector<double> ev(eval_gpu_flat.begin() + (std::size_t)m * N,
                               eval_gpu_flat.begin() + (std::size_t)(m + 1) * N);
        res_gpu[m] = analyze_molecule(ev, batch.n[m]);
    }

    // ---- 4. Verify: GPU eigenvalues match CPU per molecule ----------------
    double worst = 0.0;
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < batch.n[m]; ++k)
            worst = std::fmax(worst,
                std::fabs(eval_cpu[m][k] - eval_gpu_flat[(std::size_t)m * N + k]));
    const bool pass = worst <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    // All printed numbers come from the GPU path (verified == CPU). Energies are
    // in units of |beta| relative to alpha (Huckel convention; see THEORY.md).
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("Huckel tight-binding on %d molecules (padded dim N=%d)\n", M, N);
    std::printf("energies in units of |beta| (alpha=0, beta=-1)\n");
    std::printf("%-16s %5s %14s %10s %10s %10s\n",
                "molecule", "atoms", "E_pi", "HOMO", "LUMO", "gap");
    for (int m = 0; m < M; ++m) {
        const MoleculeResult& r = res_gpu[m];
        std::printf("%-16s %5d %14.6f %10.6f %10.6f %10.6f\n",
                    batch.name[m].c_str(), r.n_atoms,
                    clean_zero(r.total_pi_energy), clean_zero(r.homo_energy),
                    clean_zero(r.lumo_energy), clean_zero(r.homo_lumo_gap));
    }
    // A single interpretable headline: the most reactive molecule = smallest gap.
    int min_gap_mol = 0;
    for (int m = 1; m < M; ++m)
        if (res_gpu[m].homo_lumo_gap < res_gpu[min_gap_mol].homo_lumo_gap)
            min_gap_mol = m;
    std::printf("smallest HOMO-LUMO gap: %s (gap=%.6f |beta|) -- most reactive/polarizable\n",
                batch.name[min_gap_mol].c_str(), clean_zero(res_gpu[min_gap_mol].homo_lumo_gap));
    std::printf("RESULT: %s (GPU batched eigensolve matches CPU Jacobi within tol=1.0e-09)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d molecules, padded N=%d)\n",
                 path.c_str(), M, N);
    std::fprintf(stderr, "[timing] CPU (build+Jacobi+fill, all molecules): %.3f ms\n", cpu_ms);
    std::fprintf(stderr, "[timing] GPU build kernel: %.3f ms   GPU cuSOLVER batched solve: %.3f ms\n",
                 build_ms, solve_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny batch is dominated by launch/"
                         "setup overhead; the batched solver's edge grows with thousands of molecules.\n");
    std::fprintf(stderr, "[verify] worst eigenvalue diff = %.3e  (tolerance %.1e)\n", worst, TOLERANCE);

    return pass ? 0 : 1;
}
