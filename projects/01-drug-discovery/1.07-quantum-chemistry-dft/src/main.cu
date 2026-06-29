// ===========================================================================
// src/main.cu  --  Entry point: build integrals, run SCF on CPU + GPU, verify
// ---------------------------------------------------------------------------
// Project 1.7 : Quantum Chemistry / DFT  (reduced-scope RHF/SCF -- see THEORY.md)
//
// 5-step shape (the same skeleton as every flagship):
//   1. Load the molecule (data/sample), build the basis + cheap 1-electron matrices.
//   2. Build the O(N^4) two-electron tensor BOTH ways:
//        - CPU reference (build_eri_cpu)
//        - GPU kernel    (build_eri_gpu)   <-- the project's headline kernel
//      and VERIFY they agree to ~machine precision (the integrals are the crux).
//   3. Run the SCF loop with the CPU eigensolver (run_scf) -> reference energy.
//   4. Run the SAME SCF loop with the cuSOLVER eigensolver -> GPU energy.
//   5. VERIFY the two total energies agree within tolerance; REPORT the energy,
//      orbital levels, and HOMO-LUMO gap deterministically to stdout.
//
// Code tour: start here, then gaussian_integrals.h (the shared formulas),
//   reference_cpu.cpp (basis + SCF), kernels.cu (the GPU ERI kernel + cuSOLVER).
//   The science / GPU-mapping lives in ../THEORY.md.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // build_eri_gpu, cusolver_generalized
#include "reference_cpu.h"    // load_molecule, build_*, run_scf, build_density/fock
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "1.7";
static const char* PROJECT_NAME = "Quantum Chemistry / DFT (reduced-scope RHF/SCF)";

// SCF + verification controls.
static constexpr int    MAX_ITER   = 50;       // SCF cycle cap (converges in < 20)
static constexpr double E_TOL      = 1.0e-9;   // SCF energy convergence (Hartree)
static constexpr double ERI_TOL    = 1.0e-12;  // GPU-vs-CPU integral agreement
static constexpr double ENERGY_TOL = 1.0e-9;   // GPU-vs-CPU final energy agreement

// ---------------------------------------------------------------------------
// run_scf_gpu: the identical RHF self-consistent loop as run_scf(), but the
//   per-cycle generalized eigensolve is done by cuSOLVER (cusolver_generalized)
//   instead of the CPU Jacobi path. Lives here (not in kernels.cu) because it just
//   orchestrates host-side helpers + one GPU library call per iteration. Proving
//   THIS converges to the same energy as the CPU loop is the end-to-end check that
//   the whole GPU pipeline (integrals + eigensolver) is correct.
// ---------------------------------------------------------------------------
static ScfResult run_scf_gpu(const std::vector<double>& S, const std::vector<double>& Hcore,
                             const std::vector<double>& eri, int N, int n_occ,
                             double e_nuclear, int max_iter, double e_tol) {
    ScfResult res;
    res.e_nuclear = e_nuclear;
    std::vector<double> F = Hcore;            // core guess
    std::vector<double> C, eps, P;
    double e_elec_prev = 0.0;

    for (int iter = 1; iter <= max_iter; ++iter) {
        cusolver_generalized(F, S, N, C, eps);          // 1. orbitals (GPU eigensolve)
        build_density(C, N, n_occ, P);                  // 2. density from occ. MOs
        build_fock(Hcore, P, eri, N, F);                // 3. new Fock matrix
        double e_elec = 0.0;                            // 4. electronic energy
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                e_elec += 0.5 * P[(size_t)i * N + j] *
                          (Hcore[(size_t)i * N + j] + F[(size_t)i * N + j]);
        res.iterations = iter;
        res.orbital_energies = eps;
        res.e_electronic = e_elec;
        if (std::fabs(e_elec - e_elec_prev) < e_tol) { res.converged = true; break; }
        e_elec_prev = e_elec;
    }
    res.e_total = res.e_electronic + e_nuclear;
    if (n_occ - 1 >= 0 && n_occ - 1 < N) res.homo = res.orbital_energies[n_occ - 1];
    if (n_occ < N)                       res.lumo = res.orbital_energies[n_occ];
    return res;
}

int main(int argc, char** argv) {
    // ---- 1. Load molecule, build basis + one-electron matrices ------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/h2.txt";
    Molecule mol;
    try {
        mol = load_molecule(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const Basis bs = build_basis(mol);
    const int N = static_cast<int>(bs.size());     // # basis functions (= # atoms here)
    const int n_occ = mol.n_electrons / 2;         // doubly-occupied orbitals
    const double e_nuc = nuclear_repulsion(mol);

    std::vector<double> S, Hcore;
    build_overlap(bs, N, S);
    build_core_hamiltonian(bs, mol, N, Hcore);

    // ---- 2. Two-electron tensor: CPU reference + GPU kernel, then verify ---
    std::vector<double> eri_cpu, eri_gpu;
    util::CpuTimer cpu_eri_timer;
    cpu_eri_timer.start();
    build_eri_cpu(bs, N, eri_cpu);
    const double cpu_eri_ms = cpu_eri_timer.stop_ms();

    float gpu_eri_ms = 0.0f;
    build_eri_gpu(bs, N, eri_gpu, &gpu_eri_ms);

    double worst_eri = 0.0;
    const size_t n4 = static_cast<size_t>(N) * N * N * N;
    for (size_t t = 0; t < n4; ++t)
        worst_eri = std::fmax(worst_eri, std::fabs(eri_cpu[t] - eri_gpu[t]));
    const bool eri_pass = worst_eri <= ERI_TOL;

    // ---- 3. CPU SCF (reference energy) ------------------------------------
    util::CpuTimer cpu_scf_timer;
    cpu_scf_timer.start();
    ScfResult cpu = run_scf(S, Hcore, eri_cpu, N, n_occ, e_nuc, MAX_ITER, E_TOL);
    const double cpu_scf_ms = cpu_scf_timer.stop_ms();

    // ---- 4. GPU SCF (cuSOLVER eigensolver, GPU-built integrals) ------------
    util::CpuTimer gpu_scf_timer;
    gpu_scf_timer.start();
    ScfResult gpu = run_scf_gpu(S, Hcore, eri_gpu, N, n_occ, e_nuc, MAX_ITER, E_TOL);
    const double gpu_scf_ms = gpu_scf_timer.stop_ms();

    // ---- 5. Verify energies agree -----------------------------------------
    const double e_diff = std::fabs(cpu.e_total - gpu.e_total);
    const bool energy_pass = e_diff <= ENERGY_TOL;
    const bool pass = eri_pass && energy_pass && cpu.converged && gpu.converged;

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    // We print the GPU result (verified == CPU). Energies in Hartree, fixed width.
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("molecule: %d atoms, %d electrons, basis STO-3G (N=%d functions)\n",
                static_cast<int>(mol.atoms.size()), mol.n_electrons, N);
    std::printf("SCF converged in %d iterations\n", gpu.iterations);
    std::printf("nuclear repulsion : %12.8f Ha\n", gpu.e_nuclear);
    std::printf("electronic energy : %12.8f Ha\n", gpu.e_electronic);
    std::printf("TOTAL ENERGY      : %12.8f Ha\n", gpu.e_total);
    std::printf("orbital energies (Ha):");
    for (int k = 0; k < N; ++k) std::printf(" %9.5f", gpu.orbital_energies[k]);
    std::printf("\n");
    std::printf("HOMO = %9.5f Ha   LUMO = %9.5f Ha   gap = %9.5f Ha\n",
                gpu.homo, gpu.lumo, gpu.lumo - gpu.homo);
    std::printf("ERI verify (GPU vs CPU): %s   energy verify (GPU vs CPU): %s\n",
                eri_pass ? "PASS" : "FAIL", energy_pass ? "PASS" : "FAIL");
    std::printf("RESULT: %s\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail / timings -> STDERR ---------------------------
    std::fprintf(stderr, "[data]   source: %s  (N=%d basis fns, %lld ERIs)\n",
                 path.c_str(), N, static_cast<long long>(n4));
    std::fprintf(stderr, "[timing] ERI build  CPU: %.3f ms   GPU(kernel): %.3f ms\n",
                 cpu_eri_ms, gpu_eri_ms);
    std::fprintf(stderr, "[timing] SCF loop   CPU: %.3f ms   GPU: %.3f ms\n",
                 cpu_scf_ms, gpu_scf_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- at N=%d the GPU is launch-bound; "
                 "the O(N^4) ERI kernel's edge explodes with basis size (real molecules: "
                 "N in the hundreds-to-thousands).\n", N);
    std::fprintf(stderr, "[verify] worst |ERI_cpu - ERI_gpu| = %.3e (tol %.1e)\n",
                 worst_eri, ERI_TOL);
    std::fprintf(stderr, "[verify] |E_cpu - E_gpu| = %.3e (tol %.1e)  E_cpu = %.8f Ha\n",
                 e_diff, ENERGY_TOL, cpu.e_total);

    return pass ? 0 : 1;
}
