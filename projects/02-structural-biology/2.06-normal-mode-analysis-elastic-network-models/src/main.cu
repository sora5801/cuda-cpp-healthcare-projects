// ===========================================================================
// src/main.cu  --  Entry point: build Hessian, diagonalize, verify, report
// ---------------------------------------------------------------------------
// Project 2.06 : Normal Mode Analysis / Elastic Network Models
//
// 5-step shape:
//   1. Load the Cα structure (data/sample) and build the ANM Hessian.
//   2. CPU reference eigenvalues via Jacobi (reference_cpu.cpp).
//   3. GPU eigendecomposition via cuSOLVER (kernels.cu).
//   4. VERIFY: cuSOLVER eigenvalues match the Jacobi reference.
//   5. REPORT: zero modes, lowest functional frequencies, residue mobility.
//
// Code tour: start here, then reference_cpu.cpp (Hessian + Jacobi), kernels.cu
// (cuSOLVER). The science/GPU-mapping is in ../THEORY.md.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // cusolver_eigen, Protein
#include "reference_cpu.h"    // load_protein, build_hessian, jacobi_eigenvalues, mobility
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "2.6";
static const char* PROJECT_NAME = "Normal Mode Analysis / Elastic Network Models";

static constexpr double GAMMA     = 1.0;       // uniform spring constant
static constexpr double ZERO_THR  = 1.0e-6;    // eigenvalue below this = rigid-body mode
static constexpr double TOLERANCE = 1.0e-4;    // cuSOLVER vs Jacobi eigenvalue agreement

int main(int argc, char** argv) {
    // ---- 1. Load + build Hessian ------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/protein_ca.txt";
    Protein prot;
    try {
        prot = load_protein(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int n = 3 * prot.N;
    std::vector<double> H;
    build_hessian(prot, GAMMA, H);

    // ---- 2. CPU reference: Jacobi eigenvalues (timed) ---------------------
    std::vector<double> eig_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    jacobi_eigenvalues(H, n, eig_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU: cuSOLVER eigendecomposition (timed) ----------------------
    std::vector<double> eig_gpu, evec_gpu;
    float gpu_kernel_ms = 0.0f;
    cusolver_eigen(H, n, eig_gpu, evec_gpu, &gpu_kernel_ms);

    // ---- 4. Verify (eigenvalues agree) ------------------------------------
    double worst = 0.0;
    for (int i = 0; i < n; ++i)
        worst = std::fmax(worst, std::fabs(eig_cpu[i] - eig_gpu[i]));
    const bool pass = worst <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    int zero_modes = 0;
    for (int i = 0; i < n; ++i) if (eig_gpu[i] <= ZERO_THR) ++zero_modes;

    std::vector<double> mob;
    mobility(eig_gpu, evec_gpu, prot.N, ZERO_THR, mob);
    int most_mobile = 0;
    for (int i = 1; i < prot.N; ++i) if (mob[i] > mob[most_mobile]) most_mobile = i;

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("ANM: %d residues, cutoff=%.1f A, Hessian %dx%d\n", prot.N, prot.cutoff, n, n);
    std::printf("zero (rigid-body) modes: %d  (expected 6)\n", zero_modes);
    std::printf("lowest 6 non-zero mode eigenvalues:");
    for (int m = 0; m < 6 && zero_modes + m < n; ++m) std::printf(" %.5f", eig_gpu[zero_modes + m]);
    std::printf("\n");
    std::printf("most mobile residue: %d (mobility=%.5f)\n", most_mobile, mob[most_mobile]);
    std::printf("residue mobility (8 sampled):");
    for (int s = 0; s < 8; ++s) {
        const int i = (s * (prot.N - 1)) / 7;
        std::printf(" %.5f", mob[i]);
    }
    std::printf("\n");
    std::printf("RESULT: %s (cuSOLVER eigenvalues match CPU Jacobi within tol=1.0e-04)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d residues, n=%d)\n", path.c_str(), prot.N, n);
    std::fprintf(stderr, "[timing] CPU Jacobi: %.3f ms   GPU cuSOLVER: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- the eigensolver is O(n^3); the GPU's edge explodes "
                         "for large proteins (n = 3N up to 150,000).\n");
    std::fprintf(stderr, "[verify] worst eigenvalue diff = %.3e  (tolerance %.1e)\n", worst, TOLERANCE);

    return pass ? 0 : 1;
}
