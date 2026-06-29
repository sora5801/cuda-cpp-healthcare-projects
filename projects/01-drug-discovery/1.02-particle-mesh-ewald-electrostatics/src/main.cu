// ===========================================================================
// src/main.cu  --  Entry point: load charges, run CPU + GPU PME, verify, report
// ---------------------------------------------------------------------------
// Project 1.2 : Particle-Mesh Ewald Electrostatics
//
// 5-step shape (the repo's standard):
//   1. Load the periodic charge system (data/sample).
//   2. Choose PME parameters (grid K, beta, cutoff) deterministically.
//   3. CPU references:
//        * pme_recip_cpu        -- the SPME pipeline on the host (the GPU's twin).
//        * ewald_recip_direct   -- the textbook k-vector sum (the science check).
//        * real + self          -- so we can assemble the full Ewald energy.
//   4. GPU: pme_recip_gpu (spread -> cuFFT -> convolve), then VERIFY two ways:
//        (a) GPU E_recip  vs  pme_recip_cpu   within a tight FP32-FFT tolerance.
//        (b) Full Ewald energy is INVARIANT to the splitting parameter beta
//            (a physics check that the whole decomposition is correct).
//   5. REPORT deterministic energies to stdout; timings/detail to stderr.
//
// Code tour: start here, then pme.h (shared math), kernels.cuh -> kernels.cu
// (the cuFFT call + atomic spread), then reference_cpu.cpp. THEORY.md has the
// science and the GPU mapping.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // pme_recip_gpu
#include "reference_cpu.h"    // System, PmeParams, CPU references
#include "pme.h"              // PME_ORDER (for the report)
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "1.2";
static const char* PROJECT_NAME = "Particle-Mesh Ewald Electrostatics";

// Verification tolerances (documented; see THEORY "How we verify"):
//   * The charge grid is bit-identical CPU vs GPU (fixed-point spreading), so the
//     ONLY difference is FP32 cuFFT vs FP64 host DFT -> ~1e-6 relative per bin,
//     ~1e-4 relative on the summed reciprocal energy. We require 1e-4.
static constexpr double RECIP_RTOL = 1.0e-4;
//   * SPME is an APPROXIMATION to the exact Ewald reciprocal sum; at the sample's
//     grid/order the truncation error is ~1e-3 relative. This is a SCIENCE check
//     (is SPME a good method?), not a CPU==GPU check, so its tolerance is looser.
static constexpr double SPME_VS_DIRECT_RTOL = 5.0e-3;
//   * Total Ewald energy must be invariant to beta to ~1e-2 relative here (the
//     sample uses a small grid + cutoff; the residual is real discretization
//     error, discussed in THEORY). Still a clear, falsifiable physics check.
static constexpr double BETA_INVARIANCE_RTOL = 2.0e-2;

// Build the Hermitian multiplicity array for the R2C half-spectrum, in the SAME
// layout the energy sum uses: index = (mx*K + my)*(K/2+1) + mz. Interior mz bins
// (0 < mz < K/2) each represent two physical modes (+mz, -mz) -> multiplicity 2;
// mz==0 and mz==K/2 (K even) are self-conjugate -> multiplicity 1.
static void build_multiplicity(const PmeParams& p, std::vector<double>& mult) {
    const int K = p.K, Kh = K / 2 + 1;
    mult.assign(static_cast<std::size_t>(K) * K * Kh, 1.0);
    for (int mx = 0; mx < K; ++mx)
        for (int my = 0; my < K; ++my)
            for (int mz = 0; mz < Kh; ++mz) {
                const std::size_t idx = (static_cast<std::size_t>(mx) * K + my) * Kh + mz;
                mult[idx] = (mz == 0 || (K % 2 == 0 && mz == K / 2)) ? 1.0 : 2.0;
            }
}

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/charges_sample.txt";
    System sys;
    try {
        sys = load_system(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. Parameters -----------------------------------------------------
    const PmeParams p = choose_params(sys);

    // ---- 3. CPU references (timed) ----------------------------------------
    std::vector<double> influence, mult;
    build_influence(sys, p, influence);     // B(m)C(m), uploaded to the GPU too
    build_multiplicity(p, mult);            // Hermitian half-spectrum weights

    util::CpuTimer cpu_timer;
    cpu_timer.start();
    const double recip_cpu = pme_recip_cpu(sys, p);          // SPME on the host
    const double cpu_ms = cpu_timer.stop_ms();

    const double recip_direct = ewald_recip_direct_cpu(sys, p);  // gold standard
    const double e_real = ewald_real_cpu(sys, p);
    const double e_self = ewald_self(sys, p);

    // ---- 4a. GPU: SPME reciprocal energy (timed) --------------------------
    float gpu_ms = 0.0f;
    const double recip_gpu = pme_recip_gpu(sys, p, influence, mult, &gpu_ms);

    // ---- 4b. Verify -------------------------------------------------------
    // (a) GPU == CPU (SPME pipeline twin): the tight numerical check.
    const double rel_gpu_cpu = std::fabs(recip_gpu - recip_cpu) /
                               (std::fabs(recip_cpu) + 1e-30);
    const bool pass_gpu_cpu = rel_gpu_cpu <= RECIP_RTOL;

    // (b) SPME ~ direct Ewald (the science check).
    const double rel_spme_direct = std::fabs(recip_cpu - recip_direct) /
                                   (std::fabs(recip_direct) + 1e-30);
    const bool pass_science = rel_spme_direct <= SPME_VS_DIRECT_RTOL;

    // (c) Beta-invariance of the TOTAL energy: recompute the full direct Ewald
    //     energy at a DIFFERENT beta; the physics says it must be unchanged.
    PmeParams p2 = p;
    p2.beta = p.beta * 1.5;                 // a different real/recip split
    const double total_beta1 = e_real + recip_direct - e_self;
    const double total_beta2 = ewald_total_direct_cpu(sys, p2);
    const double rel_beta = std::fabs(total_beta1 - total_beta2) /
                            (std::fabs(total_beta1) + 1e-30);
    const bool pass_beta = rel_beta <= BETA_INVARIANCE_RTOL;

    const bool pass = pass_gpu_cpu && pass_science && pass_beta;

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("system: %d charges in a cubic box of side %.4f (reduced units)\n",
                sys.n, sys.box);
    std::printf("PME params: grid %dx%dx%d, B-spline order %d, beta %.6f, rcut %.6f\n",
                p.K, p.K, p.K, PME_ORDER, p.beta, p.rcut);
    std::printf("E_recip (GPU SPME) = %.8f\n", recip_gpu);
    std::printf("E_recip (CPU SPME) = %.8f\n", recip_cpu);
    std::printf("E_recip (direct Ewald) = %.8f\n", recip_direct);
    std::printf("E_real = %.8f   E_self = %.8f\n", e_real, e_self);
    std::printf("E_total (real + recip - self) = %.8f\n", total_beta1);
    std::printf("CHECK GPU==CPU SPME      : %s (rel %.2e <= %.0e)\n",
                pass_gpu_cpu ? "PASS" : "FAIL", rel_gpu_cpu, RECIP_RTOL);
    std::printf("CHECK SPME~=direct Ewald : %s (rel %.2e <= %.0e)\n",
                pass_science ? "PASS" : "FAIL", rel_spme_direct, SPME_VS_DIRECT_RTOL);
    std::printf("CHECK total invariant to beta : %s (rel %.2e <= %.0e)\n",
                pass_beta ? "PASS" : "FAIL", rel_beta, BETA_INVARIANCE_RTOL);
    std::printf("RESULT: %s\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d charges, box %.3f)\n",
                 path.c_str(), sys.n, sys.box);
    std::fprintf(stderr, "[timing] CPU SPME (naive separable DFT): %.3f ms   "
                         "GPU SPME (cuFFT): %.3f ms\n", cpu_ms, gpu_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- the host DFT is O(K^4); cuFFT is "
                         "O(K^3 log K). The gap grows fast with grid size K.\n");
    std::fprintf(stderr, "[verify] GPU-vs-CPU SPME rel err = %.3e (FP32 cuFFT vs FP64 DFT)\n",
                 rel_gpu_cpu);
    std::fprintf(stderr, "[verify] SPME-vs-direct rel err  = %.3e (SPME truncation, K=%d order=%d)\n",
                 rel_spme_direct, p.K, PME_ORDER);
    std::fprintf(stderr, "[verify] beta1=%.4f total=%.6f ; beta2=%.4f total=%.6f ; rel=%.3e\n",
                 p.beta, total_beta1, p2.beta, total_beta2, rel_beta);

    return pass ? 0 : 1;
}
