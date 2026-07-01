// ===========================================================================
// src/main.cu  --  Entry point: load sinogram, decompose, verify, report
// ---------------------------------------------------------------------------
// Project 4.20 : Dual-Energy / Spectral CT Reconstruction
//
// The 5-step shape every project in this repo follows:
//   1. Load the problem (a dual-energy sinogram from data/sample).
//   2. CPU reference  (reference_cpu.cpp)  -> trusted (t1,t2) per bin.
//   3. GPU decompose  (kernels.cu)         -> the thing being taught.
//   4. VERIFY: GPU agrees with CPU within tolerance; and (bonus) the recovered
//      path lengths match the KNOWN synthetic truth -> the science is right.
//   5. REPORT: deterministic per-bin decomposition + a virtual-monoenergetic
//      value to stdout; timing + errors to stderr.
//
// STDOUT is byte-for-byte deterministic (diffed by demo/run_demo against
// demo/expected_output.txt); run-to-run timings go to STDERR.
//
// Code tour: start here, then dect.h (the physics) -> kernels.cuh -> kernels.cu,
// then reference_cpu.*.
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // decompose_gpu
#include "reference_cpu.h"    // load_sinogram, build_spectral_model, decompose_cpu
#include "dect.h"             // virtual_mono_mu, NUM_ENERGIES
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "4.20";
static const char* PROJECT_NAME = "Dual-Energy / Spectral CT Reconstruction";

// Tolerance for GPU-vs-CPU agreement. Both sides call the IDENTICAL
// __host__ __device__ Newton core (dect.h) with the identical linear seed, so
// they execute the same double-precision operations in the same order and agree
// to the last bit; 1e-9 cm is a generous safety margin (PATTERNS.md §4, the
// "same exact operations on both sides" case). Path lengths here are O(1-20) cm.
static constexpr double TOLERANCE_CM = 1.0e-9;

// How many bins to print in the deterministic table (keeps stdout compact).
static constexpr int PRINT_BINS = 8;

// max |a-b| over two double vectors; +inf on a size mismatch so a shape bug
// cannot masquerade as agreement.
static double max_abs_err(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return 1e300;
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double d = std::fabs(a[i] - b[i]);
        if (d > worst) worst = d;
    }
    return worst;
}

int main(int argc, char** argv) {
    // ---- 1. Load ----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1]
                                        : "data/sample/dect_sinogram_sample.txt";
    DectSinogram sino;
    try {
        sino = load_sinogram(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    // The scanner physics is built in code (deterministic) rather than loaded, so
    // the demo is fully reproducible and needs no extra data file.
    const SpectralModel sm = build_spectral_model();

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<double> t1_cpu, t2_cpu;
    std::vector<int>    it_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    decompose_cpu(sino, sm, t1_cpu, t2_cpu, it_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU decompose (kernel timed inside the wrapper) ---------------
    std::vector<double> t1_gpu, t2_gpu;
    std::vector<int>    it_gpu;
    float gpu_kernel_ms = 0.0f;
    decompose_gpu(sino, sm, t1_gpu, t2_gpu, it_gpu, &gpu_kernel_ms);

    // ---- 4. Verify --------------------------------------------------------
    // (a) GPU vs CPU: must agree to ~machine precision (they run identical math).
    const double err_t1 = max_abs_err(t1_cpu, t1_gpu);
    const double err_t2 = max_abs_err(t2_cpu, t2_gpu);
    const double err = (err_t1 > err_t2) ? err_t1 : err_t2;
    const bool pass = err <= TOLERANCE_CM;

    // (b) Bonus science check: recovery error vs the KNOWN synthetic truth. This
    // validates the DECOMPOSITION (not just CPU==GPU). It will be a small
    // non-zero number: the forward model that made the data is exactly the one we
    // invert, so the only error is Newton's residual floor -> ~1e-10 cm.
    double worst_recovery = 0.0;
    const bool have_truth = !sino.true_t1.empty();
    if (have_truth) {
        for (int i = 0; i < sino.n; ++i) {
            double e1 = std::fabs(t1_gpu[i] - sino.true_t1[i]);
            double e2 = std::fabs(t2_gpu[i] - sino.true_t2[i]);
            if (e1 > worst_recovery) worst_recovery = e1;
            if (e2 > worst_recovery) worst_recovery = e2;
        }
    }

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    // We choose a fixed virtual-monoenergetic energy to synthesize (index into
    // the sampled energy grid). VMI at a low keV boosts iodine contrast; here we
    // report the mid-grid energy so the number is stable and meaningful.
    const int vmi_k = NUM_ENERGIES / 2;             // ~85 keV on the 30-140 grid
    const double vmi_keV = sm.energy_keV[vmi_k];

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("Projection-domain material decomposition: %d sinogram bins, "
                "2x2 Newton per bin\n", sino.n);
    std::printf("basis 1 = water-equivalent, basis 2 = iodine-equivalent "
                "(path lengths in cm)\n");
    const int nprint = (sino.n < PRINT_BINS) ? sino.n : PRINT_BINS;
    std::printf("first %d bins (recovered on GPU):\n", nprint);
    std::printf("  bin   m_lo     m_hi     t1_water  t2_iodine  iters\n");
    for (int i = 0; i < nprint; ++i) {
        std::printf("  %3d  %7.4f  %7.4f  %8.4f  %9.4f   %3d\n",
                    i, sino.m_lo[i], sino.m_hi[i],
                    t1_gpu[i], t2_gpu[i], it_gpu[i]);
    }
    // Virtual monoenergetic image value for bin 0: the monochromatic
    // log-attenuation a single-energy scan at vmi_keV would have measured.
    const double vmi0 = virtual_mono_mu(t1_gpu[0], t2_gpu[0], sm, vmi_k);
    std::printf("virtual monoenergetic (bin 0) at %.1f keV: mu*L = %.4f\n",
                vmi_keV, vmi0);
    if (have_truth) {
        std::printf("max recovery error vs known truth: %.2e cm\n", worst_recovery);
    }
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-09 cm)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (n=%d bins, has_truth=%d)\n",
                 path.c_str(), sino.n, have_truth ? 1 : 0);
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny sample is "
                         "dominated by launch/copy overhead; the GPU wins at scanner "
                         "scale (~10^8 bins).\n");
    std::fprintf(stderr, "[verify] max|GPU-CPU| = %.3e cm (t1) / %.3e cm (t2), "
                         "tol %.1e\n", err_t1, err_t2, TOLERANCE_CM);

    return pass ? 0 : 1;
}
