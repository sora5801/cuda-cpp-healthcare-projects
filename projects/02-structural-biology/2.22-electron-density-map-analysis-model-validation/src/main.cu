// ===========================================================================
// src/main.cu  --  Entry point: load maps, run CPU + GPU validation, report
// ---------------------------------------------------------------------------
// Project 2.22 : Electron Density Map Analysis & Model Validation
//
// WHAT THIS FILE DOES  (the 5-step shape EVERY project in this repo follows)
//   1. Load the two co-sampled density maps (data/sample/<file>).
//   2. CPU reference: RSCC + naive-DFT FSC curve (reference_cpu.cpp) -> trusted.
//   3. GPU: cuFFT-based FSC + block-reduced RSCC (kernels.cu)        -> taught.
//   4. VERIFY: assert the GPU RSCC and FSC agree with the CPU within tolerance.
//   5. REPORT: deterministic validation result to stdout; timing to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. We print the CPU's fully-deterministic DOUBLE
//   values (RSCC, FSC curve, resolution); the GPU's single-precision FFT agrees
//   to ~1e-5, which we report on STDERR (the worst error) but do not diff.
//   Anything that varies run-to-run (timings) also goes to STDERR.
//
// READ THIS FIRST in the code tour, then kernels.cuh -> kernels.cu (the cuFFT
// call), map_core.h (shared math), and reference_cpu.cpp (the baseline). The
// science/GPU-mapping/numerics live in ../THEORY.md.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // validate_gpu (GPU path)
#include "reference_cpu.h"    // load_map, rscc_cpu, fsc_cpu, resolution_at_threshold, shell_to_res
#include "map_core.h"         // max_shell (for the shared shell bound)
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "2.22";
static const char* PROJECT_NAME = "Electron Density Map Analysis & Model Validation";

// Verification tolerances (documented; PATTERNS.md §4).
//   RSCC: both sides use the SAME double formula on the same maps -> agreement is
//   ~machine precision; 1e-9 is comfortably tight.
//   FSC : the GPU FFT is SINGLE precision while the CPU DFT is double, so the per
//   shell correlation differs by ~1e-5 (real, and worth teaching). We verify the
//   FSC curve agrees to 1e-4 -- physically negligible for a correlation in
//   [-1,1], and we print the worst error so the gap is honest, not hidden.
static constexpr double RSCC_TOL = 1.0e-9;
static constexpr double FSC_TOL  = 1.0e-4;

// The two FSC resolution thresholds we report:
//   0.143 -- the cryo-EM "gold-standard" cutoff for two INDEPENDENT half-maps.
//   0.5   -- the classic cutoff used for map-vs-model FSC.
static constexpr double FSC_GOLD = 0.143;
static constexpr double FSC_HALF = 0.5;

int main(int argc, char** argv) {
    // ---- 1. Load the maps --------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/map_sample.txt";
    DensityMap d;
    try {
        d = load_map(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) ------------------------------------------
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    const double rscc_cpu_val = rscc_cpu(d);
    std::vector<double> fsc_cpu_curve;
    std::vector<long long> shell_count;
    fsc_cpu(d, fsc_cpu_curve, shell_count);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU result (kernels timed inside the wrapper) ------------------
    double rscc_gpu_val = 0.0;
    std::vector<double> fsc_gpu_curve;
    std::vector<long long> shell_count_gpu;
    float gpu_kernel_ms = 0.0f;
    validate_gpu(d, &rscc_gpu_val, fsc_gpu_curve, shell_count_gpu, &gpu_kernel_ms);

    // ---- 4. Verify (GPU vs CPU) --------------------------------------------
    const double rscc_err = std::fabs(rscc_cpu_val - rscc_gpu_val);
    double fsc_worst = 0.0;
    for (std::size_t s = 0; s < fsc_cpu_curve.size(); ++s) {
        const double e = std::fabs(fsc_cpu_curve[s] - fsc_gpu_curve[s]);
        if (e > fsc_worst) fsc_worst = e;
    }
    const bool pass = (rscc_err <= RSCC_TOL) && (fsc_worst <= FSC_TOL);

    // Resolution estimates from the (deterministic) CPU FSC curve.
    const int s_gold = resolution_at_threshold(fsc_cpu_curve, shell_count, FSC_GOLD);
    const int s_half = resolution_at_threshold(fsc_cpu_curve, shell_count, FSC_HALF);
    const double res_gold = shell_to_res(s_gold, d.n, d.voxel_angstrom);
    const double res_half = shell_to_res(s_half, d.n, d.voxel_angstrom);

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    // We print the CPU's double-precision values so stdout is byte-identical
    // every run; the GPU agreement is asserted (and printed on stderr).
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("map: %d^3 voxels, %.2f A/voxel (box = %.1f A)\n",
                d.n, d.voxel_angstrom, d.n * d.voxel_angstrom);
    std::printf("RSCC (real-space correlation) = %.6f\n", rscc_cpu_val);
    std::printf("FSC curve [shell: freq(cyc/box)  res(A)  FSC  nvox]:\n");
    for (std::size_t s = 0; s < fsc_cpu_curve.size(); ++s) {
        if (shell_count[s] == 0) continue;                 // skip empty shells
        const double res = shell_to_res(static_cast<int>(s), d.n, d.voxel_angstrom);
        if (std::isinf(res))
            std::printf("  shell %2zu:   %2zu     inf   %+.6f  %lld\n",
                        s, s, fsc_cpu_curve[s], shell_count[s]);
        else
            std::printf("  shell %2zu:   %2zu   %6.2f   %+.6f  %lld\n",
                        s, s, res, fsc_cpu_curve[s], shell_count[s]);
    }
    std::printf("resolution @ FSC=0.143 (gold-standard half-map): %.2f A (shell %d)\n",
                res_gold, s_gold);
    std::printf("resolution @ FSC=0.5   (map-vs-model)          : %.2f A (shell %d)\n",
                res_half, s_half);
    std::printf("RESULT: %s (GPU cuFFT validation matches CPU DFT: RSCC<=1e-9, FSC<=1e-4)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%lld voxels)\n", path.c_str(), d.voxels());
    std::fprintf(stderr, "[timing] CPU naive DFT+RSCC: %.3f ms   GPU cuFFT+RSCC: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- the naive DFT is O(N^2) per axis; "
                         "cuFFT is O(N log N). The gap explodes with map size.\n");
    std::fprintf(stderr, "[verify] RSCC abs err = %.3e (tol %.1e); worst FSC abs err = %.3e (tol %.1e)\n",
                 rscc_err, RSCC_TOL, fsc_worst, FSC_TOL);

    return pass ? 0 : 1;
}
