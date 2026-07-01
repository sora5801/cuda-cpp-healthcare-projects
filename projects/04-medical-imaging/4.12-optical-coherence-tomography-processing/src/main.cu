// ===========================================================================
// src/main.cu  --  Entry point: load B-scan, reconstruct on CPU+GPU, verify
// ---------------------------------------------------------------------------
// Project 4.12 : Optical Coherence Tomography Processing (SD-OCT reconstruction)
//
// WHAT THIS FILE DOES  (the 5-step shape EVERY project in this repo follows)
//   1. Load the raw B-scan (data/sample, or a path given on argv[1]).
//   2. CPU reference: reconstruct via naive DFT (reference_cpu.cpp) -> trusted.
//   3. GPU: reconstruct via custom kernels + cuFFT (kernels.cu)     -> taught.
//   4. VERIFY two ways:
//        (a) the DETERMINISTIC integer result -- per-A-scan peak-depth index --
//            must match EXACTLY between CPU and GPU (order-independent argmax);
//        (b) the normalised image agrees within a documented FLOAT tolerance
//            (cuFFT is single precision; the naive DFT is double).
//   5. REPORT: deterministic result (peak depths + an ASCII B-scan) to stdout;
//      timings and the float error to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Anything that varies run-to-run (timings, the float
//   error magnitude) goes to STDERR, which the demo shows but does not diff.
//
// READ THIS FIRST in the code tour, then oct_core.h -> kernels.cuh -> kernels.cu,
// and reference_cpu.cpp for the baseline. See ../THEORY.md for the "why".
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // reconstruct_gpu (GPU path), OctBscan
#include "reference_cpu.h"    // load_bscan, reconstruct_cpu, peak_depths
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "4.12";
static const char* PROJECT_NAME = "Optical Coherence Tomography Processing (SD-OCT)";

// FLOAT-image tolerance. cuFFT works in single precision and reorders additions
// via its radix decomposition, while the CPU reference sums a naive DFT in
// double. Over N ~ 10^3 terms that diverges at the ~1e-5..1e-6 level even though
// the algorithms are mathematically identical -- an honest, documented tolerance
// (PATTERNS.md #4). The DETERMINISTIC result we diff is the INTEGER peak depth,
// which is exact; this float check is a second, stronger correctness guard.
static constexpr double IMAGE_ATOL = 2.0e-4;

// ASCII grey-ramp for the tiny B-scan preview, dark -> bright. Purely a
// human-readable rendering of the normalised power (deterministic given the
// deterministic image), so it is safe to put on stdout.
static const char* RAMP = " .:-=+*#%@";   // 10 levels

int main(int argc, char** argv) {
    // ---- 1. Load the raw B-scan --------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/oct_bscan.txt";
    OctBscan b;
    try {
        b = load_bscan(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int nd = oct_depth_count(b.n_spec);   // depths kept = N/2

    // ---- 2. CPU reference reconstruction (timed) ---------------------------
    std::vector<double> image_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    reconstruct_cpu(b, image_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU reconstruction (kernels + cuFFT, timed inside the wrapper) --
    std::vector<double> image_gpu;
    float gpu_kernel_ms = 0.0f;
    reconstruct_gpu(b, image_gpu, &gpu_kernel_ms);

    // ---- 4a. Deterministic integer result: per-A-scan peak depth -----------
    std::vector<int> peak_cpu, peak_gpu;
    peak_depths(image_cpu, b.n_ascan, nd, peak_cpu);
    peak_depths(image_gpu, b.n_ascan, nd, peak_gpu);
    bool peaks_match = true;
    for (int a = 0; a < b.n_ascan; ++a)
        if (peak_cpu[a] != peak_gpu[a]) { peaks_match = false; break; }

    // ---- 4b. Float-image agreement (second, stronger check) ----------------
    double worst = 0.0;
    for (std::size_t i = 0; i < image_cpu.size(); ++i) {
        const double d = std::fabs(image_cpu[i] - image_gpu[i]);
        if (d > worst) worst = d;
    }
    const bool image_close = worst <= IMAGE_ATOL;
    const bool pass = peaks_match && image_close;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("B-scan: %d A-scans x %d spectral samples  (N/2=%d depths)  "
                "dispersion a2=%.1f a3=%.1f\n",
                b.n_ascan, b.n_spec, nd, b.a2, b.a3);

    // Per-A-scan strongest-reflector depth (the recovered surface/peak). Integer,
    // exact, order-independent -> byte-identical CPU vs GPU and run to run.
    std::printf("peak depth per A-scan (bin index of strongest reflector):\n ");
    for (int a = 0; a < b.n_ascan; ++a) std::printf(" %3d", peak_gpu[a]);
    std::printf("\n");

    // A compact ASCII B-scan: rows are depth (top = surface), columns are A-scans.
    // We show the top `SHOW_DEPTH` depth bins so the layered structure is visible.
    const int SHOW_DEPTH = nd < 24 ? nd : 24;
    std::printf("reconstructed B-scan (depth down, A-scans across; dispersion-compensated):\n");
    for (int z = 0; z < SHOW_DEPTH; ++z) {
        std::printf("  ");
        for (int a = 0; a < b.n_ascan; ++a) {
            const double v = image_gpu[static_cast<std::size_t>(a) * nd + z];  // 0..1
            int level = static_cast<int>(v * 9.999);      // map to 0..9 grey ramp
            if (level < 0) level = 0; else if (level > 9) level = 9;
            std::putchar(RAMP[level]);
        }
        std::printf("\n");
    }
    std::printf("RESULT: %s (GPU peak depths match CPU exactly; images agree within atol=2e-4)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%d x %d)\n", path.c_str(), b.n_ascan, b.n_spec);
    std::fprintf(stderr, "[timing] CPU naive-DFT reconstruction: %.3f ms   "
                         "GPU kernels+cuFFT: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- the naive DFT is O(N^2) per A-scan; "
                         "cuFFT is O(N log N) and batches all A-scans. The gap explodes "
                         "with N and A-scan count (real B-scans are 2048x2048).\n");
    std::fprintf(stderr, "[verify] peak-depth exact match: %s   worst image |CPU-GPU| = %.3e "
                         "(atol %.1e)\n", peaks_match ? "yes" : "NO", worst, IMAGE_ATOL);

    return pass ? 0 : 1;
}
