// ===========================================================================
// src/main.cu  --  Entry point: ART pipeline on CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 5.5 : Deformable Dose Accumulation & Adaptive Radiotherapy
//               (reduced-scope 2-D teaching version)
//
// 5-step shape (the shape EVERY project in this repo follows):
//   1. Load the ART case (planning image, daily image, planning + daily dose).
//   2. CPU reference: Demons DIR -> DVF; warp+accumulate the delivered dose over
//      NFRACTIONS fractions; build the dose-volume histogram (DVH). All serial,
//      all trusted.
//   3. GPU: the same pipeline (kernels.cu), reusing the identical per-voxel math.
//   4. VERIFY: GPU DVF matches CPU (1e-3 px), GPU accumulated dose matches CPU
//      (1e-9 Gy), GPU DVH matches CPU EXACTLY (integer counts).
//   5. REPORT: deterministic dose statistics + the DVH to stdout; timing/stderr.
//
// WHY THE STDOUT IS DETERMINISTIC
//   Every number printed to stdout comes from the CPU pipeline (serial double
//   precision -> bit-identical every run) at fixed precision, or from the integer
//   DVH counts (exact). The GPU results are used ONLY for the PASS/FAIL verdicts
//   (thresholded comparisons), never printed as raw floats. So demo/run_demo can
//   diff stdout against expected_output.txt. Timings and the exact GPU-vs-CPU
//   differences go to stderr (shown, not diffed).
//
// THE TEACHING POINT (see THEORY §1)
//   The anatomy moved between the plan and delivery. If you naively add each
//   fraction's dose in the delivery frame and pretend it lands where the plan
//   intended (RIGID accumulation), you mis-estimate the dose to the target/organs.
//   DEFORMABLE accumulation warps each fraction's dose back through the DVF into
//   the common planning frame first -- the anatomically-correct total. We report
//   both so the learner SEES the difference (max-dose discrepancy).
//
// Code tour: start here, then demons.h + dose.h (the per-voxel physics),
// kernels.cu, and reference_cpu.cpp for the serial baseline. See ../THEORY.md.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // register_gpu, accumulate_dose_gpu, ArtCase, params
#include "reference_cpu.h"    // load_case, register_cpu, warp/accumulate/DVH, sums
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "5.5";
static const char* PROJECT_NAME = "Deformable Dose Accumulation & Adaptive Radiotherapy";

// How many (identical) daily fractions to accumulate. A real course is ~30
// fractions; we sum a handful so the accumulated dose exceeds one fraction's peak
// yet stays under the DVH ceiling (DVH_MAX). Deterministic (fixed constant).
static const int NFRACTIONS = 3;

// --- Verification tolerances (documented; PATTERNS.md §4) ------------------
// DIR is a long iterative solver: over ~120 iterations the GPU's fused-multiply-
// add and the host compiler diverge by ~1e-5 even in double precision. Our
// displacements are O(5) px, so 1e-3 px is far below anything visible ("the same
// deformation") yet honest about FP drift. We do NOT claim bit-identical fields.
static constexpr double DVF_TOL_PX  = 1.0e-3;
// The dose warp is a SINGLE bilinear gather from the (matching) DVF, then a few
// integer-count additions -- a short computation, so the accumulated dose matches
// to near machine precision. 1e-9 Gy is generous headroom.
static constexpr double DOSE_TOL_GY = 1.0e-9;

// The Demons run parameters (same shape as the DIR flagship 4.8). Chosen so the
// demo converges visibly in a fraction of a second on the tiny sample.
static DemonsParams make_params(int nx, int ny) {
    DemonsParams P;
    P.nx      = nx;
    P.ny      = ny;
    P.iters   = 120;      // outer iterations (enough to register the sample)
    P.sigma   = 1.5;      // Gaussian regularization width, in pixels
    P.radius  = 5;        // kernel half-width (>= ceil(3*sigma)=5): captures ~99%
    P.epsilon = 1.0e-6;   // force-denominator floor (avoids 0/0 in flat regions)
    return P;
}

int main(int argc, char** argv) {
    // ---- 1. Load the ART case ---------------------------------------------
    const std::string path =
        (argc > 1) ? argv[1] : "data/sample/art_case.txt";
    ArtCase c;
    try {
        c = load_case(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const DemonsParams P = make_params(c.nx, c.ny);
    const std::size_t N  = static_cast<std::size_t>(c.nx) * c.ny;

    // ---- 2. CPU reference pipeline (timed) --------------------------------
    util::CpuTimer cpu_timer;
    cpu_timer.start();

    // (2a) Register daily -> planning: recover the DVF.
    std::vector<double> ux_cpu, uy_cpu;
    register_cpu(c, P, ux_cpu, uy_cpu);

    // (2b) Warp the delivered daily dose into the planning frame (one fraction).
    std::vector<double> warped_cpu;
    warp_dose_cpu(c, ux_cpu, uy_cpu, warped_cpu);

    // (2c) Accumulate NFRACTIONS deformed fractions -> the total delivered dose.
    std::vector<double> total_cpu;
    for (int f = 0; f < NFRACTIONS; ++f)
        accumulate_cpu(total_cpu, warped_cpu);

    // (2d) Dose-volume histogram of the accumulated dose (integer counts).
    std::vector<unsigned> dvh_cpu = build_dvh_cpu(total_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // (2e) The RIGID (no-DIR) accumulation for contrast: just sum the delivered
    //      dose in place, ignoring that the anatomy moved. This is what a naive
    //      pipeline would report; comparing its hot-spot to the deformable one
    //      shows why DIR matters (THEORY §1). Deterministic, CPU-only.
    std::vector<double> total_rigid(N, 0.0);
    for (int f = 0; f < NFRACTIONS; ++f)
        for (std::size_t i = 0; i < N; ++i)
            total_rigid[i] += c.daily_dose[i];

    // ---- 3. GPU pipeline (loops timed inside the wrappers) ----------------
    std::vector<double> ux_gpu, uy_gpu;
    float gpu_dir_ms = 0.0f;
    register_gpu(c, P, ux_gpu, uy_gpu, &gpu_dir_ms);

    std::vector<double> total_gpu;
    std::vector<unsigned> dvh_gpu;
    float gpu_dose_ms = 0.0f;
    accumulate_dose_gpu(c, ux_gpu, uy_gpu, NFRACTIONS, total_gpu, dvh_gpu, &gpu_dose_ms);

    // ---- 4. Verify (three independent checks) -----------------------------
    // (4a) DVF: GPU field agrees with CPU field within the FP-drift tolerance.
    double worst_dvf = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        worst_dvf = std::fmax(worst_dvf, std::fabs(ux_cpu[i] - ux_gpu[i]));
        worst_dvf = std::fmax(worst_dvf, std::fabs(uy_cpu[i] - uy_gpu[i]));
    }
    const bool pass_dvf = worst_dvf <= DVF_TOL_PX;

    // (4b) Accumulated dose: GPU total agrees with CPU total (near machine prec).
    double worst_dose = 0.0;
    for (std::size_t i = 0; i < N; ++i)
        worst_dose = std::fmax(worst_dose, std::fabs(total_cpu[i] - total_gpu[i]));
    const bool pass_dose = worst_dose <= DOSE_TOL_GY;

    // (4c) DVH: integer histograms must match EXACTLY (deterministic reduction).
    bool pass_dvh = (dvh_cpu.size() == dvh_gpu.size());
    if (pass_dvh)
        for (std::size_t b = 0; b < dvh_cpu.size(); ++b)
            if (dvh_cpu[b] != dvh_gpu[b]) { pass_dvh = false; break; }

    const bool pass = pass_dvf && pass_dose && pass_dvh;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) ----------
    // All numbers below come from the CPU field / integer DVH (bit-stable).
    const double d_sum   = dose_sum(total_cpu);   // total deposited (Gy * voxels)
    const double d_hot   = dose_max(total_cpu);   // accumulated hot-spot (Gy)
    const double d_rigid = dose_max(total_rigid); // naive-rigid hot-spot (Gy)
    double disp_sum = 0.0;                         // mean |DVF| magnitude (px)
    for (std::size_t i = 0; i < N; ++i)
        disp_sum += std::sqrt(ux_cpu[i]*ux_cpu[i] + uy_cpu[i]*uy_cpu[i]);
    const double mean_disp = disp_sum / static_cast<double>(N);

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("ART pipeline: %dx%d grid | DIR %d iters, sigma=%.2f px | %d fractions\n",
                c.nx, c.ny, P.iters, P.sigma, NFRACTIONS);
    std::printf("DIR: mean |displacement| = %.4f px\n", mean_disp);
    std::printf("accumulated dose: sum = %.4f Gy*vox, hot-spot = %.4f Gy\n",
                d_sum, d_hot);
    std::printf("rigid (no-DIR) hot-spot = %.4f Gy  (deformable - rigid = %+.4f Gy)\n",
                d_rigid, d_hot - d_rigid);

    // A compact DVH: print every 4th bin's cumulative volume fraction (>= dose).
    // Cumulative = fraction of voxels receiving AT LEAST the bin's lower dose,
    // the standard clinical reading. Computed from the integer counts -> exact.
    std::printf("cumulative DVH (dose_Gy: vol%%):");
    unsigned total_vox = 0;
    for (unsigned v : dvh_cpu) total_vox += v;
    for (int b = 0; b < DVH_BINS; b += 4) {
        unsigned at_least = 0;                        // voxels in bins >= b
        for (int k = b; k < DVH_BINS; ++k) at_least += dvh_cpu[k];
        const double dose_lo = (double)b / (double)DVH_BINS * DVH_MAX;
        const double volpct  = 100.0 * (double)at_least / (double)total_vox;
        std::printf("  %.2f:%.1f", dose_lo, volpct);
    }
    std::printf("\n");
    std::printf("RESULT: %s (DVF<=1e-3px, dose<=1e-9Gy, DVH exact)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) -----------------
    std::fprintf(stderr, "[data]   source: %s  (%d x %d voxels)\n",
                 path.c_str(), c.nx, c.ny);
    std::fprintf(stderr, "[timing] CPU pipeline: %.3f ms | GPU DIR: %.3f ms | "
                         "GPU dose: %.3f ms\n", cpu_ms, gpu_dir_ms, gpu_dose_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- the GPU's edge grows with "
                         "volume size; a real 256^3 ART step is ~10^7 voxels, "
                         "where the GPU is essential (catalog: <5 min online).\n");
    std::fprintf(stderr, "[verify] worst DVF diff = %.3e px (tol %.1e) | "
                         "worst dose diff = %.3e Gy (tol %.1e) | DVH %s\n",
                 worst_dvf, DVF_TOL_PX, worst_dose, DOSE_TOL_GY,
                 pass_dvh ? "exact-match" : "MISMATCH");

    return pass ? 0 : 1;
}
