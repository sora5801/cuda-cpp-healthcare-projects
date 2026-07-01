// ===========================================================================
// src/main.cu  --  Entry point: load dose maps, run CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 5.9 -- Gamma-Index Dose Comparison
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the problem (two dose maps from data/sample, or a synthetic
//      fallback so the program always runs).
//   2. Compute the CPU reference gamma map (reference_cpu.cpp) -> trusted answer.
//   3. Compute the GPU gamma map          (kernels.cu)          -> the thing taught.
//   4. VERIFY: assert the GPU map equals the CPU map within tolerance, and that
//      the derived gamma pass-rate matches exactly.
//   5. REPORT: deterministic result (pass-rate, min/max gamma, a small map
//      slice) to stdout; timings to stderr.
//
//   STDOUT is kept byte-for-byte deterministic so demo/run_demo can diff it
//   against demo/expected_output.txt. Anything that varies run-to-run (timings)
//   goes to STDERR, which the demo shows but does not diff (PATTERNS.md §3).
//
// READ THIS FIRST in the code tour, then kernels.cuh -> kernels.cu, and
// reference_cpu.cpp for the baseline. See ../THEORY.md for the "why".
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "dose_problem.h"    // DoseProblem (the two dose maps + criteria)
#include "kernels.cuh"       // gamma_map_gpu (GPU path)
#include "reference_cpu.h"   // gamma_map_cpu (CPU baseline)
#include "util/io.hpp"       // util::CpuTimer, util::read_floats, util::max_abs_err

// These two tokens identify the program; they MUST stay in sync with
// demo/expected_output.txt (which is captured from a real run of this binary).
static const char* PROJECT_ID   = "5.9";
static const char* PROJECT_NAME = "Gamma-Index Dose Comparison";

// Correctness tolerance for the gamma MAP. Because the CPU and GPU call the same
// gamma_core.h math over the same fixed candidate window and reduce with an
// EXACT float min (not a reordered sum), we expect bit-identical results; we
// keep a tiny non-zero epsilon only as insurance against a compiler contracting
// a multiply-add differently on one side (THEORY §6). In practice the observed
// error is exactly 0.
static constexpr double TOLERANCE = 1.0e-6;

// ---------------------------------------------------------------------------
// make_synthetic -- a small, interpretable dose comparison with a KNOWN answer.
//   We build a smooth 2-D dose "hill" (a Gaussian bump, standing in for a
//   treatment field) as the reference, then produce the evaluated map by
//   applying TWO deliberate, localized perturbations whose gamma behavior we can
//   predict (PATTERNS.md §6 -- engineer the sample so the result is meaningful):
//
//     * a small global scaling (+1.5%) everywhere: WITHIN the 3% dose criterion,
//       so those points should PASS (gamma < 1).
//     * a sharp "hot spot" patch near the center where the evaluated dose is
//       boosted by ~12% over a few voxels: OUTSIDE 3%/3mm, so those points
//       should FAIL (gamma > 1).
//
//   The demo therefore shows a high-but-not-perfect pass-rate with a
//   spatially-localized failure region -- exactly the shape of a real IMRT QA
//   result. Everything here is SYNTHETIC and labeled as such.
// ---------------------------------------------------------------------------
static void make_synthetic(DoseProblem& p) {
    p.width  = 32;
    p.height = 32;
    p.spacing_mm = 2.0;              // 2 mm voxels (typical QA array pitch)
    p.dd_percent = 3.0;             // 3% / 3 mm -- the classic clinical criterion
    p.dta_mm     = 3.0;
    p.dose_threshold_frac = 0.10;   // analyze points above 10% of max

    const int W = p.width, H = p.height;
    p.ref.assign((std::size_t)W * H, 0.0f);
    p.eval.assign((std::size_t)W * H, 0.0f);

    const double cx = 0.5 * (W - 1), cy = 0.5 * (H - 1);   // grid center
    const double sigma = 8.0;                              // hill width [voxels]

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const double dx = x - cx, dy = y - cy;
            const double r2 = dx * dx + dy * dy;
            // Reference: a 100-unit Gaussian dose hill.
            const double dose = 100.0 * std::exp(-r2 / (2.0 * sigma * sigma));
            const int idx = y * W + x;
            p.ref[idx]  = (float)dose;

            // Evaluated: reference * 1.015 (a +1.5% global bias, within 3%).
            double e = dose * 1.015;

            // Inject a localized hot spot near the center (a 3x3 patch) that is
            // ~12% high -> outside 3%/3mm -> should FAIL the gamma test.
            if (std::abs(dx - 3.0) <= 1.0 && std::abs(dy + 2.0) <= 1.0) {
                e = dose * 1.12;
            }
            p.eval[idx] = (float)e;
        }
    }
}

// ---------------------------------------------------------------------------
// load_sample -- parse a dose-comparison file. Layout (whitespace-separated):
//     line/tokens 1-2 : width height
//     next 4 tokens   : spacing_mm dd_percent dta_mm dose_threshold_frac
//     next W*H tokens : reference dose map, row-major
//     next W*H tokens : evaluated dose map, row-major
//   Returns false if the file is missing/short so the caller can fall back to
//   the synthetic problem. (This mirrors the scaffold's read_floats approach.)
// ---------------------------------------------------------------------------
static bool load_sample(const std::string& path, DoseProblem& p) {
    std::vector<float> v;
    try {
        v = util::read_floats(path);
    } catch (const std::exception&) {
        return false;   // file not found -> caller uses synthetic data
    }
    if (v.size() < 6) return false;
    p.width  = static_cast<int>(v[0]);
    p.height = static_cast<int>(v[1]);
    p.spacing_mm          = v[2];
    p.dd_percent          = v[3];
    p.dta_mm              = v[4];
    p.dose_threshold_frac = v[5];
    const std::size_t n = static_cast<std::size_t>(p.width) * p.height;
    if (p.width <= 0 || p.height <= 0) return false;
    if (v.size() < 6 + 2 * n) return false;
    p.ref.assign (v.begin() + 6,      v.begin() + 6 + n);
    p.eval.assign(v.begin() + 6 + n,  v.begin() + 6 + 2 * n);
    return true;
}

// ---------------------------------------------------------------------------
// gamma_stats -- turn a gamma map into the clinical summary numbers.
//   Only voxels with gamma > 0 were ANALYZED (below-threshold background was set
//   to exactly 0 by both implementations), so we count over gamma > 0.
//   Everything here is INTEGER counting -> perfectly deterministic (PATTERNS §3).
//
//   Fills: analyzed (points scored), passed (gamma <= 1), and the pass-rate in
//   tenths-of-a-percent as an INTEGER (so the printed rate never wobbles in the
//   last decimal). Also returns min/max gamma scaled to milli-units (x1000, as
//   ints) for a deterministic headline without float formatting drift.
// ---------------------------------------------------------------------------
static void gamma_stats(const std::vector<float>& g,
                        long& analyzed, long& passed,
                        long& rate_milli, long& gmin_milli, long& gmax_milli) {
    analyzed = passed = 0;
    double gmin = 1.0e30, gmax = 0.0;
    for (float gv : g) {
        if (gv <= 0.0f) continue;             // background, not analyzed
        ++analyzed;
        if (gv <= 1.0f) ++passed;             // gamma <= 1 == passes
        if (gv < gmin) gmin = gv;
        if (gv > gmax) gmax = gv;
    }
    // Pass-rate in per-mille (0..1000), rounded deterministically.
    rate_milli = analyzed ? (long)std::lround(1000.0 * (double)passed / analyzed) : 0;
    gmin_milli = analyzed ? (long)std::lround(1000.0 * gmin) : 0;
    gmax_milli = analyzed ? (long)std::lround(1000.0 * gmax) : 0;
}

int main(int argc, char** argv) {
    // ---- 1. Load the problem ------------------------------------------------
    DoseProblem prob;
    const char* source = "synthetic (built-in)";
    if (argc > 1 && load_sample(argv[1], prob)) {
        source = argv[1];
    } else {
        make_synthetic(prob);
    }

    // ---- 2. CPU reference (timed) ------------------------------------------
    std::vector<float> gamma_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    gamma_map_cpu(prob, gamma_cpu);
    double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU result (kernel timed inside the wrapper) -------------------
    std::vector<float> gamma_gpu;
    float gpu_kernel_ms = 0.0f;
    gamma_map_gpu(prob, gamma_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ----------------------------------------------------------
    // (a) the gamma MAPS agree voxel-for-voxel within tolerance;
    double err = util::max_abs_err(gamma_cpu, gamma_gpu);
    // (b) the derived pass-rate STATISTICS agree exactly (integer counts).
    long a_cpu, p_cpu, r_cpu, gmin_c, gmax_c;
    long a_gpu, p_gpu, r_gpu, gmin_g, gmax_g;
    gamma_stats(gamma_cpu, a_cpu, p_cpu, r_cpu, gmin_c, gmax_c);
    gamma_stats(gamma_gpu, a_gpu, p_gpu, r_gpu, gmin_g, gmax_g);
    bool stats_match = (a_cpu == a_gpu) && (p_cpu == p_gpu) && (r_cpu == r_gpu);
    bool pass = (err <= TOLERANCE) && stats_match;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    // We report from the CPU numbers (identical to the GPU when pass==true).
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("grid: %d x %d voxels @ %.1f mm   criterion: %.0f%%/%.0f mm   "
                "low-dose cutoff: %.0f%%\n",
                prob.width, prob.height, prob.spacing_mm,
                prob.dd_percent, prob.dta_mm, prob.dose_threshold_frac * 100.0);
    std::printf("analyzed points : %ld\n", a_cpu);
    std::printf("passing (g<=1)  : %ld\n", p_cpu);
    std::printf("gamma pass-rate : %ld.%01ld %%\n", r_cpu / 10, r_cpu % 10);
    std::printf("gamma min / max : %ld.%03ld / %ld.%03ld\n",
                gmin_c / 1000, gmin_c % 1000, gmax_c / 1000, gmax_c % 1000);

    // A tiny deterministic slice of the gamma map: the center row, printed as
    // integer milli-gamma (x1000) so there is no float-formatting drift. This
    // lets the learner SEE the localized failure (values > 1000) next to the
    // passing background (values < 1000).
    int mid = prob.height / 2;
    int c0 = prob.width / 2 - 4;
    if (c0 < 0) c0 = 0;
    int c1 = c0 + 8;
    if (c1 > prob.width) c1 = prob.width;
    std::printf("gamma[row %d, cols %d..%d] (x1000):", mid, c0, c1 - 1);
    for (int x = c0; x < c1; ++x) {
        long mg = (long)std::lround(1000.0 * gamma_cpu[(std::size_t)mid * prob.width + x]);
        std::printf(" %ld", mg);
    }
    std::printf("\n");

    std::printf("RESULT: %s (GPU gamma map matches CPU within tol=%.0e)\n",
                pass ? "PASS" : "FAIL", TOLERANCE);

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s\n", source);
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- tiny grids are dominated "
                         "by launch/copy overhead; the GPU's edge grows with grid size.\n");
    std::fprintf(stderr, "[verify] max_abs_err(gamma map) = %.6e  (tolerance %.1e)\n",
                 err, TOLERANCE);
    std::fprintf(stderr, "[verify] stats match (analyzed/passed/rate): %s\n",
                 stats_match ? "yes" : "NO");

    // Exit code feeds the demo's pass/fail gate.
    return pass ? 0 : 1;
}
