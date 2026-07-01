// ===========================================================================
// src/main.cu  --  Entry point: load QA planes, run CPU + GPU gamma, verify,
//                  print the linac-QA scorecard
// ---------------------------------------------------------------------------
// Project 5.8 : Linac QA & Machine Performance Assessment  (catalog ID 5.8)
//
// THE 5-STEP SHAPE (every project in this repo follows it)
//   1. Load the problem: the planned (reference) + measured (EPID/portal) dose
//      planes and the QA tolerances (data/sample, or a built-in synthetic set).
//   2. CPU reference: gamma map + pass rate + flatness/symmetry metrics.
//   3. GPU: the same gamma map, one thread per measured pixel (kernels.cu).
//   4. VERIFY: assert the GPU gamma map equals the CPU map (tol = 0 -- they call
//      the identical __host__ __device__ core, so agreement is exact).
//   5. REPORT: a deterministic QA scorecard to stdout; timings to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Timings (which vary run to run) go to STDERR,
//   which the demo shows but does not diff (PATTERNS.md §3).
//
// READ THIS FIRST in the code tour, then kernels.cuh -> kernels.cu (the GPU
// gather), gamma.h (the shared math), and reference_cpu.cpp (the baseline).
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // gamma_map_gpu (GPU path)
#include "reference_cpu.h"    // load_qa, gamma_map_cpu, metrics (CPU baseline)
#include "util/io.hpp"        // util::CpuTimer, util::max_abs_err

// Program identity. MUST stay in sync with demo/expected_output.txt.
static const char* PROJECT_ID   = "5.8";
static const char* PROJECT_NAME = "Linac QA & Machine Performance Assessment";

// Verification tolerance. The GPU kernel and the CPU reference both call the
// SAME gamma_value_at() with the same float operations in the same order, so
// the two gamma maps are bit-identical -> we demand EXACT agreement (0.0).
// (PATTERNS.md §4: "exact" is appropriate when identical ops run on both sides.)
static constexpr double TOLERANCE = 0.0;

// TG-218 per-beam IMRT QA universal action limit: gamma pass rate >= 95% at
// 3%/3mm is the clinical "action level" (below it, the plan is investigated).
static constexpr float TG218_PASS_RATE = 95.0f;

// ---------------------------------------------------------------------------
// make_synthetic_qa: a tiny built-in QA problem used when no data file is given.
//   Builds a 24x24 open-field dose plane (a smooth "flat top" with penumbra),
//   copies it as the reference, then perturbs the measured copy with a small,
//   deterministic, physically-plausible error (a slight output scaling + a
//   1-pixel shift on one side) so the gamma map is non-trivial but MOSTLY
//   passing -- exactly the shape of a real, healthy daily-QA result.
//   The SAME construction lives in scripts/make_synthetic.py (which writes the
//   committed sample), so the file path and this fallback agree.
// ---------------------------------------------------------------------------
static void make_synthetic_qa(QAProblem& q) {
    q.nx = 24; q.ny = 24;
    q.spacing_mm = 2.0f;      // 2 mm pixels -> a 48 mm plane
    q.dd_percent = 3.0f;      // 3% dose-difference criterion
    q.dta_mm     = 3.0f;      // 3 mm distance-to-agreement
    q.norm_dose  = 0.0f;      // 0 => normalise to the reference max (set in loader)

    const int nx = q.nx, ny = q.ny;
    q.ref.assign((size_t)nx * ny, 0.0f);
    q.meas.assign((size_t)nx * ny, 0.0f);

    const float cx = 0.5f * (nx - 1);   // plane centre (columns)
    const float cy = 0.5f * (ny - 1);   // plane centre (rows)
    const float half_field = 8.0f;      // flat-top half-width, in pixels
    const float penumbra   = 2.0f;      // edge softness, in pixels

    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            // Separable "flat top with soft edges": 1 inside the field, ramping
            // to 0 across the penumbra. Deterministic, no RNG.
            auto edge = [&](float pos, float c) -> float {
                const float d = (pos - c) >= 0 ? (pos - c) : (c - pos);   // |pos-c|
                if (d <= half_field)              return 1.0f;            // flat top
                if (d >= half_field + penumbra)   return 0.0f;            // outside
                return 1.0f - (d - half_field) / penumbra;               // ramp
            };
            const float prof = 100.0f * edge((float)x, cx) * edge((float)y, cy);
            q.ref[(size_t)y * nx + x] = prof;

            // Measured = planned with a small, realistic machine error:
            //   * 1% low output overall (a common daily drift), and
            //   * the right half shifted "hotter" by 2% (a mild asymmetry).
            float meas = prof * 0.99f;
            if (x > (int)cx) meas *= 1.02f;
            q.meas[(size_t)y * nx + x] = meas;
        }
    }
}

int main(int argc, char** argv) {
    // ---- 1. Load ----------------------------------------------------------
    QAProblem q;
    const char* source = "synthetic (built-in)";
    if (argc > 1) {
        try {
            q = load_qa(argv[1]);
            source = argv[1];
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[warn] could not load '%s' (%s); using built-in synthetic.\n",
                         argv[1], e.what());
            make_synthetic_qa(q);
            // Re-apply the loader's norm_dose default (max of ref) for parity.
            float mx = 0.0f; for (float v : q.ref) if (v > mx) mx = v;
            q.norm_dose = mx;
        }
    } else {
        make_synthetic_qa(q);
        float mx = 0.0f; for (float v : q.ref) if (v > mx) mx = v;
        q.norm_dose = mx;
    }

    // Derive the gamma tolerances (shared by CPU + GPU) and the low-dose cut.
    const GammaParams p = make_gamma_params(q);
    // Standard low-dose threshold: ignore pixels below 10% of the norm dose
    // (near-zero background where gamma is meaningless). TG-218 §"analysis".
    const float dose_threshold = 0.10f * q.norm_dose;

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<float> gamma_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    gamma_map_cpu(q, p, gamma_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    int cpu_eval = 0, cpu_pass = 0;
    const float cpu_rate = gamma_pass_rate(q, gamma_cpu, p.pass_gamma, dose_threshold,
                                           cpu_eval, cpu_pass);
    const QAMetrics qm = compute_qa_metrics(q);

    // ---- 3. GPU gamma map (kernel timed inside the wrapper) ---------------
    std::vector<float> gamma_gpu;
    float gpu_kernel_ms = 0.0f;
    gamma_map_gpu(q, p, gamma_gpu, &gpu_kernel_ms);

    // Pass rate recomputed from the GPU map (must equal the CPU rate exactly).
    int gpu_eval = 0, gpu_pass = 0;
    const float gpu_rate = gamma_pass_rate(q, gamma_gpu, p.pass_gamma, dose_threshold,
                                           gpu_eval, gpu_pass);

    // ---- 4. Verify: GPU gamma map == CPU gamma map (exact) ----------------
    const double err = util::max_abs_err(gamma_cpu, gamma_gpu);
    const bool pass = (err <= TOLERANCE) && (gpu_pass == cpu_pass) && (gpu_eval == cpu_eval);

    // Also find the worst (largest-gamma) evaluated pixel: the "hottest" QA
    // failure a physicist would inspect first. Deterministic scan (first max).
    int worst_idx = -1; float worst_g = -1.0f;
    for (size_t i = 0; i < gamma_gpu.size(); ++i) {
        if (q.meas[i] < dose_threshold) continue;
        if (gamma_gpu[i] > worst_g) { worst_g = gamma_gpu[i]; worst_idx = (int)i; }
    }
    const int worst_x = (worst_idx >= 0) ? worst_idx % q.nx : -1;
    const int worst_y = (worst_idx >= 0) ? worst_idx / q.nx : -1;

    // ---- 5a. Deterministic QA scorecard -> STDOUT (diffed by the demo) ----
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("plane: %d x %d px  spacing=%.2f mm  norm_dose=%.2f\n",
                q.nx, q.ny, q.spacing_mm, q.norm_dose);
    std::printf("gamma criteria: %.1f%% / %.1f mm  (search radius=%d px, low-dose cut=%.0f%%)\n",
                q.dd_percent, q.dta_mm, p.search_radius, 100.0f * dose_threshold / q.norm_dose);
    std::printf("gamma pass rate = %.2f%%  (%d/%d evaluated points pass, gamma<=%.1f)\n",
                gpu_rate, gpu_pass, gpu_eval, p.pass_gamma);
    std::printf("worst gamma = %.4f at pixel (%d,%d)\n", worst_g, worst_x, worst_y);
    std::printf("TG-218 action limit (>=%.0f%%): %s\n",
                TG218_PASS_RATE, (gpu_rate >= TG218_PASS_RATE) ? "MEETS" : "BELOW");
    std::printf("machine QA (measured plane, central axis):\n");
    std::printf("  CAX output   = %.2f\n",      qm.cax_dose);
    std::printf("  field width  = %.2f mm (FWHM)\n", qm.field_width_mm);
    std::printf("  flatness     = %.3f %%\n",   qm.flatness_pct);
    std::printf("  symmetry     = %.3f %%\n",   qm.symmetry_pct);
    std::printf("RESULT: %s (GPU gamma map matches CPU exactly, tol=0)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) -----------------
    std::fprintf(stderr, "[data]   source: %s  (%d x %d dose planes)\n", source, q.nx, q.ny);
    std::fprintf(stderr, "[timing] CPU gamma map: %.3f ms   GPU gamma map: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- the GPU's edge over the CPU grows "
                         "with plane size (clinical EPID frames are ~1024^2, far larger than "
                         "this sample).\n");
    std::fprintf(stderr, "[verify] max_abs_err(gamma) = %.3e  (tolerance %.1e);  "
                         "CPU pass %d/%d, GPU pass %d/%d\n",
                 err, TOLERANCE, cpu_pass, cpu_eval, gpu_pass, gpu_eval);

    // Exit code feeds the demo's pass/fail gate.
    return pass ? 0 : 1;
}
