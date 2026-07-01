// ===========================================================================
// src/main.cu  --  Entry point: simulate tumor growth + treatment, verify, report
// ---------------------------------------------------------------------------
// Project 6.8 : Tumor Growth & Treatment-Response Modeling
//
// WHAT THIS FILE DOES  (the 5-step shape EVERY project in this repo follows)
//   1. Load params + build the seeded tumor (data/sample + init_field).
//   2. CPU reference simulation (reference_cpu.cpp) -- the trusted baseline.
//   3. GPU simulation (kernels.cu) -- identical per-cell stencil (tumor.h).
//   4. VERIFY: the final density fields match within a documented FP tolerance.
//   5. REPORT: deterministic tumor-burden / treatment-response metrics.
//
//   To make the TREATMENT visible, we run the schedule from the sample TWICE:
//   once with the fractions ON (as given) and once as an untreated CONTROL
//   (n_fractions forced to 0). The difference in final tumor burden is the
//   modelled treatment response. Both runs are CPU-vs-GPU verified.
//
//   STDOUT is deterministic (diffed by demo/run_demo). Timing -> STDERR.
//
// Code tour: start here, then tumor.h (the physics), kernels.cu, reference_cpu.cpp.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // simulate_gpu, TumorParams, tumor_grow_kernel
#include "reference_cpu.h"    // load_tumor, init_field, simulate_cpu, lq_survival
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "6.8";
static const char* PROJECT_NAME = "Tumor Growth & Treatment-Response Modeling";

// Verification tolerance. Both paths run the SAME double-precision math from
// tumor.h, but over thousands of nonlinear steps the GPU's fused multiply-add
// (FMA) and the host compiler contract the diffusion+reaction expressions
// slightly differently, so the fields drift at the ~1e-6 level even in double
// precision (docs/PATTERNS.md section 4, "long iterative solvers"). Densities are
// O(1), so 1e-6 is physically negligible -- the SAME tumor. We do NOT claim
// bit-identity; we verify to a small physical tolerance and say so.
static constexpr double TOLERANCE = 1.0e-6;

// Summary statistics of a density field, computed deterministically.
struct FieldStats {
    double burden;   // total tumor "mass" = sum of u over all cells (cell units)
    double max_u;    // peak density anywhere
    int    active;   // number of cells with u > 0.5 (the tumor "core" area)
    double radius;   // effective radius [mm] of the active core (from its area)
};

// Compute the summary statistics above from a final field.
static FieldStats analyze(const TumorParams& P, const std::vector<double>& u) {
    FieldStats s{0.0, 0.0, 0, 0.0};
    for (int i = 0; i < P.nx * P.ny; ++i) {
        s.burden += u[i];
        s.max_u = std::fmax(s.max_u, u[i]);
        if (u[i] > 0.5) ++s.active;
    }
    // Effective radius from the active-core area A = pi r^2  ->  r = sqrt(A/pi).
    // Area of one cell is dx^2, so core area = active * dx^2.
    const double area = s.active * P.dx * P.dx;
    s.radius = std::sqrt(area / 3.14159265358979323846);
    return s;
}

// Run ONE scenario (CPU + GPU) and verify they agree. Returns the GPU field
// stats through `out_stats` and the worst |CPU-GPU| diff through `out_worst`.
// `label` is only for the stderr timing line. Returns true on agreement.
static bool run_scenario(const char* label, const TumorParams& P,
                         const std::vector<double>& u0,
                         FieldStats* out_stats, double* out_worst,
                         double* cpu_ms, float* gpu_ms) {
    // CPU reference (timed).
    std::vector<double> u_cpu = u0;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    simulate_cpu(P, u_cpu);
    *cpu_ms = cpu_timer.stop_ms();

    // GPU (loop timed with CUDA events inside simulate_gpu).
    std::vector<double> u_gpu = u0;
    *gpu_ms = 0.0f;
    simulate_gpu(P, u_gpu, gpu_ms);

    // Verify: worst per-cell density difference.
    double worst = 0.0;
    for (int i = 0; i < P.nx * P.ny; ++i)
        worst = std::fmax(worst, std::fabs(u_cpu[i] - u_gpu[i]));
    *out_worst = worst;
    *out_stats = analyze(P, u_gpu);

    std::fprintf(stderr, "[timing] %-9s CPU: %8.3f ms   GPU: %8.3f ms\n",
                 label, *cpu_ms, static_cast<double>(*gpu_ms));
    return worst <= TOLERANCE;
}

int main(int argc, char** argv) {
    // ---- 1. Load + seed ----------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/tumor_params.txt";
    TumorParams P;
    try {
        P = load_tumor(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    std::vector<double> u0;
    init_field(P, u0);

    // ---- 2-4. Treated scenario (schedule as given) -------------------------
    FieldStats treated{}; double worst_t = 0.0, cpu_t = 0.0; float gpu_t = 0.0f;
    const bool pass_t = run_scenario("treated", P, u0, &treated, &worst_t, &cpu_t, &gpu_t);

    // ---- 2-4. Untreated control (same growth, zero fractions) --------------
    TumorParams Pc = P; Pc.n_fractions = 0;
    FieldStats control{}; double worst_c = 0.0, cpu_c = 0.0; float gpu_c = 0.0f;
    const bool pass_c = run_scenario("control", Pc, u0, &control, &worst_c, &cpu_c, &gpu_c);

    const bool pass = pass_t && pass_c;
    const double worst = std::fmax(worst_t, worst_c);

    // Treatment response = how much smaller the treated tumor's burden is vs the
    // untreated control. Also report the exact per-fraction LQ surviving fraction
    // so the learner can sanity-check the radiobiology by hand.
    const double S_per_fx = lq_survival(P.alpha, P.beta, P.dose);
    const double bed = P.n_fractions * P.dose *
                       (1.0 + P.dose / (P.alpha / P.beta));   // biologically-eff. dose
    const double reduction = (control.burden > 0.0)
        ? 100.0 * (1.0 - treated.burden / control.burden) : 0.0;

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("Fisher-KPP: %dx%d grid, dx=%.3f mm, D=%.4f mm^2/day, rho=%.3f /day, %d steps\n",
                P.nx, P.ny, P.dx, P.D, P.rho, P.steps);
    std::printf("Radiotherapy: %d fractions x %.2f Gy  (alpha=%.3f /Gy, beta=%.4f /Gy^2, a/b=%.1f Gy)\n",
                P.n_fractions, P.dose, P.alpha, P.beta, P.alpha / P.beta);
    std::printf("LQ per-fraction surviving fraction S = %.4f ; BED = %.2f Gy\n", S_per_fx, bed);
    std::printf("control  (untreated): burden=%.4f, max u=%.4f, core cells=%d, core radius=%.3f mm\n",
                control.burden, control.max_u, control.active, control.radius);
    std::printf("treated  (RT on)    : burden=%.4f, max u=%.4f, core cells=%d, core radius=%.3f mm\n",
                treated.burden, treated.max_u, treated.active, treated.radius);
    std::printf("treatment response: tumor burden reduced by %.2f%% vs control\n", reduction);
    // A density profile along the centre row makes the front shape legible.
    std::printf("treated u along center row (8 samples):");
    const int cy = P.ny / 2;
    // Re-run the treated GPU field into a local vector for the profile (cheap,
    // deterministic) so we sample exactly what we reported above.
    {
        std::vector<double> u_prof = u0;
        float ms = 0.0f;
        simulate_gpu(P, u_prof, &ms);
        for (int s = 0; s < 8; ++s) {
            const int x = (s * (P.nx - 1)) / 7;
            std::printf(" %.4f", u_prof[cy * P.nx + x]);
        }
    }
    std::printf("\n");
    std::printf("RESULT: %s (GPU fields match CPU within tol=1.0e-06)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d cells, %d steps)\n",
                 path.c_str(), P.nx * P.ny, P.steps);
    std::fprintf(stderr, "[verify] worst density diff = %.3e  (tolerance %.1e)\n", worst, TOLERANCE);
    std::fprintf(stderr, "[timing] teaching artifact only -- the GPU's edge grows with grid size; "
                         "clinical 3-D models use 256^3-512^3 voxels over many GPUs.\n");

    return pass ? 0 : 1;
}
