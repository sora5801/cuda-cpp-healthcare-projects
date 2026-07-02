// ===========================================================================
// src/main.cu  --  Entry point: run remodeling on CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 6.22 : Bone Remodeling Simulation   (REDUCED-SCOPE teaching version)
//
// 5-step shape (the shape EVERY project in this repo follows):
//   1. Load the remodeling job (data/sample, or a built-in synthetic fallback).
//   2. CPU reference remodeling (reference_cpu.cpp)          -> trusted answer.
//   3. GPU remodeling (kernels.cu) -- identical per-voxel physics via the shared
//      __host__ __device__ bone_remodel.h.
//   4. VERIFY: the GPU density field matches the CPU field within tolerance.
//   5. REPORT: a deterministic summary (total bone mass, per-column mass
//      profile, mechanostat state histogram) to stdout; timing to stderr.
//
//   THE SCIENCE YOU SHOULD SEE: the synthetic job loads the top edge and supports
//   the base, so remodeling concentrates bone into the columns under the load and
//   thins the lightly-loaded flanks -- the per-column mass profile makes this
//   visible, and the state histogram shows most voxels settling into the lazy
//   (homeostatic) zone once the structure has adapted.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Run-varying numbers (timings) go to STDERR, which
//   the demo shows but does not diff (PATTERNS.md section 3).
//
// Code tour: start here, then bone_remodel.h (the per-voxel physics), kernels.cu
// (the GPU mapping), reference_cpu.cpp (the baseline). The "why" is in THEORY.md.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // bone_gpu, BoneParams
#include "reference_cpu.h"    // load_bone, bone_cpu, bone_summary
#include "bone_remodel.h"     // bone_state (shared classifier for the histogram)
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "6.22";
static const char* PROJECT_NAME = "Bone Remodeling Simulation";

// Verification tolerance. CPU and GPU run the SAME double-precision arithmetic
// from the shared header, but over many remodeling steps the GPU's fused
// multiply-add (FMA) contraction and the host compiler's separate mul/add can
// diverge by a few ULPs, and that tiny difference is amplified a little by the
// nonlinear (clamped, dead-band) remodeling map. We therefore verify to a small
// PHYSICAL tolerance rather than pretend bit-identity (PATTERNS.md section 4,
// the "long iterative solver" case). 1e-9 on a density in [rho_min,1] is
// physically negligible (nano-fraction of full mineralization).
static constexpr double TOLERANCE = 1.0e-9;

// ---------------------------------------------------------------------------
// make_synthetic : the built-in job used when no data file is supplied. Chosen
//   so the demo tells a clear story AND is stable: a compression specimen loaded
//   on top, supported at the base. Values match scripts/make_synthetic.py and
//   data/sample so the committed sample and the fallback agree.
// ---------------------------------------------------------------------------
static BoneParams make_synthetic() {
    BoneParams p;
    p.nx = 24; p.ny = 16;      // 24 columns x 16 rows voxel grid
    p.remodel_steps = 60;      // ~60 "months" of remodeling
    p.relax_iters   = 80;      // Jacobi sweeps to settle the stimulus each step
    p.load     = 4.0;          // localized mechanical load pushed in on the top edge
    p.load_x0  = 10;           // loaded footprint spans the CENTER columns [10,13]:
    p.load_x1  = 13;           //   a joint/implant contact patch, not a uniform press
    p.setpoint = 0.55;         // homeostatic SED-per-mass target k
    p.lazy     = 0.20;         // lazy-zone half-width w (dead band around k)
    p.rate     = 0.05;         // remodeling gain
    p.rho_min  = 0.05;         // density floor (bone never fully vanishes)
    p.rho_init = 0.50;         // uniform starting density (unremodeled blank)
    return p;
}

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    BoneParams p;
    const char* source = "synthetic (built-in)";
    if (argc > 1) {
        try {
            p = load_bone(argv[1]);
            source = argv[1];
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[error] %s\n", e.what());
            return 2;
        }
    } else {
        p = make_synthetic();
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<double> rho_cpu, S_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    bone_cpu(p, rho_cpu, S_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU remodeling (all kernels timed) ----------------------------
    std::vector<double> rho_gpu, S_gpu;
    float gpu_kernel_ms = 0.0f;
    bone_gpu(p, rho_gpu, S_gpu, &gpu_kernel_ms);

    // ---- 4. Verify (density fields agree) ---------------------------------
    double err = 0.0;
    for (std::size_t k = 0; k < rho_cpu.size(); ++k) {
        const double d = std::fabs(rho_cpu[k] - rho_gpu[k]);
        if (d > err) err = d;
    }
    const bool pass = err <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) ----------
    // Use the GPU density for the report (verified equal to the CPU within tol).
    double total_mass = 0.0;
    std::vector<double> col_mass;
    bone_summary(p, rho_gpu, total_mass, col_mass);

    // Mechanostat state histogram over all voxels, using the shared classifier
    // and the GPU's settled stimulus field. Integer counts -> fully deterministic.
    int n_resorb = 0, n_home = 0, n_form = 0;
    for (int y = 0; y < p.ny; ++y)
        for (int x = 0; x < p.nx; ++x) {
            const int s = bone_state(x, y, p.nx, p.setpoint, p.lazy,
                                     S_gpu.data(), rho_gpu.data());
            if      (s == 0) ++n_resorb;
            else if (s == 2) ++n_form;
            else             ++n_home;
        }

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("[reduced-scope teaching model: mechanostat remodeling on a 2-D voxel grid]\n");
    std::printf("grid %dx%d, %d remodel steps, %d Jacobi sweeps/step\n",
                p.nx, p.ny, p.remodel_steps, p.relax_iters);
    std::printf("localized load %.2f on top-edge columns [%d,%d]\n",
                p.load, p.load_x0, p.load_x1);
    std::printf("mechanostat: setpoint k=%.3f, lazy zone +/-%.3f, rate=%.3f, rho in [%.3f,1]\n",
                p.setpoint, p.lazy, p.rate, p.rho_min);
    std::printf("total bone mass (sum rho) = %.6f\n", total_mass);
    std::printf("mechanostat state: resorbing=%d homeostatic=%d forming=%d (of %d voxels)\n",
                n_resorb, n_home, n_form, p.nx * p.ny);
    std::printf("per-column bone mass profile (x=0..%d):\n", p.nx - 1);
    for (int x = 0; x < p.nx; ++x) std::printf(" %.4f", col_mass[static_cast<std::size_t>(x)]);
    std::printf("\n");
    std::printf("RESULT: %s (GPU density matches CPU within tol=%.1e)\n",
                pass ? "PASS" : "FAIL", TOLERANCE);

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) -----------------
    std::fprintf(stderr, "[data]   source: %s  (%d x %d grid, %d steps)\n",
                 source, p.nx, p.ny, p.remodel_steps);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU (all kernels): %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- on this tiny grid the many small "
                         "kernel launches are launch-bound; the GPU's edge grows with grid size.\n");
    std::fprintf(stderr, "[verify] max density diff = %.3e  (tolerance %.1e)\n", err, TOLERANCE);

    // Exit code feeds the demo's pass/fail gate.
    return pass ? 0 : 1;
}
