// ===========================================================================
// src/main.cu  --  Entry point: load plan, run CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 5.7 : Brachytherapy Dose & Source Modeling
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the PLAN (source TG-43 tables + dwell positions + dose grid) from
//      data/sample, or fall back to a built-in synthetic plan.
//   2. Compute the CPU reference dose (reference_cpu.cpp)   -> trusted answer.
//   3. Compute the GPU dose           (kernels.cu)          -> the thing taught.
//   4. VERIFY: assert GPU agrees with CPU within a tolerance -> correctness.
//   5. REPORT: deterministic dose numbers to stdout; timing to stderr.
//
//   STDOUT is kept byte-for-byte deterministic so demo/run_demo can diff it
//   against demo/expected_output.txt. Anything that varies run-to-run (timings)
//   goes to STDERR, which the demo shows but does not diff.
//
// READ THIS FIRST in the code tour, then tg43_physics.h (the formula),
// kernels.cuh -> kernels.cu (the GPU path), reference_cpu.cpp (the baseline).
// See ../THEORY.md for the "why".
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // dose_gpu (GPU path); pulls in reference_cpu.h
#include "util/io.hpp"        // util::CpuTimer

// These two tokens identify the program and MUST stay in sync with the
// expected_output.txt header line.
static const char* PROJECT_ID   = "5.7";
static const char* PROJECT_NAME = "Brachytherapy Dose & Source Modeling";

// Correctness tolerance. Dose spans several orders of magnitude across the grid
// (it falls ~1/r^2 away from the sources), so a single absolute tolerance is
// meaningless -- a 1e-4 error is huge in a low-dose voxel and negligible in a
// hot one. We instead bound the RELATIVE error per voxel. CPU and GPU run the
// identical tg43_physics.h math with the identical double accumulation order,
// so they differ only by fused-multiply-add contraction; 1e-5 relative is a
// safe, honest ceiling (PATTERNS.md section 4). See THEORY "How we verify".
static constexpr double REL_TOLERANCE = 1.0e-5;

// ---------------------------------------------------------------------------
// max_rel_err: largest |cpu-gpu| / max(|cpu|, floor) over the dose grid. The
// floor keeps near-zero voxels from producing a divide-by-almost-zero blowup;
// it is set well below any clinically meaningful dose in this synthetic plan.
// ---------------------------------------------------------------------------
static double max_rel_err(const std::vector<float>& cpu, const std::vector<float>& gpu) {
    if (cpu.size() != gpu.size()) return 1.0e300;   // shape mismatch -> "fail"
    const double floor = 1.0e-6;                     // reference-scale denominator
    double worst = 0.0;
    for (std::size_t i = 0; i < cpu.size(); ++i) {
        const double c = cpu[i], gv = gpu[i];
        const double denom = std::fabs(c) > floor ? std::fabs(c) : floor;
        const double rel = std::fabs(c - gv) / denom;
        if (rel > worst) worst = rel;
    }
    return worst;
}

// ---------------------------------------------------------------------------
// make_synthetic_plan: the built-in fallback plan (used if no data file loads).
//   It mirrors data/sample/plan_sample.txt EXACTLY so the program still runs --
//   and produces the SAME stdout -- with zero arguments. A single Ir-192-like
//   line source (L=0.35 cm) stepped to three collinear dwells along z, with a
//   coarse but realistic-shaped g_L(r) and a mild anisotropy table. The grid is
//   a small 2-D-ish slab (nz=1) so the demo stays tiny and fast.
//
//   This is SYNTHETIC teaching data (labeled everywhere) -- it is NOT the AAPM
//   consensus dataset for any real source. See data/README.md.
// ---------------------------------------------------------------------------
static Plan make_synthetic_plan() {
    Plan p;
    SourceModel& s = p.source;

    s.L      = 0.35;   // active length [cm], HDR Ir-192 pellet scale
    s.Lambda = 1.109;  // dose-rate constant [cGy/(h*U)], Ir-192-like magnitude

    // Radial dose function g_L(r): ~1 near 1 cm, rising slightly then falling as
    // scatter+absorption trade off (a plausible teaching shape, not measured).
    s.n_g = 7;
    const double gr[7]  = {0.25, 0.50, 1.00, 2.00, 3.00, 5.00, 7.00};
    const double gv[7]  = {0.980, 1.010, 1.000, 0.965, 0.910, 0.780, 0.650};
    for (int i = 0; i < s.n_g; ++i) { s.g_r[i] = gr[i]; s.g_val[i] = gv[i]; }

    // Anisotropy F(r,theta): 1 on the transverse plane (theta=90), dropping near
    // the poles (theta -> 0,180) where the source self-absorbs. 3 radii x 5 angles.
    s.n_Fr = 3; s.n_Ft = 5;
    const double Fr[3] = {0.5, 1.0, 3.0};
    const double Ft[5] = {0.0, 45.0, 90.0, 135.0, 180.0};
    for (int i = 0; i < s.n_Fr; ++i) s.F_r[i] = Fr[i];
    for (int j = 0; j < s.n_Ft; ++j) s.F_t[j] = Ft[j];
    const double Fv[3][5] = {
        {0.62, 0.90, 1.00, 0.90, 0.62},   // r = 0.5 cm
        {0.68, 0.93, 1.00, 0.93, 0.68},   // r = 1.0 cm
        {0.75, 0.95, 1.00, 0.95, 0.75},   // r = 3.0 cm
    };
    for (int i = 0; i < s.n_Fr; ++i)
        for (int j = 0; j < s.n_Ft; ++j)
            s.F_val[i * s.n_Ft + j] = Fv[i][j];

    // Dose grid: a 41 x 41 x 1 slab, 1 mm voxels, centered so the sources sit
    // near the middle. Origin places voxel(0,0,0) at (-2,-2,0) cm.
    DoseGrid& g = p.grid;
    g.nx = 41; g.ny = 41; g.nz = 1;
    g.ox = -2.0; g.oy = -2.0; g.oz = 0.0;
    g.spacing = 0.1;

    // Three collinear dwells along z at x=y=0 (a mini catheter). Equal weights.
    p.dwells = {
        {0.0, 0.0, -0.5, 1.0},
        {0.0, 0.0,  0.0, 1.0},
        {0.0, 0.0,  0.5, 1.0},
    };
    return p;
}

int main(int argc, char** argv) {
    // ---- 1. Load the plan ---------------------------------------------------
    Plan plan;
    std::string source_desc = "synthetic (built-in)";
    if (argc > 1) {
        try {
            plan = load_plan(argv[1]);           // parse the sample plan file
            source_desc = argv[1];
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[warn] could not load '%s' (%s); using built-in plan.\n",
                         argv[1], e.what());
            plan = make_synthetic_plan();
        }
    } else {
        plan = make_synthetic_plan();
    }

    const DoseGrid& g = plan.grid;

    // ---- 2. CPU reference (timed) ------------------------------------------
    std::vector<float> dose_cpu_v;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    dose_cpu(plan, dose_cpu_v);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU dose (kernel timed inside the wrapper) ---------------------
    std::vector<float> dose_gpu_v;
    float gpu_kernel_ms = 0.0f;
    dose_gpu(plan, dose_gpu_v, &gpu_kernel_ms);

    // ---- 4. Verify ----------------------------------------------------------
    const double rel_err = max_rel_err(dose_cpu_v, dose_gpu_v);
    const bool pass = rel_err <= REL_TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    // Helper lambda: flat index of voxel (ix,iy,iz), x-fastest (matches layout).
    auto vidx = [&](int ix, int iy, int iz) { return (iz * g.ny + iy) * g.nx + ix; };

    // Find the hottest voxel (highest dose) and its grid coordinates.
    int amax = 0; float vmax = dose_gpu_v[0];
    for (std::size_t i = 1; i < dose_gpu_v.size(); ++i)
        if (dose_gpu_v[i] > vmax) { vmax = dose_gpu_v[i]; amax = static_cast<int>(i); }
    const int mx = amax % g.nx;
    const int my = (amax / g.nx) % g.ny;
    const int mz = amax / (g.nx * g.ny);

    // A physically meaningful probe: dose at 1 cm along +x from the grid center
    // (transverse-ish direction). The grid center voxel is at (nx/2,ny/2,0);
    // 1 cm at 1 mm spacing is +10 voxels in x.
    const int cix = g.nx / 2, ciy = g.ny / 2, ciz = 0;
    const int probe = vidx(cix + 10 < g.nx ? cix + 10 : g.nx - 1, ciy, ciz);

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("[TG-43 analytic dose | SYNTHETIC teaching source, not clinical]\n");
    std::printf("source: line L=%.2f cm  Lambda=%.4f cGy/(h*U)  dwells=%d\n",
                plan.source.L, plan.source.Lambda,
                static_cast<int>(plan.dwells.size()));
    std::printf("grid: %d x %d x %d voxels @ %.2f cm  (%d voxels)\n",
                g.nx, g.ny, g.nz, g.spacing, g.size());
    std::printf("max dose = %.6f cGy/h at voxel (%d,%d,%d)\n", vmax, mx, my, mz);
    std::printf("dose @ ~1cm transverse probe = %.6f cGy/h\n", dose_gpu_v[probe]);

    // A short deterministic profile ACROSS the center row (y = center, z = 0),
    // 8 evenly spaced columns -- shows the 1/r^2-like falloff either side.
    std::printf("center-row profile (8 samples):");
    for (int s = 0; s < 8; ++s) {
        const int ix = (s * (g.nx - 1)) / 7;
        std::printf(" %.6f", dose_gpu_v[vidx(ix, ciy, ciz)]);
    }
    std::printf("\n");
    std::printf("RESULT: %s (GPU matches CPU within rel-tol=1.0e-05)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%d dwells, %d voxels)\n",
                 source_desc.c_str(), static_cast<int>(plan.dwells.size()), g.size());
    std::fprintf(stderr, "[timing] CPU dose: %.3f ms   GPU dose: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny plan is dominated "
                         "by launch/copy overhead; the GPU's edge grows with grid size and "
                         "dwell count (a real prostate HDR plan is millions of voxels).\n");
    std::fprintf(stderr, "[verify] max_rel_err = %.3e  (tolerance %.1e)\n",
                 rel_err, REL_TOLERANCE);

    return pass ? 0 : 1;
}
