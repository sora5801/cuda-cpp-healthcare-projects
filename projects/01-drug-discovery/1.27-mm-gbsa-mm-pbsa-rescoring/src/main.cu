// ===========================================================================
// src/main.cu  --  Entry point: load complex, rescore on CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 1.27 : MM-GBSA / MM-PBSA Rescoring
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the problem -- a protein-ligand complex (rigid receptor + S MD
//      snapshots of the ligand) from data/sample, or a built-in synthetic
//      fallback if no file is given.
//   2. CPU reference (reference_cpu.cpp) -> trusted per-snapshot energies.
//   3. GPU rescoring (kernels.cu)         -> the thing being taught.
//   4. VERIFY: assert GPU agrees with CPU within a documented tolerance.
//   5. REPORT: deterministic per-snapshot dG + the MM-GBSA mean to stdout;
//      timing + max error to stderr.
//
//   STDOUT is byte-for-byte deterministic (diffed by demo/run_demo against
//   demo/expected_output.txt). Anything that varies run-to-run (wall-clock
//   timings) goes to STDERR, which the demo shows but does NOT diff.
//
// Code tour: start here, then kernels.cuh -> kernels.cu, then reference_cpu.*.
// The science/math/GPU-mapping live in ../THEORY.md.
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // rescore_gpu (GPU path), Atom, Complex
#include "reference_cpu.h"    // load_complex, rescore_cpu, mean, snapshot_dg
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "1.27";
static const char* PROJECT_NAME = "MM-GBSA / MM-PBSA Rescoring";

// ---------------------------------------------------------------------------
// Verification tolerance. The CPU and GPU run the SAME snapshot_dg() source, so
// they agree to near machine precision; the only term that can differ is exp()
// (libm vs. CUDA math, ~1 ULP), and only after dividing into an O(R*L) sum. A
// 1e-6 kcal/mol absolute tolerance is therefore generous yet physically
// negligible (binding energies are tens of kcal/mol). PATTERNS.md §4: a short
// double-precision computation -> ~machine precision, not exactly 0.
// ---------------------------------------------------------------------------
static constexpr double TOLERANCE = 1.0e-6;

// max_abs_err over two double vectors (util::max_abs_err is float-only; this is
// the double twin we need for energies). Returns +inf on a length mismatch so a
// shape bug cannot masquerade as agreement.
static double max_abs_err_d(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return 1e300;
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        if (d < 0) d = -d;
        if (d > worst) worst = d;
    }
    return worst;
}

// ---------------------------------------------------------------------------
// make_synthetic_complex: a tiny, deterministic, clearly-SYNTHETIC fallback so
// the program runs even with no data file. It mirrors the committed sample's
// structure (a small charged receptor pocket + a few ligand snapshots that
// drift outward), so the printed energies climb toward zero as the ligand
// unbinds -- an interpretable result (PATTERNS.md §6). The committed sample is
// the canonical input; this fallback exists only so `./exe` with no args works.
// ---------------------------------------------------------------------------
static Complex make_synthetic_complex() {
    Complex cx;
    cx.R = 3;            // 3 receptor atoms forming a small charged pocket
    cx.L = 2;            // 2 ligand atoms
    cx.S = 4;            // 4 MD snapshots (the ligand drifts outward)
    cx.minus_TdS = 8.0;  // constant entropy penalty [kcal/mol]

    // Receptor: a line of alternating charges (a crude binding pocket).
    cx.receptor = {
        // x     y    z     q     sigma  eps    born
        {  0.0,  0.0, 0.0, -0.6,  3.40,  0.10,  2.0 },
        {  3.0,  0.0, 0.0,  0.5,  3.40,  0.10,  2.0 },
        {  6.0,  0.0, 0.0, -0.4,  3.40,  0.10,  2.0 },
    };
    // Ligand snapshots: 2 atoms, displaced by +d along z each frame (unbinding).
    cx.ligand_snapshots.clear();
    for (int s = 0; s < cx.S; ++s) {
        const double d = 3.0 + 1.5 * s;   // z-offset grows each snapshot
        cx.ligand_snapshots.push_back({ 1.5, 0.0, d,        0.55, 3.25, 0.12, 1.8 });
        cx.ligand_snapshots.push_back({ 4.5, 0.0, d + 1.0, -0.45, 3.25, 0.12, 1.8 });
    }
    return cx;
}

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    Complex cx;
    std::string source;
    if (argc > 1) {
        try {
            cx = load_complex(argv[1]);
            source = argv[1];
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[error] %s\n", e.what());
            return 2;
        }
    } else {
        cx = make_synthetic_complex();
        source = "synthetic (built-in)";
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<double> dg_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    rescore_cpu(cx, dg_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU rescoring (kernel timed inside the wrapper) ---------------
    std::vector<double> dg_gpu;
    float gpu_kernel_ms = 0.0f;
    rescore_gpu(cx, dg_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    const double err  = max_abs_err_d(dg_cpu, dg_gpu);
    const bool   pass = err <= TOLERANCE;

    // The MM-GBSA binding free-energy estimate = mean of per-snapshot dG. We
    // report the GPU's mean (verified equal to the CPU's) as the headline number.
    const double dG_bind = mean(dg_gpu);

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) ----------
    // Fixed precision so the bytes are reproducible. Per-snapshot energies use
    // %.4f kcal/mol; the headline mean uses %.4f too. (The exp() ULP difference
    // is far below the 4th decimal, so the rounded text is identical CPU vs GPU.)
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("Rescoring 1 complex: receptor=%d atoms, ligand=%d atoms, snapshots=%d\n",
                cx.R, cx.L, cx.S);
    std::printf("per-snapshot dG (kcal/mol):\n");
    for (int s = 0; s < cx.S; ++s)
        std::printf("  frame %2d : dG = %10.4f\n", s, dg_gpu[static_cast<std::size_t>(s)]);
    std::printf("MM-GBSA dG_bind (ensemble mean) = %.4f kcal/mol\n", dG_bind);
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-06)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) -----------------
    std::fprintf(stderr, "[data]   source: %s  (R=%d, L=%d, S=%d, -TdS=%.3f)\n",
                 source.c_str(), cx.R, cx.L, cx.S, cx.minus_TdS);
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny S is dominated by "
                         "launch/copy overhead; the GPU wins as snapshots scale to thousands.\n");
    std::fprintf(stderr, "[verify] max_abs_err = %.3e kcal/mol  (tolerance %.1e)\n", err, TOLERANCE);

    // Exit code feeds the demo's pass/fail gate.
    return pass ? 0 : 1;
}
