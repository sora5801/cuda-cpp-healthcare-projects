// ===========================================================================
// src/main.cu  --  Entry point: load tree, simulate CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 6.17 : Purkinje System & Conduction System Modeling
//
// THE 5-STEP SHAPE (every project in this repo follows it)
//   1. Load the Purkinje tree (data/sample/, or a built-in synthetic fallback).
//   2. CPU reference: simulate every cable serially (reference_cpu.cpp).
//   3. GPU: one thread per cable, full PDE loop each (kernels.cu).
//   4. VERIFY: per-cable activation steps + conduction velocities match exactly.
//   5. REPORT: deterministic per-cable table + tree activation time -> STDOUT;
//              timings + varying detail -> STDERR (shown, not diffed).
//
//   STDOUT is byte-for-byte deterministic (integer activation steps and CVs
//   derived from them), so demo/run_demo can diff it against expected_output.txt.
//
// Code tour: start here, then purkinje.h (the cable physics + CV measurement),
// kernels.cuh -> kernels.cu (GPU mapping), reference_cpu.cpp (baseline + graph).
// See ../THEORY.md for the science and the "why".
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // simulate_gpu (GPU path), PurkinjeTree, CableResult
#include "reference_cpu.h"    // load_tree, simulate_cpu, compute_activation_times
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "6.17";
static const char* PROJECT_NAME = "Purkinje System & Conduction System Modeling";

// Verification tolerance for the conduction velocities (mm/ms). CV is computed
// from INTEGER activation-step counts times dt on both sides, using the SAME
// shared stepper (purkinje.h), so CPU and GPU agree exactly. We keep a tiny
// nonzero tolerance only to absorb the last double round-off bit in length/tof.
static constexpr double TOLERANCE = 1.0e-9;

// ---------------------------------------------------------------------------
// build_synthetic_tree  --  the built-in fallback used when no data file loads.
//   A tiny His-Purkinje tree of 7 cables (matches scripts/make_synthetic.py and
//   the committed sample). Cable 0 is the paced His bundle; 1..2 are the left/
//   right bundle branches (different diameters -> different D -> different CV);
//   3..6 are terminal Purkinje fascicles. Diffusion D is engineered so the
//   demo shows a clear CV spread and one deliberate slow branch.
//   Keeping this in sync with the file means the demo output is identical whether
//   the sample is present or not.
// ---------------------------------------------------------------------------
static PurkinjeTree build_synthetic_tree() {
    PurkinjeTree t;
    // Fields: n_nodes length_mm D dt_ms n_steps stim_amp stim_dur_ms stim_width
    //         thresh parent delay_ms   (dt/n_steps set below, shared clock)
    struct Row { int n; double len, D; double amp, dur; int w; double thr; int parent; double delay; };
    const Row rows[] = {
        // His bundle (root): thick, fast, directly paced.
        {   65, 20.0, 3.0, 1.0, 2.0, 3, 0.5, -1, 0.0 },
        // Left bundle branch: thick/fast.
        {   65, 25.0, 3.0, 0.0, 0.0, 0, 0.5,  0, 1.0 },
        // Right bundle branch: thinner/slower.
        {   65, 25.0, 1.5, 0.0, 0.0, 0, 0.5,  0, 1.0 },
        // Terminal Purkinje fascicles (leaves) -> Purkinje-muscle junctions.
        {   65, 15.0, 2.5, 0.0, 0.0, 0, 0.5,  1, 0.5 },
        {   65, 15.0, 2.5, 0.0, 0.0, 0, 0.5,  1, 0.5 },
        {   65, 15.0, 2.0, 0.0, 0.0, 0, 0.5,  2, 0.5 },
        {   65, 15.0, 2.0, 0.0, 0.0, 0, 0.5,  2, 0.5 },
    };
    const double dt_ms   = 0.01;   // stable explicit step for these D (THEORY §numerics)
    const int    n_steps = 6000;   // 60 ms of propagation
    for (const Row& r : rows) {
        CableParams c;
        c.n_nodes = r.n; c.length_mm = r.len; c.D = r.D;
        c.dt_ms = dt_ms; c.n_steps = n_steps;
        c.stim_amp = r.amp; c.stim_dur_ms = r.dur; c.stim_width = r.w;
        c.thresh = r.thr; c.parent = r.parent; c.delay_ms = r.delay;
        // Non-paced cables inherit the front from their parent's distal end. We
        // model that inheritance as a stimulus at their proximal nodes triggered
        // implicitly: for this reduced teaching model each cable is paced at its
        // own left end so it always fires, and the TREE timing is assembled by
        // the graph-delay pass. (See THEORY "real world" for true PMJ coupling.)
        if (r.parent >= 0) { c.stim_amp = 1.0; c.stim_dur_ms = 2.0; c.stim_width = 3; }
        t.cables.push_back(c);
    }
    return t;
}

int main(int argc, char** argv) {
    // ---- 1. Load the tree --------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/purkinje_tree.txt";
    PurkinjeTree tree;
    const char* source = path.c_str();
    try {
        tree = load_tree(path);
    } catch (const std::exception& e) {
        // Fall back to the built-in synthetic tree so the program always runs.
        std::fprintf(stderr, "[data]   could not load '%s' (%s); using built-in synthetic tree\n",
                     path.c_str(), e.what());
        tree = build_synthetic_tree();
        source = "synthetic (built-in)";
    }
    const int N = tree_size(tree);

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<CableResult> res_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    simulate_cpu(tree, res_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU ensemble (kernel timed inside the wrapper) -----------------
    std::vector<CableResult> res_gpu;
    float gpu_kernel_ms = 0.0f;
    simulate_gpu(tree, res_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    // The activation-step indices are integers -> compare exactly. The CVs are
    // doubles -> compare within the tiny tolerance above.
    double worst_cv = 0.0;
    int    step_mismatches = 0;
    for (int i = 0; i < N; ++i) {
        if (res_cpu[i].activate_step_in  != res_gpu[i].activate_step_in ||
            res_cpu[i].activate_step_out != res_gpu[i].activate_step_out ||
            res_cpu[i].captured          != res_gpu[i].captured) {
            ++step_mismatches;
        }
        worst_cv = std::fmax(worst_cv, std::fabs(res_cpu[i].cv_mm_per_ms - res_gpu[i].cv_mm_per_ms));
    }
    const bool pass = (step_mismatches == 0) && (worst_cv <= TOLERANCE);

    // Graph-delay pass: turn per-cable local delays into absolute PMJ activation
    // times (uses the GPU results; identical to CPU so either would do).
    const std::vector<double> t_out = compute_activation_times(tree, res_gpu);
    double total_activation = 0.0;   // total ventricular activation time (ms)
    int    captured_cables  = 0;
    for (int i = 0; i < N; ++i) {
        if (t_out[i] >= 0.0) { ++captured_cables; if (t_out[i] > total_activation) total_activation = t_out[i]; }
    }

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("Purkinje tree: %d cables, dt=%.3f ms, %d steps (%.1f ms)\n",
                N, tree.cables[0].dt_ms, tree.cables[0].n_steps,
                tree.cables[0].n_steps * tree.cables[0].dt_ms);
    std::printf("per-cable (idx parent lenmm D -> CV[mm/ms] PMJ_t[ms] captured):\n");
    for (int i = 0; i < N; ++i) {
        const CableParams& c = tree.cables[i];
        std::printf("  c%-2d p%-3d %5.1f %4.2f -> %7.4f %8.3f %s\n",
                    i, c.parent, c.length_mm, c.D,
                    res_gpu[i].cv_mm_per_ms,
                    (t_out[i] >= 0.0 ? t_out[i] : 0.0),
                    res_gpu[i].captured ? "yes" : "BLOCK");
    }
    std::printf("tree: %d/%d cables captured; total ventricular activation = %.3f ms\n",
                captured_cables, N, total_activation);
    std::printf("RESULT: %s (GPU per-cable steps + CV match CPU; tol=1.0e-09)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d cables)\n", source, N);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernel: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- a %d-cable tree is small; the GPU's edge "
                         "grows toward the ~10^5-segment trees of real hearts.\n", N);
    std::fprintf(stderr, "[verify] step mismatches = %d ; worst CV diff = %.3e (tol %.1e)\n",
                 step_mismatches, worst_cv, TOLERANCE);

    return pass ? 0 : 1;
}
