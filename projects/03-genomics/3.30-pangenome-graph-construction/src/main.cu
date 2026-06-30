// ===========================================================================
// src/main.cu  --  Entry point: build the layout problem, run CPU + GPU, verify
// ---------------------------------------------------------------------------
// Project 3.30 : Pangenome Graph Construction
//
// 5-step shape (every project in this repo follows it):
//   1. Load the pangenome (data/sample) and BUILD the stress problem (terms).
//   2. CPU reference layout (reference_cpu.cpp)        -> trusted positions.
//   3. GPU layout (kernels.cu)                          -> the thing taught.
//   4. VERIFY: GPU positions match CPU within tolerance (fixed-point => exact),
//      and the derived 1-D NODE ORDER matches.
//   5. REPORT: deterministic layout + node order + stress to stdout; timing to
//      stderr. STDOUT is byte-stable so demo/run_demo can diff it.
//
// Code tour: start here, then layout.h (the per-term physics), reference_cpu.*
// (problem construction + CPU baseline), kernels.cuh -> kernels.cu (the GPU).
// ===========================================================================
#include <algorithm>   // std::sort, std::stable_sort
#include <cmath>       // std::fabs
#include <cstdio>
#include <numeric>     // std::iota
#include <string>
#include <vector>

#include "kernels.cuh"        // layout_gpu, LayoutProblem
#include "reference_cpu.h"    // load_pangenome, build_problem, layout_cpu, ...
#include "util/io.hpp"        // util::CpuTimer

// Identity tokens. MUST stay in sync with demo/expected_output.txt.
static const char* PROJECT_ID   = "3.30";
static const char* PROJECT_NAME = "Pangenome Graph Construction";

// --- Layout hyper-parameters (fixed so the result is deterministic) ---------
// hops  : how many path-steps apart two nodes may be and still get a term.
//         Larger = a denser, stiffer problem (more long-range constraints).
// iters : number of full-batch SMACOF (stress-majorization) sweeps. SMACOF is
//         monotone and parameter-free, so this is the only knob the solver needs.
static constexpr int HOPS  = 3;
static constexpr int ITERS = 100;

// Verification tolerance. The GPU and CPU share the per-term physics AND the
// fixed-point integer reduction, so positions should agree to the last bit; we
// allow 1e-9 bp of slack purely as defensive slack against compiler reordering
// of the (associative, here) final integer-to-double conversions.
static constexpr double TOLERANCE = 1.0e-9;

// Derive the 1-D node ORDER from positions: the permutation of node ids sorted by
// coordinate (ties broken by id for determinism). This permutation IS the ODGI
// deliverable -- "odgi sort" emits exactly this order so a downstream tool can
// linearise the graph. Returned as a vector of node ids in left-to-right order.
static std::vector<int> order_from_positions(const std::vector<double>& x) {
    std::vector<int> order(x.size());
    std::iota(order.begin(), order.end(), 0);                 // 0,1,2,...,N-1
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return x[a] < x[b]; });  // by coordinate
    return order;
}

int main(int argc, char** argv) {
    // ---- 1. Load + build ---------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/pangenome_sample.txt";
    Pangenome g;
    try {
        g = load_pangenome(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const LayoutProblem prob = build_problem(g, HOPS, ITERS);
    const double stress0 = compute_stress(prob, prob.init_x);   // stress before layout

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<double> x_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    const double stress_cpu = layout_cpu(prob, x_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU layout (loop timed inside) --------------------------------
    std::vector<double> x_gpu;
    float gpu_kernel_ms = 0.0f;
    const double stress_gpu = layout_gpu(prob, x_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    double max_pos_diff = 0.0;
    for (int k = 0; k < g.num_nodes; ++k)
        max_pos_diff = std::fmax(max_pos_diff, std::fabs(x_cpu[k] - x_gpu[k]));
    const std::vector<int> order_cpu = order_from_positions(x_cpu);
    const std::vector<int> order_gpu = order_from_positions(x_gpu);
    bool order_match = true;
    for (int k = 0; k < g.num_nodes; ++k)
        if (order_cpu[k] != order_gpu[k]) { order_match = false; break; }
    const bool pass = (max_pos_diff <= TOLERANCE) && order_match;

    // Anchor the printed coordinate frame: shift so the LEFTMOST node sits at 0.
    // (Stress only depends on differences, so absolute offset is a free gauge; we
    // fix it for a clean, reproducible printout.) We use the GPU positions.
    double min_x = x_gpu[0];
    for (int k = 1; k < g.num_nodes; ++k) min_x = std::fmin(min_x, x_gpu[k]);

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) ----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("graph: %d nodes, %d genome paths -> %d layout terms\n",
                g.num_nodes, static_cast<int>(g.paths.size()),
                static_cast<int>(prob.terms.size()));
    std::printf("layout: %d SMACOF sweeps (hops=%d, Guttman transform)\n", ITERS, HOPS);

    // Per-node final coordinate (left edge), printed in node-id order.
    std::printf("node coordinates (bp, leftmost node at 0):\n");
    for (int k = 0; k < g.num_nodes; ++k)
        std::printf("  node %2d (len %4d bp): x = %10.3f\n",
                    k, g.node_len[k], x_gpu[k] - min_x);

    // The 1-D node order -- the ODGI "sort" deliverable.
    std::printf("1-D node order (left to right):");
    for (int k = 0; k < g.num_nodes; ++k) std::printf(" %d", order_gpu[k]);
    std::printf("\n");

    // Stress drop: how much the layout tightened the drawing (lower is better).
    std::printf("stress: initial %.4f -> final %.4f\n", stress0, stress_gpu);
    std::printf("RESULT: %s (GPU layout matches CPU; same 1-D order)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) -----------------
    std::fprintf(stderr, "[data]   source: %s  (%d nodes, %d paths)\n",
                 path.c_str(), g.num_nodes, static_cast<int>(g.paths.size()));
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU loop: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- tiny graphs are launch-bound "
                         "(2 kernels x %d sweeps); ODGI's GPU edge appears at millions "
                         "of nodes / billions of terms.\n", ITERS);
    std::fprintf(stderr, "[verify] max position diff = %.3e bp, order match = %s, "
                         "stress(cpu/gpu) = %.6f / %.6f\n",
                 max_pos_diff, order_match ? "yes" : "no", stress_cpu, stress_gpu);

    return pass ? 0 : 1;
}
