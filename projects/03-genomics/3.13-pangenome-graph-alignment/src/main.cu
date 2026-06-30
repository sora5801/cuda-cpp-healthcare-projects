// ===========================================================================
// src/main.cu  --  Entry point: load graph + read, align, verify, report
// ---------------------------------------------------------------------------
// Project 3.13 : Pangenome Graph Alignment
//
// The repo's 5-step shape:
//   1. Load the pangenome graph + query read (data/sample, see data/README.md).
//   2. CPU reference fills every node's score block (reference_cpu.cpp).
//   3. GPU fills the SAME blocks via the per-node anti-diagonal wavefront
//      (kernels.cu), nodes in topological order.
//   4. VERIFY: every cell of every GPU block equals the CPU block (exact ints).
//   5. REPORT: deterministic best score + node path + alignment to stdout;
//      timing to stderr.
//
// We traceback ONCE on the host (from the GPU-filled blocks) to display the best
// path; the GPU teaching point is the parallel block FILL, not the serial
// traceback. STDOUT is byte-for-byte deterministic so demo/run_demo can diff it;
// timings (run-to-run varying) go to STDERR.
//
// Code tour: start here, then kernels.cuh -> kernels.cu, then reference_cpu.*.
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // graph_sw_gpu
#include "reference_cpu.h"    // load_problem, graph_sw_cpu, traceback, structs
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "3.13";
static const char* PROJECT_NAME = "Pangenome Graph Alignment";

static constexpr int PREVIEW_COLS = 60;   // how many alignment columns to print

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/graph_sample.txt";
    Problem prob;
    try {
        prob = load_problem(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const Graph& g = prob.graph;

    // ---- 2. CPU reference (timed) -----------------------------------------
    GraphDP dp_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    graph_sw_cpu(prob, dp_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU wavefront (timed) -----------------------------------------
    GraphDP dp_gpu;
    float gpu_kernel_ms = 0.0f;
    graph_sw_gpu(prob, dp_gpu, &gpu_kernel_ms);

    // ---- 4. Verify (all blocks must be identical, exact integers) ---------
    int max_abs_diff = 0, mismatches = 0;
    const std::size_t ncells = dp_cpu.H.size();
    const bool shape_ok = (dp_gpu.H.size() == ncells);
    if (shape_ok) {
        for (std::size_t k = 0; k < ncells; ++k) {
            const int d = dp_cpu.H[k] - dp_gpu.H[k];
            const int ad = d < 0 ? -d : d;
            if (ad) { ++mismatches; if (ad > max_abs_diff) max_abs_diff = ad; }
        }
    }
    const bool pass = shape_ok && (mismatches == 0);

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    const PathAlignment a = traceback(prob, dp_gpu);
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("Graph: %d nodes, %d bases; query length = %d; "
                "scoring match=+%d mismatch=%d gap=%d\n",
                g.num_nodes, g.total_bases, prob.qlen, MATCH, MISMATCH, GAP);
    std::printf("best local score = %d  ending at node %s, cell (i,j)=(%d,%d)\n",
                a.score, (a.end_node >= 0 ? g.name[a.end_node].c_str() : "-"),
                a.end_i, a.end_j);
    std::printf("best path through graph = %s\n", a.node_path.c_str());
    if (a.length > 0) {
        const double pct = 100.0 * a.identities / a.length;
        std::printf("aligned length = %d, identities = %d/%d (%.1f%%)\n",
                    a.length, a.identities, a.length, pct);
        const int show = a.length < PREVIEW_COLS ? a.length : PREVIEW_COLS;
        std::printf("alignment (first %d columns):\n", show);
        std::printf("  Q: %s\n", a.q_line.substr(0, show).c_str());
        std::printf("     %s\n", a.m_line.substr(0, show).c_str());
        std::printf("  G: %s\n", a.t_line.substr(0, show).c_str());
    } else {
        std::printf("aligned length = 0 (no positive-scoring local alignment)\n");
    }
    std::printf("RESULT: %s (GPU blocks match CPU exactly)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d nodes, %zu DP cells across all blocks)\n",
                 path.c_str(), g.num_nodes, ncells);
    std::fprintf(stderr, "[timing] CPU fill: %.3f ms   GPU wavefront: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- a tiny graph issues many small per-diagonal "
                         "launches; the GPU wins on long reads / large bubbles / batched reads.\n");
    if (!shape_ok)
        std::fprintf(stderr, "[verify] SHAPE MISMATCH: cpu cells=%zu gpu cells=%zu\n",
                     ncells, dp_gpu.H.size());
    std::fprintf(stderr, "[verify] block-cell mismatches = %d, max_abs_diff = %d\n",
                 mismatches, max_abs_diff);

    return pass ? 0 : 1;
}
