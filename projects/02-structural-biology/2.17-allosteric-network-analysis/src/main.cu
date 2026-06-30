// ===========================================================================
// src/main.cu  --  Entry point: load trajectory, run CPU+GPU DCC, verify, report
// ---------------------------------------------------------------------------
// Project 2.17 : Allosteric Network Analysis
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the MD trajectory sample (data/sample) into a Trajectory.
//   2. Compute the per-residue means, then the DCC matrix on the CPU (reference).
//   3. Compute the DCC matrix on the GPU (kernels.cu) -- the thing being taught.
//   4. VERIFY: assert the GPU matrix matches the CPU matrix EXACTLY (the shared
//      dcc_core.h makes both run identical double-precision math).
//   5. ANALYZE + REPORT: build the residue contact graph, run Floyd-Warshall on
//      the -log|C| weights, and print the allosteric communication pathway from
//      the annotated allosteric site to the active site. Deterministic result to
//      STDOUT; timing/diagnostics to STDERR.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Run-to-run varying numbers (timings) go to STDERR,
//   which the demo shows but does not diff (PATTERNS.md section 3).
//
// READ THIS FIRST in the code tour, then dcc_core.h (the per-pair math),
// kernels.cuh -> kernels.cu (the GPU path), and reference_cpu.cpp (the baseline).
// See ../THEORY.md for the science and the GPU mapping.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // dcc_matrix_gpu (GPU path), Trajectory (via reference_cpu.h)
#include "reference_cpu.h"    // load_trajectory, residue_means, dcc_matrix_cpu, network fns
#include "dcc_core.h"         // comm_weight (for reporting the path's correlations)
#include "util/io.hpp"        // util::CpuTimer

// Identify the program. These MUST stay in sync with demo/expected_output.txt.
static const char* PROJECT_ID   = "2.17";
static const char* PROJECT_NAME = "Allosteric Network Analysis";

// Correctness tolerance. The GPU and CPU DCC matrices are computed by the SAME
// dcc_core.h code in double precision, so they agree to the LAST BIT -- we demand
// an EXACT match (0.0). This is the strongest, most honest tolerance and is the
// right one here (PATTERNS.md section 4: exact when identical ops run on both sides).
static constexpr float MATRIX_TOLERANCE = 0.0f;

// Contact cutoff in angstroms: residues within this equilibrium Cα-Cα distance
// are network neighbors (plus backbone neighbors). 8.0 Å is a standard residue
// contact radius used in protein contact-network analysis (Bio3D, ProDy).
static constexpr double CONTACT_CUTOFF = 8.0;

int main(int argc, char** argv) {
    // ---- 1. Load the trajectory --------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/trajectory.txt";
    Trajectory traj;
    try {
        traj = load_trajectory(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int N = traj.N, T = traj.T;

    // ---- 2. CPU reference: means + DCC matrix (timed) ----------------------
    std::vector<double> mean;
    residue_means(traj, mean);

    std::vector<float> C_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    dcc_matrix_cpu(traj, mean, C_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU: the same DCC matrix (kernel timed inside the wrapper) ------
    std::vector<float> C_gpu;
    float gpu_kernel_ms = 0.0f;
    dcc_matrix_gpu(traj, mean, C_gpu, &gpu_kernel_ms);

    // ---- 4. Verify GPU == CPU (exact) --------------------------------------
    // Walk every matrix entry; the worst absolute difference must be 0.
    float worst = 0.0f;
    for (std::size_t k = 0; k < C_cpu.size(); ++k)
        worst = std::fmax(worst, std::fabs(C_cpu[k] - C_gpu[k]));
    const bool pass = (worst <= MATRIX_TOLERANCE);

    // ---- 5. Network analysis on the verified matrix ------------------------
    // Use the GPU matrix from here on (it equals the CPU one). Build the contact
    // graph, then all-pairs shortest paths on the -log|C| communication weights.
    std::vector<char> adj;
    build_contacts(traj, mean, CONTACT_CUTOFF, adj);

    std::vector<double> dist;
    std::vector<int> next;
    shortest_paths(C_gpu, adj, N, dist, next);

    // The headline biological question: how does the allosteric site talk to the
    // active site? Reconstruct the optimal communication pathway between them.
    const int src = traj.site_allo, dst = traj.site_active;
    std::vector<int> comm_path = reconstruct_path(next, N, src, dst);
    const double path_cost = dist[static_cast<std::size_t>(src) * N + dst];

    // Count the edges of the contact graph (a teaching statistic about the network).
    long edge_count = 0;
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j)
            if (adj[static_cast<std::size_t>(i) * N + j]) ++edge_count;

    // The single strongest direct coupling between the two sites (for contrast
    // with the multi-hop pathway): just |C[src][dst]| itself.
    const float direct_corr = C_gpu[static_cast<std::size_t>(src) * N + dst];

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("trajectory: %d residues, %d frames (synthetic)\n", N, T);
    std::printf("DCC matrix: %dx%d  contact graph: %ld edges (cutoff %.1f A)\n",
                N, N, edge_count, CONTACT_CUTOFF);
    std::printf("allosteric site: residue %d   active site: residue %d\n", src, dst);
    std::printf("direct correlation C[%d][%d] = %.4f\n", src, dst, direct_corr);

    // The communication pathway and its total cost (sum of -log|C| edge weights).
    if (!comm_path.empty()) {
        std::printf("communication path (%zu residues, cost %.4f):",
                    comm_path.size(), path_cost);
        for (std::size_t s = 0; s < comm_path.size(); ++s) std::printf(" %d", comm_path[s]);
        std::printf("\n");
        // The weakest link on the path: the hop with the largest -log|C| (lowest
        // |correlation|). It is the bottleneck of allosteric signal transmission.
        double weakest_w = -1.0;
        int weakest_a = -1, weakest_b = -1;
        for (std::size_t s = 0; s + 1 < comm_path.size(); ++s) {
            const int u = comm_path[s], v = comm_path[s + 1];
            const double w = comm_weight(C_gpu[static_cast<std::size_t>(u) * N + v]);
            if (w > weakest_w) { weakest_w = w; weakest_a = u; weakest_b = v; }
        }
        std::printf("bottleneck hop: %d-%d  |C| = %.4f\n",
                    weakest_a, weakest_b,
                    std::fabs(C_gpu[static_cast<std::size_t>(weakest_a) * N + weakest_b]));
    } else {
        std::printf("communication path: NONE (sites are in disconnected components)\n");
    }

    // A small fixed sample of the matrix so the demo's stdout exercises real
    // numbers (self-correlations are exactly 1.0000 by construction).
    std::printf("C diagonal sample:");
    for (int s = 0; s < 4 && s < N; ++s) {
        const int i = (s * (N - 1)) / 3;
        std::printf(" %.4f", C_gpu[static_cast<std::size_t>(i) * N + i]);
    }
    std::printf("\n");

    std::printf("RESULT: %s (GPU DCC matrix matches CPU exactly)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%d residues, %d frames)\n", path.c_str(), N, T);
    std::fprintf(stderr, "[timing] CPU DCC: %.3f ms   GPU DCC kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- the GPU computes all %lld matrix "
                         "entries concurrently; its edge grows with N and T.\n",
                 static_cast<long long>(N) * N);
    std::fprintf(stderr, "[verify] worst |C_cpu - C_gpu| = %.3e  (tolerance %.1e, exact)\n",
                 worst, MATRIX_TOLERANCE);

    // Exit code feeds the demo's pass/fail gate.
    return pass ? 0 : 1;
}
