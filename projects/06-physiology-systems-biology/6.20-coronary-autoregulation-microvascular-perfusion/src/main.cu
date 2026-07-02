// ===========================================================================
// src/main.cu  --  Entry point: solve coronary perfusion on CPU + GPU, verify
// ---------------------------------------------------------------------------
// Project 6.20 : Coronary Autoregulation & Microvascular Perfusion
//
// 5-STEP SHAPE (the standard teaching flow)
//   1. Load the synthetic coronary microvascular network (data/sample).
//   2. CPU reference: autoregulation loop of sparse-SPD CG solves (reference_cpu).
//   3. GPU: the SAME loop, each solve done with CSR-SpMV Conjugate Gradient
//      (kernels.cu). Both call the identical per-vessel physics in coronary.h.
//   4. VERIFY: the GPU nodal pressures match the CPU pressures within a
//      documented tolerance (see TOLERANCE below).
//   5. REPORT: a deterministic summary to stdout -- pressures at each node,
//      total perfusion, the modeled stenosis's FFR, and PASS/FAIL. Timings and
//      run-varying detail go to stderr (so stdout is byte-stable for the demo).
//
// THE PHENOMENON: because arteriolar radius feeds conductance as r^4, a small
// autoregulatory dilation dramatically raises flow. The demo shows perfusion
// being driven toward the metabolic set-point across the outer iterations, and
// computes a virtual Fractional Flow Reserve (FFR = distal/proximal pressure)
// across a modeled stenosis -- the clinical read-out this class of model targets.
//
// Code tour: start here, then coronary.h (the physics), reference_cpu.cpp (the
// serial solve), kernels.cu (the GPU solve). Depth is in ../THEORY.md.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // solve_gpu
#include "reference_cpu.h"    // Network, Solution, load_network, solve_cpu
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "6.20";
static const char* PROJECT_NAME = "Coronary Autoregulation & Microvascular Perfusion";

// Solve parameters. n_autoreg outer feedback steps; CG tolerance is a RELATIVE
// residual. These are fixed so stdout is deterministic.
static constexpr int    N_AUTOREG    = 8;        // autoregulation iterations
static constexpr double CG_TOL       = 1.0e-10;  // CG relative-residual stop
static constexpr int    CG_MAX_ITER  = 2000;     // safety cap (never hit here)

// Verification tolerance on nodal pressures (mmHg). CPU and GPU run the SAME
// double-precision CG, but the CSR SpMV sums a node's incident edges in a
// different order than the CPU's edge loop, so FMA/rounding diverges by ~1e-11
// per operation and can accumulate over the CG iterations. A few 1e-8 mmHg is
// physically negligible (pressures are ~10-100 mmHg). See PATTERNS.md §4 and
// THEORY §numerics -- we verify to a small physical tolerance and say so.
static constexpr double TOLERANCE = 1.0e-6;

int main(int argc, char** argv) {
    // ---- 1. Load ----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1]
                                        : "data/sample/coronary_network.txt";
    Network net_cpu;
    try {
        net_cpu = load_network(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    // The autoregulation loop MUTATES radii, so give CPU and GPU independent
    // copies of the network -- otherwise the second solver would start from the
    // first solver's regulated geometry.
    Network net_gpu = net_cpu;

    // ---- 2. CPU reference (timed) -----------------------------------------
    Solution sol_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    solve_cpu(net_cpu, N_AUTOREG, CG_TOL, CG_MAX_ITER, sol_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU solve (device-timed) --------------------------------------
    Solution sol_gpu;
    float gpu_ms = 0.0f;
    solve_gpu(net_gpu, N_AUTOREG, CG_TOL, CG_MAX_ITER, sol_gpu, &gpu_ms);

    // ---- 4. Verify (nodal pressures agree) --------------------------------
    double perr = 0.0;
    for (int i = 0; i < net_cpu.n_nodes; ++i) {
        const double d = std::fabs(sol_cpu.p[i] - sol_gpu.p[i]);
        if (d > perr) perr = d;
    }
    const bool pass = perr <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    // We print the GPU solution's numbers (verified to match the CPU).
    //
    // Virtual FFR across the tagged stenosis. Clinically FFR is defined relative
    // to venous pressure Pv:  FFR = (Pd - Pv) / (Pa - Pv), where Pa is aortic
    // (inlet) and Pd is distal-to-lesion. A value < 0.80 is the guideline
    // cut-point for a flow-limiting lesion. This is a synthetic teaching read-out.
    const double p_dist = sol_gpu.p[net_gpu.ffr_dist];
    double pv = 1e300;
    for (int i = 0; i < net_gpu.n_nodes; ++i)
        if (net_gpu.is_fixed[i] && net_gpu.fixed_p[i] < pv) pv = net_gpu.fixed_p[i];
    const double pa  = net_gpu.aortic_p;
    const double ffr = (pa - pv) != 0.0 ? (p_dist - pv) / (pa - pv) : 0.0;

    // Total inlet perfusion = net flow leaving the inlet node (shared helper).
    const double perfusion = inlet_perfusion(net_gpu, sol_gpu.q);

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("network: %d nodes, %d segments, hematocrit=%.2f, aortic P=%.1f mmHg\n",
                net_gpu.n_nodes, net_gpu.n_segs, net_gpu.hct, net_gpu.aortic_p);
    std::printf("autoregulation steps: %d   CG tol: %.1e\n", N_AUTOREG, CG_TOL);
    std::printf("nodal pressures (mmHg):\n");
    for (int i = 0; i < net_gpu.n_nodes; ++i)
        std::printf(" P[%d]=%.4f", i, sol_gpu.p[i]);
    std::printf("\n");
    // Show autoregulation at work: perfusion right after the first solve vs the
    // regulated steady state. The pre-autoreg value is LARGE (the un-regulated
    // network is grossly over-perfused), and its low-order digits are sensitive
    // to compiler FMA choices, so we print it in scientific notation at modest
    // precision -- stable across Debug/Release and honest about its significance.
    // The regulated value is small and prints exactly; it is the headline result.
    std::printf("inlet perfusion (pre-autoreg)  = %.4e (um^3/s)\n", sol_gpu.perfusion_first);
    std::printf("inlet perfusion (regulated)    = %.4f (um^3/s)\n", perfusion);
    std::printf("stenosis segment       = %d (nodes %d->%d)\n",
                net_gpu.ffr_seg, net_gpu.ffr_prox, net_gpu.ffr_dist);
    std::printf("virtual FFR (Pd/Pa)    = %.4f  [%s]\n",
                ffr, ffr < 0.80 ? "flow-limiting (<0.80)" : "non-significant");
    std::printf("RESULT: %s (GPU pressures match CPU within tol=%.1e mmHg)\n",
                pass ? "PASS" : "FAIL", TOLERANCE);

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d nodes, %d segments)\n",
                 path.c_str(), net_gpu.n_nodes, net_gpu.n_segs);
    std::fprintf(stderr, "[solve]  cold-start CG iters -- CPU: %d   GPU: %d   "
                         "(later autoregulation solves warm-start -> ~0 iters)\n",
                 sol_cpu.cg_iters_first, sol_gpu.cg_iters_first);
    std::fprintf(stderr, "[solve]  final CG residual -- CPU: %.2e   GPU: %.2e\n",
                 sol_cpu.cg_resid, sol_gpu.cg_resid);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU(all solves): %.3f ms\n", cpu_ms, gpu_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- on this tiny network the GPU is launch-bound; "
                         "its edge grows with 10^4-10^6 segment networks (see THEORY).\n");
    std::fprintf(stderr, "[verify] max |P_cpu - P_gpu| = %.3e mmHg  (tolerance %.1e)\n", perr, TOLERANCE);

    return pass ? 0 : 1;
}
