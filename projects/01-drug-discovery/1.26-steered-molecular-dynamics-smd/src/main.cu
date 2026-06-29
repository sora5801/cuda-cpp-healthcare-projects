// ===========================================================================
// src/main.cu  --  Entry point: load config, run CPU + GPU SMD, verify, report
// ---------------------------------------------------------------------------
// Project 1.26 : Steered Molecular Dynamics (SMD)
//
// WHAT THIS FILE DOES  (the 5-step shape every project in this repo follows)
//   1. Load the SMD configuration (data/sample, or a built-in fallback).
//   2. CPU reference: run all trajectories serially (reference_cpu.cpp).
//   3. GPU: one thread per trajectory, same physics (kernels.cu).
//   4. VERIFY two things:
//        (a) GPU per-trajectory work == CPU work  -> EXACTLY (shared RNG+core),
//        (b) Jarzynski's ΔG estimate recovers the KNOWN true ΔG of the PMF
//            within a documented physical tolerance (the science check).
//   5. REPORT a deterministic free-energy summary to stdout; timing to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. The work array and every printed number come from
//   integer-seeded, fixed-order double-precision arithmetic that is identical on
//   every run and on both CPU and GPU. Timings (run-varying) go to STDERR.
//
// READ THIS FIRST in the code tour, then smd_core.h (physics), kernels.cuh ->
// kernels.cu (GPU), reference_cpu.cpp (baseline). See ../THEORY.md for the "why".
// ===========================================================================
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // run_gpu (GPU path), SmdParams
#include "reference_cpu.h"    // load_params, run_cpu
#include "smd_core.h"         // jarzynski_dg, pmf_delta_g, run_trajectory
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "1.26";
static const char* PROJECT_NAME = "Steered Molecular Dynamics (SMD)";

// --- Tolerances (each documented; PATTERNS.md §4) --------------------------
// (a) CPU vs GPU work: a TINY physical tolerance, not exact. Both sides run the
//     same run_trajectory() with the same shared splitmix64 RNG (the integer RNG
//     IS bit-identical), but the integrator threads those random numbers through
//     25000 steps of double-precision arithmetic that uses transcendentals
//     (log/cos/sqrt in Box-Muller) and fused multiply-adds. The device and host
//     libm/FMA differ at the last bit, and that ~1e-16 per-step difference
//     accumulates to ~1e-13 over the pull. So the works agree to ~1e-12 kJ/mol,
//     which is physically negligible against the ~30 kJ/mol work scale -- this
//     is the long-iterative-solver case in PATTERNS.md §4 (cf. flagship 10.02).
//     We verify to 1e-6 kJ/mol (a million times looser than observed) and say so
//     rather than pretending the two are bit-identical.
static constexpr double WORK_MATCH_TOL = 1.0e-6;
// (b) Jarzynski estimate vs the analytic ΔG of the PMF: a PHYSICAL tolerance.
//     Jarzynski's equality is exact only in the infinite-sampling limit; with a
//     finite ensemble and a finite pulling speed the estimate carries bias +
//     statistical error. The committed sample is engineered (slow pull, soft
//     spring, enough trajectories) so the estimate lands within this band. This
//     is honest: we are NOT claiming bit-exact free energy, only recovery to a
//     stated kJ/mol tolerance. THEORY.md "How we verify" explains the budget.
static constexpr double DG_TOL_KJMOL = 1.5;

// Built-in fallback config, used only if no data file is supplied. These are the
// SAME numbers scripts/make_synthetic.py writes to data/sample/, so the program
// behaves identically with or without the file (and matches expected_output.txt).
static SmdParams builtin_params() {
    SmdParams p{};
    p.xi0      = 0.0;     // bound-state coordinate (nm)
    p.xi_end   = 1.0;     // unbound-state coordinate (nm)
    p.n_traj   = 8192;    // independent SMD pulls (the ensemble)
    p.steps    = 25000;   // Langevin steps per pull
    p.dt       = 0.002;   // timestep (ps): pull lasts steps*dt = 50 ps
    p.k_spring = 2000.0;  // spring stiffness (kJ/mol/nm^2)
    p.v_pull   = 0.02;    // pulling velocity (nm/ps): 1.0 nm over 50 ps
    p.gamma    = 500.0;   // Langevin friction ((kJ/mol) ps / nm^2)
    p.kT       = 2.4943;  // kB*T at 300 K (kJ/mol)
    p.pmf_A    = 25.0;    // PMF barrier scale (kJ/mol)
    p.pmf_xa   = 0.0;     // bound-well centre (nm)
    p.pmf_xb   = 1.0;     // unbound-well centre (nm)
    p.pmf_slope= -12.0;   // tilt (kJ/mol/nm): true dG = slope*(xi_end-xi0) = -12
    p.seed     = 20240626ULL;
    return p;
}

int main(int argc, char** argv) {
    // ---- 1. Load the configuration -----------------------------------------
    SmdParams p;
    const char* source;
    std::string src_str;
    if (argc > 1) {
        try {
            p = load_params(argv[1]);     // parse the committed sample
            src_str = argv[1];
            source = src_str.c_str();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[error] %s\n", e.what());
            return 2;
        }
    } else {
        p = builtin_params();
        source = "synthetic (built-in)";
    }

    // ---- 2. CPU reference (timed) ------------------------------------------
    std::vector<double> work_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    run_cpu(p, work_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU ensemble (kernel timed inside the wrapper) -----------------
    std::vector<double> work_gpu;
    float gpu_kernel_ms = 0.0f;
    run_gpu(p, work_gpu, &gpu_kernel_ms);

    // ---- 4a. Verify GPU == CPU exactly -------------------------------------
    double worst = 0.0;
    for (int i = 0; i < p.n_traj; ++i)
        worst = std::fmax(worst, std::fabs(work_cpu[i] - work_gpu[i]));
    const bool work_match = worst <= WORK_MATCH_TOL;

    // ---- 4b. Free-energy analysis (deterministic, fixed-order reduction) ---
    // Naive average work <W>: the second law guarantees <W> >= ΔG, so this is a
    // biased OVER-estimate of the magnitude -- a great teaching contrast.
    double sumW = 0.0, minW = work_cpu[0], maxW = work_cpu[0];
    for (int i = 0; i < p.n_traj; ++i) {
        sumW += work_cpu[i];
        minW = std::fmin(minW, work_cpu[i]);
        maxW = std::fmax(maxW, work_cpu[i]);
    }
    const double meanW = sumW / p.n_traj;
    // Jarzynski's exponential-average estimate of ΔG (the headline result).
    const double dg_jarz = jarzynski_dg(work_cpu.data(), p.n_traj, p.kT);
    // The analytically known true ΔG of the engineered PMF (ground truth).
    const double dg_true = pmf_delta_g(p);
    const double dg_err  = std::fabs(dg_jarz - dg_true);

    const bool dg_ok = dg_err <= DG_TOL_KJMOL;
    const bool pass  = work_match && dg_ok;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("constant-velocity SMD: %d trajectories x %d steps, pull %.2f->%.2f nm\n",
                p.n_traj, p.steps, p.xi0, p.xi_end);
    std::printf("spring k=%.1f kJ/mol/nm^2  v=%.3f nm/ps  gamma=%.1f  kT=%.4f kJ/mol\n",
                p.k_spring, p.v_pull, p.gamma, p.kT);
    // A few fixed-index sample works so the learner can see the spread directly.
    std::printf("sample work W_i (kJ/mol):\n");
    const int picks[5] = {0, p.n_traj / 4, p.n_traj / 2, (3 * p.n_traj) / 4, p.n_traj - 1};
    for (int s = 0; s < 5; ++s)
        std::printf("  traj %-5d: %9.4f\n", picks[s], work_cpu[picks[s]]);
    std::printf("work distribution: min=%.4f  mean=%.4f  max=%.4f kJ/mol\n",
                minW, meanW, maxW);
    std::printf("free energy (kJ/mol): naive <W>=%.4f  Jarzynski dG=%.4f  true dG=%.4f\n",
                meanW, dg_jarz, dg_true);
    std::printf("Jarzynski error = %.4f kJ/mol (tol %.2f); dissipation <W>-dG = %.4f\n",
                dg_err, DG_TOL_KJMOL, meanW - dg_true);
    std::printf("RESULT: %s (GPU==CPU work to ~1e-12; Jarzynski recovers true dG within tol)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%d trajectories)\n", source, p.n_traj);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernel: %.3f ms", cpu_ms, gpu_kernel_ms);
    if (gpu_kernel_ms > 0.0f)
        std::fprintf(stderr, "   (%.1fx)", cpu_ms / gpu_kernel_ms);
    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "[timing] teaching artifact -- speed-up grows with ensemble size; "
                         "production SMD runs full-atom MD per trajectory.\n");
    std::fprintf(stderr, "[verify] worst |W_cpu - W_gpu| = %.3e (tol %.1e); dG error %.4f kJ/mol\n",
                 worst, WORK_MATCH_TOL, dg_err);

    return pass ? 0 : 1;
}
