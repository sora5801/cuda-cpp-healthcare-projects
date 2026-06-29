// ===========================================================================
// src/main.cu  --  Entry point: load, titrate on CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 1.22 : Constant-pH Molecular Dynamics (reduced-scope teaching model)
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the titration problem (data/sample, else a built-in synthetic one).
//   2. CPU reference titration (reference_cpu.cpp)         -> trusted answer.
//   3. GPU ensemble titration  (kernels.cu)                -> the thing taught.
//   4. VERIFY: the integer protonation tallies match EXACTLY (atomics commute
//      on integers, and both sides run identical RNG-seeded chains).
//   5. REPORT: deterministic titration curves + predicted pKa to STDOUT;
//      timings + the analytic sanity check to STDERR.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Anything run-to-run variable (timings) goes to
//   STDERR, which the demo shows but does not diff (PATTERNS.md §3).
//
// READ THIS FIRST in the code tour, then cph_core.h (the physics), kernels.cuh
// -> kernels.cu (the GPU mapping), and reference_cpu.cpp (the baseline). See
// ../THEORY.md for the "why".
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // titrate_gpu (GPU path), CphProblem/CphResult
#include "reference_cpu.h"    // load_cph_problem, titrate_cpu, estimate_pKa
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "1.22";
static const char* PROJECT_NAME = "Constant-pH Molecular Dynamics";

// ---------------------------------------------------------------------------
// make_synthetic: a built-in fallback problem identical to the committed sample,
// used when no data file is supplied. Three titratable residues placed on a line
// so the electrostatic coupling is easy to reason about:
//   * ASP (acid, intrinsic pKa 4.0):  protonated q=0, deprotonated q=-1
//   * HIS (base, intrinsic pKa 6.5):  protonated q=+1, deprotonated q=0
//   * LYS (base, intrinsic pKa 10.5): protonated q=+1, deprotonated q=0
// They sit 5 A apart. Coupling is ON (coulomb_k>0), so the printed pKa values
// shift away from the intrinsic ones -- the lesson of the project. The exact
// numbers are reproduced by demo/expected_output.txt from a real run.
// ---------------------------------------------------------------------------
static void make_synthetic(CphProblem& p) {
    p.sys.n_res     = 3;
    p.sys.coulomb_k = 12.0;   // 332.06/epsilon, eff. dielectric ~28 (partly solvated)
    p.sys.kT        = 0.593;  // k_B*T at ~298 K, kcal/mol
    p.sys.sweeps    = 6000;   // MC sweeps per chain
    p.sys.burn_in   = 1000;   // discard the first 1000 sweeps (equilibration)
    p.pH_min   = 0.0;
    p.pH_max   = 14.0;
    p.n_pH     = 15;          // pH = 0,1,...,14 (integer grid for a clean readout)
    p.replicas = 8;           // 8 independent chains averaged per pH
    p.seed     = 20260628ULL;
    //                pKa  q_prot q_deprot   x    y    z
    p.sys.res[0] = {  4.0,  0.0,  -1.0,    0.0, 0.0, 0.0 };  // ASP (acid)
    p.sys.res[1] = {  6.5,  1.0,   0.0,    7.0, 0.0, 0.0 };  // HIS (base)
    p.sys.res[2] = { 10.5,  1.0,   0.0,   14.0, 0.0, 0.0 };  // LYS (base)
}

// short residue labels for the report (purely cosmetic; index order = file order)
static const char* residue_label(int i) {
    static const char* L[] = {"ASP", "HIS", "LYS", "GLU", "CYS",
                              "R5", "R6", "R7", "R8", "R9",
                              "R10","R11","R12","R13","R14","R15"};
    return L[i];
}

int main(int argc, char** argv) {
    // ---- 1. Load ------------------------------------------------------------
    CphProblem prob;
    const char* source = "synthetic (built-in)";
    if (argc > 1) {
        try {
            prob = load_cph_problem(argv[1]);
            source = argv[1];
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[error] %s\n", e.what());
            return 2;
        }
    } else {
        make_synthetic(prob);
    }

    // ---- 2. CPU reference (timed) ------------------------------------------
    CphResult res_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    titrate_cpu(prob, res_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU ensemble (kernel timed inside the wrapper) -----------------
    CphResult res_gpu;
    float gpu_kernel_ms = 0.0f;
    titrate_gpu(prob, res_gpu, &gpu_kernel_ms);

    // ---- 4. Verify: EXACT integer match ------------------------------------
    // Both sides ran the identical RNG-seeded chains and tallied integer counts,
    // so every slot must be equal. A single mismatch is a real bug, not noise.
    int mismatches = 0;
    for (size_t s = 0; s < res_cpu.prot_count.size(); ++s)
        if (res_cpu.prot_count[s] != res_gpu.prot_count[s]) ++mismatches;
    const bool pass = (mismatches == 0) &&
                      (res_cpu.tallied_per_pH == res_gpu.tallied_per_pH);

    const int n_res = prob.sys.n_res;
    const double denom = static_cast<double>(res_gpu.tallied_per_pH);

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("reduced-scope teaching model: ensemble Metropolis MC titration\n");
    std::printf("residues = %d, pH grid = %.1f..%.1f in %d steps, replicas = %d\n",
                n_res, prob.pH_min, prob.pH_max, prob.n_pH, prob.replicas);
    std::printf("coupling k = %.1f kcal*A/mol/e^2, kT = %.3f kcal/mol, "
                "sweeps = %d (burn-in %d)\n",
                prob.sys.coulomb_k, prob.sys.kT, prob.sys.sweeps, prob.sys.burn_in);

    // Titration curves: fraction protonated f(pH) per residue, as integer
    // percent so the printed line is exactly reproducible (round-half-up on an
    // integer ratio is deterministic). Header row lists the pH grid.
    std::printf("\nfraction protonated (%%), rows = residue, cols = pH:\n");
    std::printf("        ");
    for (int k = 0; k < prob.n_pH; ++k) {
        const double pH = prob.pH_min + (prob.pH_max - prob.pH_min) * k / (prob.n_pH - 1);
        std::printf(" pH%4.1f", pH);
    }
    std::printf("\n");
    for (int i = 0; i < n_res; ++i) {
        std::printf("%-4s    ", residue_label(i));
        for (int k = 0; k < prob.n_pH; ++k) {
            const uint64_t c = res_gpu.prot_count[static_cast<size_t>(k) * n_res + i];
            // integer percent, rounded half-up: (100*c + denom/2) / denom
            const long long pct = static_cast<long long>(
                (100.0 * c + denom * 0.5) / denom);
            std::printf("  %4lld", pct);
        }
        std::printf("\n");
    }

    // Predicted (coupling-shifted) pKa per residue vs the intrinsic input.
    std::printf("\npredicted pKa (curve crosses 50%%) vs intrinsic:\n");
    for (int i = 0; i < n_res; ++i) {
        const double pka = estimate_pKa(prob, res_gpu, i);
        const double shift = pka - prob.sys.res[i].pKa_intrinsic;
        if (std::isnan(pka))
            std::printf("  %-4s intrinsic %5.2f  ->  pKa  (off-grid)\n",
                        residue_label(i), prob.sys.res[i].pKa_intrinsic);
        else
            std::printf("  %-4s intrinsic %5.2f  ->  pKa %5.2f  (shift %+5.2f)\n",
                        residue_label(i), prob.sys.res[i].pKa_intrinsic, pka, shift);
    }

    std::printf("\nRESULT: %s (GPU protonation tally matches CPU exactly)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s\n", source);
    std::fprintf(stderr, "[timing] CPU titration: %.3f ms   GPU titration: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    if (gpu_kernel_ms > 0.0)
        std::fprintf(stderr, "[timing] speed-up ~%.1fx (teaching artifact; grows with "
                             "ensemble size n_pH*replicas).\n", cpu_ms / gpu_kernel_ms);
    std::fprintf(stderr, "[verify] tally-slot mismatches = %d / %zu "
                         "(integer counts => atomics commute => exact)\n",
                 mismatches, res_cpu.prot_count.size());
    // Analytic sanity check: with coupling OFF a residue must titrate at its
    // intrinsic pKa (Henderson-Hasselbalch). We don't rerun here, but we report
    // the residual coupling shift so the learner sees the effect's magnitude.
    std::fprintf(stderr, "[science] non-zero coupling shifts pKa from intrinsic; "
                         "set coulomb_k=0 in the sample to recover H-H exactly.\n");

    return pass ? 0 : 1;
}
