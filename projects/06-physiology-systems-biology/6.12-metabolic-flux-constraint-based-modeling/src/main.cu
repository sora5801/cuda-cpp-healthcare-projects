// ===========================================================================
// src/main.cu  --  Entry point: load model, run CPU + GPU knockout screen, verify
// ---------------------------------------------------------------------------
// Project 6.12 : Metabolic Flux / Constraint-Based Modeling
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the metabolic model (data/sample/*.txt, or a required file arg).
//   2. CPU reference: solve the wild-type FBA LP + every single-reaction
//      knockout, serially (reference_cpu.cpp)              -> trusted answer.
//   3. GPU: the same screen, one LP per thread (kernels.cu) -> the thing taught.
//   4. VERIFY: assert GPU objectives match CPU within tolerance.
//   5. REPORT: deterministic result (wild-type growth, per-knockout growth,
//      essential-gene count) to STDOUT; timing/verify detail to STDERR.
//
//   STDOUT is byte-for-byte deterministic (integer-valued fluxes on this model,
//   printed at fixed precision) so demo/run_demo can diff it against
//   demo/expected_output.txt. Timings (which vary run to run) go to STDERR.
//
// CODE TOUR: read this first, then fba.h (the LP + solver), reference_cpu.cpp
// (the CPU screen), kernels.cuh -> kernels.cu (the GPU screen). See ../THEORY.md
// for the science, math, and GPU-mapping "why".
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // screen_gpu, FbaModel, FbaResult
#include "reference_cpu.h"    // load_model, screen_cpu
#include "util/io.hpp"        // util::CpuTimer

// Identity tokens (kept in sync with demo/expected_output.txt).
static const char* PROJECT_ID   = "6.12";
static const char* PROJECT_NAME = "Metabolic Flux / Constraint-Based Modeling";

// Verification tolerance on the predicted growth (objective) of each solve.
//   Both the CPU and GPU call the identical double-precision simplex in fba.h, so
//   in principle the objectives are bit-identical. We nonetheless allow a tiny
//   1e-9 slack: the GPU may fuse a multiply-add (FMA) that the host compiler
//   emits as two rounded operations, so a reduced cost could differ in its last
//   bit. On this well-conditioned integer-stoichiometry model that never changes
//   a pivot decision, and the observed difference is 0 -- but we document the
//   honest floor rather than pretend the two float pipelines are identical
//   (PATTERNS.md section 4). If the model were ill-conditioned, an epsilon-close
//   reduced cost COULD flip a pivot and produce a genuinely different (still
//   optimal, but different) vertex; THEORY.md discusses this degeneracy caveat.
static constexpr double TOLERANCE = 1.0e-9;

// A knockout counts as LETHAL/essential if the mutant's growth is below this
// fraction of the wild-type growth. 1% is the conventional COBRA cutoff.
static constexpr double LETHAL_FRACTION = 0.01;

int main(int argc, char** argv) {
    // ---- 1. Load the model -------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/toy_core_model.txt";
    FbaModel model;
    std::vector<std::string> names;
    try {
        model = load_model(path, names);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int nrxn  = model.nrxn;
    const int njobs = nrxn + 1;                // knockouts + wild type

    // ---- 2. CPU reference screen (timed) -----------------------------------
    std::vector<FbaResult> res_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    screen_cpu(model, res_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU screen (kernel timed) --------------------------------------
    std::vector<FbaResult> res_gpu;
    float gpu_kernel_ms = 0.0f;
    screen_gpu(model, res_gpu, &gpu_kernel_ms);

    // ---- 4. Verify: every objective agrees CPU vs GPU ----------------------
    double worst = 0.0;
    for (int i = 0; i < njobs; ++i)
        worst = std::fmax(worst, std::fabs(res_cpu[i].objective - res_gpu[i].objective));
    const bool pass = worst <= TOLERANCE;

    // Wild-type growth is the LAST job (index nrxn) in both arrays.
    const double wt = res_gpu[nrxn].objective;

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("model: %d metabolites x %d reactions   (SYNTHETIC toy network)\n",
                model.nmet, nrxn);
    std::printf("wild-type max biomass flux = %.4f\n", wt);
    std::printf("single-reaction knockout screen (growth as %% of wild type):\n");

    // Per-knockout line: name, mutant growth, %WT, and a class label. We classify
    // ESSENTIAL (growth ~ 0), REDUCED (some loss), or NEUTRAL (no change). This is
    // the biological read-out: essential reactions are candidate drug targets.
    int n_essential = 0, n_reduced = 0, n_neutral = 0;
    for (int k = 0; k < nrxn; ++k) {
        const double g   = res_gpu[k].objective;
        const double pct = (wt > 0.0) ? 100.0 * g / wt : 0.0;
        const char*  cls;
        if (g <= LETHAL_FRACTION * wt)        { cls = "ESSENTIAL"; ++n_essential; }
        else if (g < wt - 1e-6)               { cls = "reduced";   ++n_reduced;   }
        else                                  { cls = "neutral";   ++n_neutral;   }
        std::printf("  KO %-12s biomass=%.4f  (%6.2f%% WT)  %s\n",
                    names[static_cast<std::size_t>(k)].c_str(), g, pct, cls);
    }
    std::printf("summary: %d essential, %d growth-reducing, %d neutral reactions\n",
                n_essential, n_reduced, n_neutral);
    std::printf("RESULT: %s (GPU screen matches CPU within tol=1.0e-09)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d LPs solved: %d knockouts + wild type)\n",
                 path.c_str(), njobs, nrxn);
    std::fprintf(stderr, "[timing] CPU screen: %.3f ms   GPU screen: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- with a handful of tiny LPs the GPU is "
                         "launch-bound; the win appears at 10^3-10^5 knockouts/conditions.\n");
    std::fprintf(stderr, "[verify] worst |CPU-GPU| objective diff = %.3e  (tolerance %.1e)\n",
                 worst, TOLERANCE);
    // Report the wild-type solver status/iterations as a numerics diagnostic.
    std::fprintf(stderr, "[solver] wild-type simplex: status=%d iters=%d\n",
                 res_gpu[nrxn].status, res_gpu[nrxn].iters);

    return pass ? 0 : 1;
}
