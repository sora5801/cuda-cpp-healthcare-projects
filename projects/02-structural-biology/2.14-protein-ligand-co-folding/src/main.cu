// ===========================================================================
// src/main.cu  --  Entry point: load complex, run CPU + GPU co-folding, verify
// ---------------------------------------------------------------------------
// Project 2.14 : Protein-Ligand Co-Folding (reduced-scope teaching version)
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the protein-ligand complex (native targets + types) and the
//      diffusion schedule from data/sample, then build the noised start x_T.
//   2. Run the CPU reference reverse diffusion (reference_cpu.cpp) -> baseline.
//   3. Run the GPU reverse diffusion (kernels.cu, per-step attention) -> taught.
//   4. VERIFY: the GPU final positions match the CPU's within tolerance AND the
//      recovered RMSD-to-native is small (the science-level success check).
//   5. REPORT: deterministic pose summary to stdout; timing to stderr.
//
//   STDOUT is kept byte-for-byte deterministic so demo/run_demo can diff it
//   against demo/expected_output.txt. Run-varying numbers (timings) go to
//   STDERR, which the demo shows but does not diff (PATTERNS.md §3).
//
// READ THIS FIRST in the code tour, then cofold.h (the math) ->
// reference_cpu.cpp (baseline) -> kernels.cuh -> kernels.cu (the GPU twin).
// See ../THEORY.md for the "why".
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // simulate_gpu (GPU path), Complex, CofoldParams
#include "reference_cpu.h"    // load_complex, init_positions, simulate_cpu
#include "cofold.h"           // rmsd_to_target, D_POS
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "2.14";
static const char* PROJECT_NAME = "Protein-Ligand Co-Folding";

// Correctness tolerance for GPU-vs-CPU final positions. The reverse diffusion is
// a LONG iterative loop in double precision; the GPU's tree-order softmax
// reduction and FMA contraction differ from the CPU's strict left-to-right sum,
// so the two drift by ~1e-4 over the schedule. We verify to a physically
// negligible 1e-3 (coordinates are O(10) Angstrom), and ALSO check the science
// metric (recovered RMSD). See THEORY "Numerical considerations", PATTERNS.md §4.
static constexpr double POS_TOLERANCE = 1.0e-3;

// Science-level success: a correct reverse diffusion should drive the noised
// cloud back to the native complex, so the final RMSD-to-native must be well
// below the starting RMSD. We assert it falls under this (Angstrom) threshold;
// the sample is engineered so the planted complex is recoverable.
static constexpr double RMSD_TARGET = 0.5;

int main(int argc, char** argv) {
    // ---- 1. Load the complex + build the noised start ----------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/complex_sample.txt";
    Complex C;
    try {
        C = load_complex(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    std::vector<double> pos0;            // the shared noised starting positions
    init_positions(C, pos0);
    const double start_rmsd = rmsd_to_target(pos0.data(), C.target.data(), C.P.n_tokens);

    // ---- 2. CPU reference reverse diffusion (timed) ------------------------
    std::vector<double> pos_cpu = pos0;  // copy: each path starts from the same x_T
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    simulate_cpu(C, pos_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU reverse diffusion (loop timed inside the wrapper) ----------
    std::vector<double> pos_gpu = pos0;
    float gpu_kernel_ms = 0.0f;
    simulate_gpu(C, pos_gpu, &gpu_kernel_ms);

    // ---- 4. Verify: GPU==CPU positions, and recovered RMSD ----------------
    double worst = 0.0;
    for (std::size_t i = 0; i < pos_cpu.size(); ++i)
        worst = std::fmax(worst, std::fabs(pos_cpu[i] - pos_gpu[i]));
    const double final_rmsd = rmsd_to_target(pos_gpu.data(), C.target.data(), C.P.n_tokens);
    const bool match  = worst <= POS_TOLERANCE;
    const bool folded = final_rmsd <= RMSD_TARGET;
    const bool pass   = match && folded;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("[reduced-scope teaching model: analytic-score diffusion, not a learned network]\n");
    std::printf("complex: %d tokens (%d protein + %d ligand), %d denoising steps\n",
                C.P.n_tokens, C.P.n_protein, C.P.n_ligand, C.P.steps);
    std::printf("schedule: temp=%.3f step_frac=%.3f type_bias=%.3f noise_scale=%.3f\n",
                C.P.temp, C.P.step_frac, C.P.type_bias, C.P.noise_scale);
    std::printf("RMSD to native: start=%.4f  ->  final=%.4f (Angstrom)\n",
                start_rmsd, final_rmsd);
    // Print the final ligand-atom positions (the binding pose -- the deliverable
    // a docking/co-folding tool reports). Deterministic to 4 decimals.
    std::printf("ligand pose (final x y z):\n");
    for (int i = C.P.n_protein; i < C.P.n_tokens; ++i) {
        std::printf("  atom %d: %.4f %.4f %.4f\n", i - C.P.n_protein,
                    pos_gpu[i * D_POS + 0], pos_gpu[i * D_POS + 1], pos_gpu[i * D_POS + 2]);
    }
    std::printf("RESULT: %s (GPU==CPU within %.0e; pose folded RMSD<%.1f)\n",
                pass ? "PASS" : "FAIL", POS_TOLERANCE, RMSD_TARGET);

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%d tokens, %d steps)\n",
                 path.c_str(), C.P.n_tokens, C.P.steps);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU loop: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- at this toy token count the per-step "
                         "attention is launch-bound; the GPU's edge grows with sequence length.\n");
    std::fprintf(stderr, "[verify] worst |GPU-CPU| position diff = %.3e  (tolerance %.1e)\n",
                 worst, POS_TOLERANCE);
    std::fprintf(stderr, "[verify] final RMSD = %.4f  (target < %.1f)\n", final_rmsd, RMSD_TARGET);

    return pass ? 0 : 1;
}
