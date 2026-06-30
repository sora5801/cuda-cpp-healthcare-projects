// ===========================================================================
// src/main.cu  --  Entry point: load data, run CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 2.23 : Protein-Ligand Interaction Energy Decomposition
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the system + trajectory (data/sample/complex_sample.txt).
//   2. Compute the CPU reference (reference_cpu.cpp)         -> trusted answer.
//   3. Compute the GPU result    (kernels.cu)                -> the thing taught.
//   4. VERIFY: assert GPU agrees with CPU within a tolerance -> correctness.
//   5. REPORT: deterministic per-residue decomposition + hot-spot ranking to
//      stdout; timing + run-varying detail to stderr.
//
//   STDOUT is kept byte-for-byte deterministic so demo/run_demo can diff it
//   against demo/expected_output.txt. Anything that varies run-to-run (timings)
//   goes to STDERR, which the demo shows but does not diff.
//
// READ THIS FIRST in the code tour, then mmgbsa.h (the physics), kernels.cuh ->
// kernels.cu (the GPU path), and reference_cpu.cpp (the baseline). The science
// and GPU mapping are in ../THEORY.md.
// ===========================================================================
#include <algorithm>   // std::sort
#include <cstdio>
#include <numeric>     // std::iota
#include <string>
#include <vector>

#include "kernels.cuh"        // decompose_gpu (GPU path)
#include "reference_cpu.h"    // load_system, decompose_cpu, MmgbsaSystem (-> mmgbsa.h)
#include "util/io.hpp"        // util::CpuTimer

// Self-identification. These MUST stay in sync with demo/expected_output.txt.
static const char* PROJECT_ID   = "2.23";
static const char* PROJECT_NAME = "Protein-Ligand Interaction Energy Decomposition";

// Correctness tolerance (kcal/mol). CPU and GPU run the SAME double-precision
// formula (mmgbsa.h) in the SAME loop order, so they agree to ~1e-12; the only
// divergence is the GPU's fused-multiply-add (FMA) contraction in the sqrt/exp
// chains, which is at the 1e-9 level for these magnitudes. 1e-4 kcal/mol is a
// physically negligible, honest tolerance (PATTERNS.md sec 4; THEORY "verify").
static constexpr double TOLERANCE = 1.0e-4;

// max_abs_err over the four components of every residue. We roll our own (the
// util helper compares float vectors; our results are doubles in a struct).
static double max_abs_err(const std::vector<PerResidueEnergy>& a,
                          const std::vector<PerResidueEnergy>& b) {
    if (a.size() != b.size()) return 1e300;   // shape mismatch -> "infinitely wrong"
    double worst = 0.0;
    auto upd = [&](double x, double y) {
        double d = x > y ? x - y : y - x;     // |x - y| without <cmath> for clarity
        if (d > worst) worst = d;
    };
    for (std::size_t i = 0; i < a.size(); ++i) {
        upd(a[i].elec, b[i].elec);
        upd(a[i].vdw,  b[i].vdw);
        upd(a[i].gb,   b[i].gb);
        upd(a[i].total, b[i].total);
    }
    return worst;
}

int main(int argc, char** argv) {
    // ---- 1. Load the system + trajectory -----------------------------------
    const std::string path = (argc > 1) ? argv[1]
                                        : "data/sample/complex_sample.txt";
    MmgbsaSystem sys;
    try {
        sys = load_system(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) ------------------------------------------
    std::vector<PerResidueEnergy> dec_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    decompose_cpu(sys, dec_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU result (kernel timed inside the wrapper) -------------------
    std::vector<PerResidueEnergy> dec_gpu;
    float gpu_kernel_ms = 0.0f;
    decompose_gpu(sys, dec_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ----------------------------------------------------------
    const double err = max_abs_err(dec_cpu, dec_gpu);
    const bool pass = err <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("system: %d residues, %d ligand atoms, %d frames, cutoff %.1f A\n",
                sys.M, sys.L, sys.F, sys.cutoff);
    std::printf("per-residue MM-GBSA decomposition (trajectory-averaged, kcal/mol):\n");
    std::printf("  %-8s %10s %10s %10s %10s\n", "residue", "elec", "vdw", "gb", "total");
    for (int m = 0; m < sys.M; ++m) {
        // The residue label comes from the params; the four energies from the GPU
        // result (verified equal to the CPU above). %+10.4f keeps the columns
        // aligned and the sign explicit (attractive contributions are negative).
        std::printf("  %-8s %+10.4f %+10.4f %+10.4f %+10.4f\n",
                    sys.res[m].name, dec_gpu[m].elec, dec_gpu[m].vdw,
                    dec_gpu[m].gb, dec_gpu[m].total);
    }

    // Hot-spot ranking: the residues with the most FAVOURABLE (most negative)
    // total contribution are the binding hot spots a medicinal chemist targets.
    // Sort an index array by total ascending (most negative first); ties broken
    // by lower index so the ranking is deterministic.
    std::vector<int> order(static_cast<std::size_t>(sys.M));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int i, int j) {
        if (dec_gpu[i].total != dec_gpu[j].total)
            return dec_gpu[i].total < dec_gpu[j].total;   // most negative first
        return i < j;                                     // tie -> lower index
    });
    const int top = sys.M < 3 ? sys.M : 3;                // show up to 3 hot spots
    std::printf("top-%d binding hot-spot residues (most favorable total):\n", top);
    for (int r = 0; r < top; ++r) {
        const int m = order[r];
        std::printf("  #%d  %-8s  total = %+10.4f kcal/mol\n",
                    r + 1, sys.res[m].name, dec_gpu[m].total);
    }
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-04 kcal/mol)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (F=%d frames, M=%d residues, L=%d ligand atoms)\n",
                 path.c_str(), sys.F, sys.M, sys.L);
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny system is dominated by "
                         "launch/copy overhead; the GPU wins at real scale (hundreds of residues "
                         "x thousands of frames).\n");
    std::fprintf(stderr, "[verify] max_abs_err = %.3e kcal/mol  (tolerance %.1e)\n", err, TOLERANCE);

    // Exit code feeds the demo's pass/fail gate.
    return pass ? 0 : 1;
}
