// ===========================================================================
// src/main.cu  --  Entry point: load a molecule, compute SASA, verify, report
// ---------------------------------------------------------------------------
// Project 1.31 : Solvent-Accessible Surface Area (SASA) on GPU
//
// The 5-step shape every project in this repo follows:
//   1. Load the problem (a molecule = atom list from data/sample).
//   2. CPU reference  (reference_cpu.cpp)  -> trusted per-atom SASA.
//   3. GPU compute    (kernels.cu)         -> the thing being taught.
//   4. VERIFY: the GPU's per-atom EXPOSED-POINT COUNTS match the CPU EXACTLY
//      (integers), and the derived areas agree to a tiny float tolerance.
//   5. REPORT: deterministic totals + top-exposed atoms to stdout; timing to
//      stderr.
//
// STDOUT is byte-for-byte deterministic (diffed by demo/run_demo against
// demo/expected_output.txt); run-to-run timings go to STDERR.
//
// Code tour: start here, then sasa_core.h (the shared math), kernels.cuh ->
// kernels.cu, then reference_cpu.*. The "why" is in ../THEORY.md.
// ===========================================================================
#include <algorithm>
#include <cmath>      // std::fabs (per-atom area error)
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "kernels.cuh"        // sasa_gpu, Molecule, Atom
#include "reference_cpu.h"    // load_molecule, sasa_cpu
#include "sasa_core.h"        // N_SPHERE_POINTS, PROBE_RADIUS (for the report)
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "1.31";
static const char* PROJECT_NAME = "Solvent-Accessible Surface Area (SASA) on GPU";

// Two tolerances, each justified (PATTERNS.md sec 4):
//   * EXPOSED COUNTS are integers computed by the identical shared function on
//     both sides -> they must match EXACTLY (a mismatch is a real bug). We assert
//     count_mismatches == 0.
//   * The per-atom AREA is exposed*(4*pi*r^2/N): the same double multiply on both
//     sides, so it agrees to the last ULP. We still check it against a tiny
//     tolerance as a guard; 1e-9 Angstrom^2 is far below any physical relevance
//     and below double round-off for these magnitudes.
static constexpr double AREA_TOL = 1.0e-9;   // Angstrom^2, area agreement guard

// Sum a double vector left-to-right. The SAME deterministic order is used for the
// CPU and GPU totals, so the two totals are computed identically -> the printed
// total is reproducible run-to-run (PATTERNS.md sec 3). (We do not use atomics.)
static double sum_left_to_right(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return s;
}

// Indices of the k atoms with the most exposed points, ties broken by lower
// index so the ranking is deterministic. Used only for the human-readable report.
static std::vector<int> top_exposed(const std::vector<int>& exposed, int k) {
    std::vector<int> idx(exposed.size());
    std::iota(idx.begin(), idx.end(), 0);                 // 0,1,2,...
    const int kk = std::min<int>(k, static_cast<int>(idx.size()));
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
        [&](int a, int b) {
            if (exposed[a] != exposed[b]) return exposed[a] > exposed[b];  // more exposed first
            return a < b;                                                  // tie -> lower index
        });
    idx.resize(kk);
    return idx;
}

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/molecule_sample.xyz";
    Molecule mol;
    try {
        mol = load_molecule(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<int>    exp_cpu;
    std::vector<double> sasa_cpu_v;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    sasa_cpu(mol, exp_cpu, sasa_cpu_v);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU compute (kernel timed inside the wrapper) -----------------
    std::vector<int>    exp_gpu;
    std::vector<double> sasa_gpu_v;
    float gpu_kernel_ms = 0.0f;
    sasa_gpu(mol, exp_gpu, sasa_gpu_v, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    // (a) integer exposed-point counts must match exactly.
    int count_mismatches = 0;
    for (int i = 0; i < mol.n; ++i)
        if (exp_cpu[static_cast<std::size_t>(i)] != exp_gpu[static_cast<std::size_t>(i)])
            ++count_mismatches;
    // (b) per-atom areas must agree within AREA_TOL.
    double max_area_err = 0.0;
    for (int i = 0; i < mol.n; ++i) {
        const double d = std::fabs(sasa_cpu_v[static_cast<std::size_t>(i)] -
                                   sasa_gpu_v[static_cast<std::size_t>(i)]);
        if (d > max_area_err) max_area_err = d;
    }
    const bool pass = (count_mismatches == 0) && (max_area_err <= AREA_TOL);

    // Totals (same deterministic summation order on both sides).
    const double total_cpu = sum_left_to_right(sasa_cpu_v);
    const double total_gpu = sum_left_to_right(sasa_gpu_v);

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    const int K = 5;
    const std::vector<int> top = top_exposed(exp_gpu, K);
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("Shrake-Rupley SASA: %d atoms, probe = %.2f A, %d test points/atom\n",
                mol.n, PROBE_RADIUS, N_SPHERE_POINTS);
    std::printf("total SASA = %.4f A^2\n", total_gpu);
    std::printf("top-%d most exposed atoms (by accessible test points):\n",
                static_cast<int>(top.size()));
    for (std::size_t r = 0; r < top.size(); ++r) {
        const int a = top[r];
        std::printf("  #%zu  atom[%d]  exposed %d/%d  SASA = %.4f A^2\n",
                    r + 1, a, exp_gpu[static_cast<std::size_t>(a)], N_SPHERE_POINTS,
                    sasa_gpu_v[static_cast<std::size_t>(a)]);
    }
    std::printf("RESULT: %s (GPU exposed-point counts match CPU exactly)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (n=%d atoms)\n", path.c_str(), mol.n);
    std::fprintf(stderr, "[verify] exposed-count mismatches: %d   max area err: %.3e A^2 (tol %.1e)\n",
                 count_mismatches, max_area_err, AREA_TOL);
    std::fprintf(stderr, "[verify] total SASA  CPU = %.6f A^2   GPU = %.6f A^2\n",
                 total_cpu, total_gpu);
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny molecule is dominated by "
                         "launch/copy overhead; the GPU's O(n^2) edge grows with atom count.\n");

    // Exit code feeds the demo's pass/fail gate.
    return pass ? 0 : 1;
}
