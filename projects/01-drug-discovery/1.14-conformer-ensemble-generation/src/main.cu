// ===========================================================================
// src/main.cu  --  Entry point: enumerate conformers, verify, prune, report
// ---------------------------------------------------------------------------
// Project 1.14 : Conformer Ensemble Generation
//
// The 5-step shape every project in this repo follows:
//   1. Load the run parameters (rmsd threshold + how many reps to print) from
//      data/sample, or fall back to built-in defaults.
//   2. CPU reference  (reference_cpu.cpp)  -> trusted per-conformer energies.
//   3. GPU compute    (kernels.cu)         -> the thing being taught: all
//      conformer energies in parallel, one thread per conformer.
//   4. VERIFY: GPU energies agree with CPU energies within tolerance.
//   5. PRUNE + REPORT: greedy RMSD clustering of the ensemble; print the global
//      minimum and the distinct representatives. Deterministic result -> stdout;
//      run-to-run timings -> stderr.
//
// STDOUT is byte-for-byte deterministic (diffed by demo/run_demo against
// demo/expected_output.txt); run-to-run timings go to STDERR.
//
// Code tour: start here, then conformer.h (the shared physics), then
// kernels.cuh -> kernels.cu, then reference_cpu.*.
// ===========================================================================
#include <cmath>      // std::fabs
#include <cstdio>
#include <string>
#include <vector>

#include "conformer.h"        // N_ATOMS, N_TORSION, N_CONFORMER, decode_torsions
#include "kernels.cuh"        // energies_gpu (GPU path)
#include "reference_cpu.h"    // enumerate_energies_cpu, rmsd_cluster (CPU path)
#include "util/io.hpp"        // util::CpuTimer, util::read_floats

static const char* PROJECT_ID   = "1.14";
static const char* PROJECT_NAME = "Conformer Ensemble Generation";

// Verification tolerance. CPU and GPU call the SAME conformer_energy() in double
// precision, but the embedding uses cos/sin/sqrt and the clash term uses several
// multiplies, so the GPU's fused-multiply-add and the host libm can differ in the
// last bit or two -- a real, expected ~1e-13 drift (PATTERNS.md §4). We verify to
// a physically negligible 1e-9 kcal/mol and say so honestly; we do NOT pretend the
// two are bit-identical.
static constexpr double TOLERANCE = 1.0e-9;

// Built-in run parameters used when no sample file is supplied.
static constexpr double DEFAULT_RMSD = 1.00;   // Angstrom: dedup threshold
static constexpr int    DEFAULT_TOPN = 5;      // representatives to print

// Parse the tiny sample file: two whitespace-separated numbers
//   <rmsd_threshold_angstrom>  <num_representatives_to_print>
// Returns false (caller uses defaults) if the file is missing or malformed.
static bool load_params(const std::string& path, double& rmsd, int& topn) {
    std::vector<float> v;
    try {
        v = util::read_floats(path);
    } catch (const std::exception&) {
        return false;   // file not found -> caller falls back to defaults
    }
    if (v.size() < 2) return false;
    rmsd = static_cast<double>(v[0]);
    topn = static_cast<int>(v[1]);
    if (rmsd <= 0.0 || topn <= 0) return false;
    return true;
}

// Return the index of the minimum-energy conformer (ties -> lower index), so the
// reported global minimum is deterministic.
static long argmin(const std::vector<double>& e) {
    long best = 0;
    for (long i = 1; i < static_cast<long>(e.size()); ++i) {
        if (e[static_cast<std::size_t>(i)] < e[static_cast<std::size_t>(best)]) best = i;
    }
    return best;
}

int main(int argc, char** argv) {
    // ---- 1. Load run parameters --------------------------------------------
    double rmsd_threshold = DEFAULT_RMSD;
    int    top_print      = DEFAULT_TOPN;
    const char* source = "defaults (built-in)";
    if (argc > 1 && load_params(argv[1], rmsd_threshold, top_print)) {
        source = argv[1];
    }

    // ---- 2. CPU reference: every conformer's energy (timed) ----------------
    std::vector<double> e_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    enumerate_energies_cpu(e_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU compute: same energies, one thread per conformer -----------
    std::vector<double> e_gpu;
    float gpu_kernel_ms = 0.0f;
    energies_gpu(e_gpu, &gpu_kernel_ms);

    // ---- 4. Verify GPU vs CPU energies -------------------------------------
    double max_err = 0.0;
    for (std::size_t i = 0; i < e_cpu.size(); ++i) {
        const double d = std::fabs(e_cpu[i] - e_gpu[i]);
        if (d > max_err) max_err = d;
    }
    const bool pass = (e_cpu.size() == e_gpu.size()) && (max_err <= TOLERANCE);

    // ---- 5. Prune the ensemble by RMSD clustering --------------------------
    // We cluster on the CPU energies (CPU==GPU within tolerance, so it does not
    // matter which we use; the CPU array is the trusted reference).
    const std::vector<long> reps = rmsd_cluster(e_cpu, rmsd_threshold);

    // The global energy minimum is the most important single conformer.
    const long  gmin = argmin(e_cpu);
    double phi_min[N_TORSION];
    decode_torsions(gmin, phi_min);   // its torsion angles, for the report

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("molecule: chain of %d atoms, %d rotatable torsions, %d rotamers each\n",
                N_ATOMS, N_TORSION, N_ROTAMER);
    std::printf("enumerated %ld conformers; pruned to %d distinct (RMSD >= %.2f A)\n",
                N_CONFORMER, static_cast<int>(reps.size()), rmsd_threshold);

    // Global minimum: index, energy, and its torsion angles in degrees.
    std::printf("global minimum: conformer #%ld  E = %.6f kcal/mol\n", gmin,
                e_cpu[static_cast<std::size_t>(gmin)]);
    std::printf("  torsions (deg):");
    for (int t = 0; t < N_TORSION; ++t) {
        // radians -> degrees, rounded to the nearest integer (all our rotamers are
        // exact 60/180-degree multiples, so this prints cleanly and deterministically).
        const double deg = phi_min[t] * 180.0 / 3.14159265358979323846;
        std::printf(" %+d", static_cast<int>(deg < 0 ? deg - 0.5 : deg + 0.5));
    }
    std::printf("\n");

    // The pruned ensemble: the lowest-energy representatives a docking run would use.
    const int show = static_cast<int>(reps.size()) < top_print
                         ? static_cast<int>(reps.size()) : top_print;
    std::printf("ensemble (top %d representatives by energy):\n", show);
    for (int k = 0; k < show; ++k) {
        const long idx = reps[static_cast<std::size_t>(k)];
        std::printf("  #%d  conformer %ld  E = %.6f kcal/mol\n",
                    k + 1, idx, e_cpu[static_cast<std::size_t>(idx)]);
    }
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-09)\n", pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (rmsd_threshold=%.2f A, top_print=%d)\n",
                 source, rmsd_threshold, top_print);
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny ensemble (%ld conformers) "
                         "is dominated by launch/copy overhead; the GPU wins when generating\n"
                         "         conformers for a whole library of molecules at once.\n",
                 N_CONFORMER);
    std::fprintf(stderr, "[verify] max_abs_err = %.3e kcal/mol  (tolerance %.1e)\n",
                 max_err, TOLERANCE);

    // Exit code feeds the demo's pass/fail gate.
    return pass ? 0 : 1;
}
