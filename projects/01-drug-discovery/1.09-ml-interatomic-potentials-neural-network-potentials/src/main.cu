// ===========================================================================
// src/main.cu  --  Entry point: load structure, run CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 1.9 : ML Interatomic Potentials (Neural Network Potentials)
//
// THE 5-STEP SHAPE every project in this repo follows:
//   1. Load the problem (a molecular structure from data/sample, or via argv[1]).
//   2. CPU reference  (reference_cpu.cpp)  -> trusted per-atom + total energy.
//   3. GPU compute    (kernels.cu)         -> the thing being taught.
//   4. VERIFY: GPU agrees with CPU within a documented tolerance.
//   5. REPORT: deterministic energies to stdout; timing + error to stderr.
//
//   STDOUT is byte-for-byte deterministic (demo/run_demo diffs it against
//   demo/expected_output.txt); run-to-run timings go to STDERR (shown, not
//   diffed). We print energies at fixed precision so the text never wobbles.
//
// Code tour: start here, then nnp.h (the shared physics), then kernels.cuh ->
//   kernels.cu (the GPU path), then reference_cpu.* (the baseline).
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // nnp_energy_gpu, Structure, AcsfParams, AtomicNet
#include "reference_cpu.h"    // load_structure, build_*, nnp_energy_cpu
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "1.9";
static const char* PROJECT_NAME = "ML Interatomic Potentials (Neural Network Potentials)";

// Verification tolerance (absolute, on energy).
//   The CPU and GPU call the IDENTICAL atomic_energy() from nnp.h in double
//   precision, so the only possible difference is the GPU's fused-multiply-add
//   (FMA) contracting "a*b+c" where the host does two rounded steps. Over a short
//   descriptor + a 2-hidden-layer MLP that divergence is ~1e-13 at most, so we
//   verify to 1e-9 -- comfortably tight yet honest about FMA (PATTERNS.md sec 4).
static constexpr double TOLERANCE = 1.0e-9;

// How many per-atom energies to print (kept small so the demo output is compact;
// the FULL structure is still summed for the total).
static constexpr int SHOW_ATOMS = 6;

// max |a[i] - b[i]| over two equal-length double arrays (our headline metric).
static double max_abs_err(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return 1e300;   // shape mismatch -> never "agree"
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        const double ad = d < 0 ? -d : d;
        if (ad > worst) worst = ad;
    }
    return worst;
}

int main(int argc, char** argv) {
    // ---- 1. Load the structure --------------------------------------------
    const std::string path = (argc > 1) ? argv[1]
                                        : "data/sample/water_cluster.xyzc";
    Structure s;
    try {
        s = load_structure(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // Build the (deterministic) model: descriptor hyperparameters + MLP weights.
    // Both CPU and GPU use these exact same objects -> identical math.
    const AcsfParams params = build_acsf_params();
    const AtomicNet  net    = build_atomic_net();

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<double> e_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    const double E_cpu = nnp_energy_cpu(s, params, net, e_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU compute (kernel timed inside the wrapper) -----------------
    std::vector<double> e_gpu;
    float gpu_kernel_ms = 0.0f;
    const double E_gpu = nnp_energy_gpu(s, params, net, e_gpu, &gpu_kernel_ms);

    // ---- 4. Verify ---------------------------------------------------------
    const double atom_err  = max_abs_err(e_cpu, e_gpu);          // worst per-atom diff
    const double total_err = (E_cpu > E_gpu ? E_cpu - E_gpu : E_gpu - E_cpu);
    const bool pass = (atom_err <= TOLERANCE) && (total_err <= TOLERANCE);

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) ----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("Behler-Parrinello NNP: %d atoms, %d radial descriptors, "
                "MLP %d->%d->%d->1\n", s.n, N_DESC, N_DESC, N_HID, N_HID);
    const int show = s.n < SHOW_ATOMS ? s.n : SHOW_ATOMS;
    std::printf("per-atom energy E_i (GPU), first %d atom(s):\n", show);
    for (int i = 0; i < show; ++i)
        std::printf("  atom %2d  E = %+.6f\n", i, e_gpu[static_cast<std::size_t>(i)]);
    std::printf("total energy E = %+.6f\n", E_gpu);
    std::printf("RESULT: %s (GPU matches CPU within tol=1.0e-09)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) -----------------
    std::fprintf(stderr, "[data]   source: %s  (n=%d atoms, Rc=%.1f A)\n",
                 path.c_str(), s.n, params.Rc);
    std::fprintf(stderr, "[timing] CPU reference: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny cluster is "
                         "dominated by launch/copy overhead; the GPU wins at protein "
                         "scale (thousands of atoms x many MD steps).\n");
    std::fprintf(stderr, "[verify] max per-atom err = %.3e   total err = %.3e   "
                         "(tolerance %.1e)\n", atom_err, total_err, TOLERANCE);

    // Exit code feeds the demo's pass/fail gate.
    return pass ? 0 : 1;
}
