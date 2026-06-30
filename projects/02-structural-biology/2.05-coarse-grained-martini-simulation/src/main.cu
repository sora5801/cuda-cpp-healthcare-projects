// ===========================================================================
// src/main.cu  --  Entry point: run CG-MD on CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 2.5 : Coarse-Grained / MARTINI Simulation
//
// 5-step shape (the shape EVERY project in this repo follows):
//   1. Load the CG system (data/sample, or a built-in synthetic fallback).
//   2. CPU reference MD simulation (reference_cpu.cpp)        -> trusted answer.
//   3. GPU MD simulation (kernels.cu) -- identical per-bead physics (martini.h).
//   4. VERIFY: the final positions/velocities agree within a documented tol.
//   5. REPORT: deterministic energy / demixing / sampled beads to stdout;
//      timing to stderr.
//
//   STDOUT is kept byte-for-byte deterministic so demo/run_demo can diff it
//   against demo/expected_output.txt. Run-varying numbers (timings) go to
//   STDERR, which the demo shows but does not diff.
//
// Code tour: start here, then martini.h (the physics), kernels.cuh -> kernels.cu,
// and reference_cpu.cpp for the baseline. See ../THEORY.md for the "why".
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // simulate_gpu, System, MdParams, Vec3
#include "reference_cpu.h"    // load_system, simulate_cpu, total_energy, cp_separation
#include "util/io.hpp"        // util::CpuTimer

// These two tokens identify the program in its banner; they MUST stay in sync
// with demo/expected_output.txt (the demo diffs the first line).
static const char* PROJECT_ID   = "2.5";
static const char* PROJECT_NAME = "Coarse-Grained / MARTINI Simulation";

// Verification tolerance on final bead coordinates (positions are O(1) nm in a
// box of edge ~6 nm). The CPU and GPU run the SAME pair-sum in the SAME order,
// so they agree to near machine precision per step; over a few hundred steps the
// only drift is the GPU's fused-multiply-add (FMA) contraction differing from the
// host compiler's, at the ~1e-9 level. We verify to a physically-negligible 1e-6
// and say so -- we do NOT pretend the trajectories are bit-identical. (THEORY 6.)
static constexpr double TOLERANCE = 1.0e-6;

// ---------------------------------------------------------------------------
// make_synthetic: a tiny built-in system used when no data file is supplied.
//   8 beads on a simple lattice in a 6 nm box: 4 apolar "C" beads clustered on
//   one side and 4 polar "P" beads on the other, all at rest. The interaction
//   matrix favours like-like contacts, so the C and P groups stay demixed and
//   relax into compact clusters -- a miniature of MARTINI oil/water behaviour.
//   These EXACT values are what demo/expected_output.txt encodes when run with
//   no argument; the committed sample file reproduces the same setup at n=64.
// ---------------------------------------------------------------------------
static System make_synthetic() {
    System sys;
    MdParams& P = sys.P;
    P.n     = 8;
    P.box   = 6.0;
    P.dt    = 0.005;
    P.steps = 200;
    P.rcut  = 2.5;
    P.mass  = 1.0;
    P.sigma = 0.47;                 // MARTINI-like bead diameter (nm)
    P.eps[0 * MD_NTYPES + 0] = 4.0; // C-C strong
    P.eps[0 * MD_NTYPES + 1] = 1.0; // C-P weak
    P.eps[1 * MD_NTYPES + 0] = 1.0; // P-C
    P.eps[1 * MD_NTYPES + 1] = 4.0; // P-P strong
    // Two 2x2 sheets of beads, C near x=2, P near x=4, all velocities zero.
    const double zc = P.box * 0.5;
    const double coords[8][4] = {
        {2.0, 2.6, zc, 0}, {2.6, 2.6, zc, 0}, {2.0, 3.4, zc, 0}, {2.6, 3.4, zc, 0},
        {4.0, 2.6, zc, 1}, {4.6, 2.6, zc, 1}, {4.0, 3.4, zc, 1}, {4.6, 3.4, zc, 1},
    };
    sys.pos.resize(8); sys.vel.resize(8); sys.type.resize(8);
    for (int i = 0; i < 8; ++i) {
        sys.pos[i]  = Vec3{coords[i][0], coords[i][1], coords[i][2]};
        sys.vel[i]  = Vec3{0.0, 0.0, 0.0};
        sys.type[i] = static_cast<int>(coords[i][3]);
    }
    return sys;
}

int main(int argc, char** argv) {
    // ---- 1. Load the problem (file arg, else built-in synthetic) ----------
    System sys0;                       // initial state (shared by CPU and GPU)
    const char* source;
    if (argc > 1) {
        try {
            sys0 = load_system(argv[1]);
            source = argv[1];
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[error] %s\n", e.what());
            return 2;
        }
    } else {
        sys0 = make_synthetic();
        source = "synthetic (built-in, n=8)";
    }

    // ---- 2. CPU reference (timed) -----------------------------------------
    System cpu = sys0;                 // copy: simulate in place
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    simulate_cpu(cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU simulation (loop timed inside the wrapper) ----------------
    System gpu = sys0;                 // same initial state -> comparable result
    float gpu_kernel_ms = 0.0f;
    simulate_gpu(gpu, &gpu_kernel_ms);

    // ---- 4. Verify: final coordinates agree (CPU vs GPU) ------------------
    double worst = 0.0;
    for (int i = 0; i < sys0.P.n; ++i) {
        worst = std::fmax(worst, std::fabs(cpu.pos[i].x - gpu.pos[i].x));
        worst = std::fmax(worst, std::fabs(cpu.pos[i].y - gpu.pos[i].y));
        worst = std::fmax(worst, std::fabs(cpu.pos[i].z - gpu.pos[i].z));
    }
    const bool pass = worst <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) ----------
    // Energy and demixing are reported from the GPU state (it matches the CPU
    // within tolerance). We print to 4 decimals: robust to the ~1e-9 FMA drift,
    // so stdout is byte-identical every run.
    const double e0   = total_energy(sys0);   // energy of the initial state
    const double eG   = total_energy(gpu);     // energy after the run
    const double sep0 = cp_separation(sys0);   // C/P centroid distance, initial
    const double sepG = cp_separation(gpu);    // ... and final (demixing signal)

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("CG-MD: %d beads, box=%.2f nm, dt=%.4f, %d steps, rcut=%.2f nm, sigma=%.3f nm\n",
                sys0.P.n, sys0.P.box, sys0.P.dt, sys0.P.steps, sys0.P.rcut, sys0.P.sigma);
    std::printf("eps matrix: CC=%.2f CP=%.2f PP=%.2f  (like-likes-like -> demixing)\n",
                sys0.P.eps[0], sys0.P.eps[1], sys0.P.eps[3]);
    std::printf("total energy: initial = %.4f   final = %.4f\n", e0, eG);
    std::printf("C/P centroid separation: initial = %.4f nm   final = %.4f nm\n", sep0, sepG);
    // A few sampled final bead positions (GPU), for a concrete fingerprint.
    const int show = sys0.P.n < 4 ? sys0.P.n : 4;
    for (int i = 0; i < show; ++i)
        std::printf("bead %d (type %s) final pos = (%.4f, %.4f, %.4f)\n",
                    i, gpu.type[i] == 0 ? "C" : "P",
                    gpu.pos[i].x, gpu.pos[i].y, gpu.pos[i].z);
    std::printf("RESULT: %s (GPU trajectory matches CPU within tol=1.0e-06)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) -----------------
    std::fprintf(stderr, "[data]   source: %s  (%d beads)\n", source, sys0.P.n);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- the all-pairs force is O(N^2); "
                         "the GPU's edge over the CPU grows steeply with bead count.\n");
    std::fprintf(stderr, "[verify] worst |pos_cpu - pos_gpu| = %.3e  (tolerance %.1e)\n",
                 worst, TOLERANCE);

    // Exit code feeds the demo's pass/fail gate.
    return pass ? 0 : 1;
}
