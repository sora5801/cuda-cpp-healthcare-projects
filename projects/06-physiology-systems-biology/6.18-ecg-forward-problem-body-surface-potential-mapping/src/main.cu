// ===========================================================================
// src/main.cu  --  Entry point: load data, run CPU + GPU, verify, report
// ---------------------------------------------------------------------------
// Project 6.18 : ECG Forward Problem & Body-Surface Potential Mapping
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the torso/heart model + source time series (data/sample).
//   2. CPU reference (reference_cpu.cpp): build lead field A, apply Phi = A*X.
//   3. GPU result  (kernels.cu): build A with a kernel, apply with cuBLAS DGEMM.
//   4. VERIFY: the GPU lead field and GPU potentials match the CPU within a
//      documented tolerance (the correctness guarantee).
//   5. REPORT: a DETERMINISTIC summary to stdout (per-lead peak-to-peak swings,
//      the recovered "peak lead", a signature body-surface potential value);
//      timing to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Anything that varies run-to-run (timings) goes to
//   STDERR, which the demo shows but does not diff (PATTERNS.md §3).
//
// THE SCIENCE IN ONE BREATH
//   The ECG "forward problem" predicts the potentials an EKG electrode would
//   record given the heart's electrical sources. We model the heart as a handful
//   of current DIPOLES with time-varying strengths and the body as a homogeneous
//   volume conductor; the quasi-static Poisson equation then gives a LINEAR map
//   -- the lead-field / transfer matrix A -- from source strengths to surface
//   potentials. Building A is a one-time cost; applying it (Phi = A*X) is the
//   per-time-step work the GPU accelerates with a single dense DGEMM.
//
// READ THIS FIRST in the code tour, then kernels.cuh -> kernels.cu, then
// reference_cpu.cpp for the baseline, and ecg_core.h for the shared physics.
// See ../THEORY.md for the full "why".
// ===========================================================================
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // gpu_build_lead_field, gpu_apply_forward
#include "reference_cpu.h"    // ECGData, load_ecg, *_reference
#include "ecg_core.h"         // ecg::Vec3, TORSO_SIGMA (for the report)
#include "util/io.hpp"        // util::CpuTimer

// Identify the program. Must stay in sync with demo/expected_output.txt.
static const char* PROJECT_ID   = "6.18";
static const char* PROJECT_NAME = "ECG Forward Problem & Body-Surface Potential Mapping";

// ---- Verification tolerances (documented; PATTERNS.md §4) ------------------
// (a) LEAD_TOL: the lead field A is built by the SAME ecg::dipole_potential on
//     both sides in the same order, so A_gpu and A_cpu agree essentially to
//     machine epsilon; we still allow a tiny double-eps slack.
// (b) PHI_TOL: Phi = A*X. The GPU's DGEMM sums the length-S dot products in a
//     different ORDER than the CPU triple loop, and uses fused multiply-add.
//     Floating-point addition is not associative, so Phi agrees only to ~1e-12
//     relative -- a real, teachable effect. Our potentials are O(1e-2..1e0) volt
//     units, so an absolute 1e-9 tolerance is far below any physical signal.
static constexpr double LEAD_TOL = 1.0e-12;   // entrywise |A_gpu - A_cpu|
static constexpr double PHI_TOL  = 1.0e-9;    // entrywise |Phi_gpu - Phi_cpu|

// peak_to_peak: max-min of electrode e's potential row over all T frames.
//   This is the clinically-meaningful "how big is this lead's deflection"
//   summary of one body-surface potential trace.
static double peak_to_peak(const std::vector<double>& Phi, int e, int T) {
    const double* row = &Phi[static_cast<std::size_t>(e) * T];
    double lo = row[0], hi = row[0];
    for (int t = 1; t < T; ++t) {
        if (row[t] < lo) lo = row[t];
        if (row[t] > hi) hi = row[t];
    }
    return hi - lo;
}

int main(int argc, char** argv) {
    // ---- 1. Load the model + source time series ----------------------------
    const std::string path =
        (argc > 1) ? argv[1] : "data/sample/ecg_sample.txt";
    ECGData d;
    try {
        d = load_ecg(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int L = d.L, S = d.S, T = d.T;

    // ---- 2. CPU reference (timed) ------------------------------------------
    //   build lead field A, then apply Phi = A*X -- all serial and obvious.
    std::vector<double> A_cpu, Phi_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    build_lead_field_reference(d, A_cpu);
    apply_forward_reference(A_cpu, d.source_strength, L, S, T, Phi_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU result (kernels timed inside the wrappers) -----------------
    std::vector<double> A_gpu, Phi_gpu;
    float build_ms = 0.0f, gemm_ms = 0.0f;
    gpu_build_lead_field(d.electrode, d.src_pos, d.src_dir, L, S, A_gpu, &build_ms);
    // Apply the GPU-built lead field to the source series with cuBLAS DGEMM.
    gpu_apply_forward(A_gpu, d.source_strength, L, S, T, Phi_gpu, &gemm_ms);

    // ---- 4. Verify GPU vs CPU ----------------------------------------------
    // (a) lead field: worst entrywise difference over the L x S matrix.
    double lead_worst = 0.0;
    for (std::size_t i = 0; i < A_cpu.size(); ++i)
        lead_worst = std::fmax(lead_worst, std::fabs(A_cpu[i] - A_gpu[i]));
    // (b) potentials: worst entrywise difference over the L x T matrix.
    double phi_worst = 0.0;
    for (std::size_t i = 0; i < Phi_cpu.size(); ++i)
        phi_worst = std::fmax(phi_worst, std::fabs(Phi_cpu[i] - Phi_gpu[i]));
    const bool pass = (lead_worst <= LEAD_TOL) && (phi_worst <= PHI_TOL);

    // ---- 5a. Deterministic report -> STDOUT --------------------------------
    // Per-lead peak-to-peak swing from the GPU potentials (verified == CPU).
    std::vector<double> ptp(L);
    for (int e = 0; e < L; ++e) ptp[e] = peak_to_peak(Phi_gpu, e, T);

    // Which lead actually swings most? (should recover d.expected_peak_lead)
    int peak_lead = 0;
    for (int e = 1; e < L; ++e) if (ptp[e] > ptp[peak_lead]) peak_lead = e;
    const bool recovered = (peak_lead == d.expected_peak_lead);

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("torso model: L=%d electrodes, S=%d dipole sources, T=%d frames "
                "(synthetic)\n", L, S, T);
    std::printf("conductivity sigma=%.4f S/m (homogeneous volume conductor)\n",
                ecg::TORSO_SIGMA);
    std::printf("per-lead peak-to-peak body-surface potential (lead: p2p):\n");
    for (int e = 0; e < L; ++e)
        std::printf("  lead %2d: p2p=%10.6f\n", e, ptp[e]);
    std::printf("largest-swing lead: %d  (expected from geometry: %d) -> %s\n",
                peak_lead, d.expected_peak_lead, recovered ? "RECOVERED" : "MISS");
    // A single deterministic "signature" surface potential: electrode 0 at the
    // last frame. Fixing the exact (e,t) keeps the printed digits reproducible.
    std::printf("signature Phi[lead 0][frame %d] = %.6f\n",
                T - 1, Phi_gpu[static_cast<std::size_t>(0) * T + (T - 1)]);
    std::printf("RESULT: %s (GPU lead field and potentials match CPU within tol)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR --------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (L=%d, S=%d, T=%d)\n",
                 path.c_str(), L, S, T);
    std::fprintf(stderr, "[timing] CPU reference (build A + apply A*X): %.3f ms\n",
                 cpu_ms);
    std::fprintf(stderr, "[timing] GPU build lead field: %.3f ms   "
                         "cuBLAS DGEMM (Phi=A*X): %.3f ms\n", build_ms, gemm_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this sample is tiny, so the "
                         "GPU is launch/copy bound; the DGEMM's edge grows as O(L*S*T).\n");
    std::fprintf(stderr, "[verify] lead-field worst entry diff = %.3e  (tol %.1e)\n",
                 lead_worst, LEAD_TOL);
    std::fprintf(stderr, "[verify] potential  worst entry diff = %.3e  (tol %.1e)\n",
                 phi_worst, PHI_TOL);

    return pass ? 0 : 1;
}
