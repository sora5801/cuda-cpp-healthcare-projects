// ===========================================================================
// src/main.cu  --  Entry point: run Turing RD, verify GPU vs CPU, report
// ---------------------------------------------------------------------------
// Project 6.24 : Reaction-Diffusion Morphogenesis (Turing Patterns)
//
// WHAT THIS FILE DOES  (the 5-step shape every project in this repo follows)
//   1. Load parameters (data/sample) and build the deterministically-seeded grid.
//   2. CPU reference simulation (reference_cpu.cpp)  -> trusted answer.
//   3. GPU simulation (kernels.cu)                   -> the thing taught.
//   4. VERIFY: assert the final GPU fields match the CPU within a tolerance.
//   5. REPORT: deterministic pattern metrics to stdout; timing to stderr.
//
//   From a near-uniform, tiny-noise seed the activator/inhibitor fields self-
//   organize into a stationary Turing PATTERN. We summarize that pattern with a
//   handful of deterministic metrics and also print an ANALYTIC linear-stability
//   check (the Turing "dispersion relation") -- a second, science-validating
//   test that the chosen parameters actually lie in the pattern-forming regime.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Timings (which vary run-to-run) go to STDERR.
//
// READ THIS FIRST in the code tour, then turing.h (the physics), kernels.cuh ->
// kernels.cu (the GPU path), and reference_cpu.cpp (the CPU baseline).
// ===========================================================================
#include <cmath>       // std::fabs, std::fmax, std::sqrt, std::cos
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // simulate_gpu, TuringParams
#include "reference_cpu.h"    // load_params, init_fields, simulate_cpu
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "6.24";
static const char* PROJECT_NAME = "Reaction-Diffusion Morphogenesis (Turing Patterns)";

// Verification tolerance. CPU and GPU run the identical tu_update() in double
// precision, but over thousands of nonlinear steps the GPU's fused multiply-add
// (FMA) and the host compiler's separate multiply+add diverge at the ~1e-13
// per-step level, which accumulates. The field values are O(1), so a 1e-6
// tolerance is physically negligible (far below "same pattern") yet strict
// enough to catch a real bug. See THEORY §"How we verify correctness" and
// PATTERNS.md §4 (long iterative solvers).
static constexpr double TOLERANCE = 1.0e-6;

// ---------------------------------------------------------------------------
// turing_growth_rate  --  the analytic dispersion relation lambda_max(k^2).
//
// Linear-stability analysis of the Gierer-Meinhardt system about its uniform
// fixed point gives a 2x2 Jacobian J of the REACTION terms; adding diffusion
// shifts the diagonal by -D*k^2 (mode of wavenumber k). The growth rate of that
// mode is the larger eigenvalue of
//     M(k^2) = [ Ja - Da*k^2      Jb        ]
//              [   Jc          Jd - Dh*k^2  ]
// lambda_max = tr/2 + sqrt((tr/2)^2 - det). A Turing instability exists when
// lambda_max(k^2) > 0 for some k^2 > 0 while the uniform (k=0) state is stable.
//
// We evaluate lambda_max on a grid of k^2 and return the peak growth rate and
// the wavenumber where it occurs -- this predicts the pattern's WAVELENGTH
// (2*pi/k) independently of the simulation, validating the science, not just
// CPU==GPU agreement (PATTERNS.md §4). The Jacobian entries below use the
// linearized Gierer-Meinhardt kinetics; see THEORY §"The math" for the algebra.
//
// Returns lambda_max at the fastest-growing mode; writes k^2 there into *k2_out.
// ---------------------------------------------------------------------------
static double turing_growth_rate(const TuringParams& P, double* k2_out) {
    // Linearize about the homogeneous steady state (a0,h0) = (a*, h*), the SAME
    // fixed point the fields are seeded at (turing.h). This makes the Jacobian
    // self-consistent, so the predicted instability matches the observed pattern.
    const double a0 = tu_baseline_activator(P);
    const double h0 = tu_baseline_inhibitor(P, a0);

    // Partial derivatives of the reaction terms f=rho*a^2/h - mu_a*a + rho_a and
    // g=rho*a^2 - mu_h*h, evaluated at (a0,h0):
    const double Ja = 2.0 * P.rho * a0 / h0 - P.mu_a;      // df/da
    const double Jb = -P.rho * a0 * a0 / (h0 * h0);        // df/dh
    const double Jc = 2.0 * P.rho * a0;                    // dg/da
    const double Jd = -P.mu_h;                             // dg/dh

    // Scan k^2 from 0 up to the largest resolvable value on the grid. The finest
    // mode a periodic grid of spacing 1 can carry has k = pi (Nyquist), so
    // k^2 in [0, pi^2]. 400 samples is plenty to locate the peak for a demo.
    const double kmax2 = 3.14159265358979323846 * 3.14159265358979323846;
    double best_lambda = -1e300, best_k2 = 0.0;
    const int SAMPLES = 400;
    for (int i = 0; i <= SAMPLES; ++i) {
        const double k2 = kmax2 * static_cast<double>(i) / SAMPLES;
        const double m11 = Ja - P.Da * k2;
        const double m22 = Jd - P.Dh * k2;
        const double tr  = m11 + m22;                      // trace of M(k^2)
        const double det = m11 * m22 - Jb * Jc;            // det of M(k^2)
        const double disc = 0.25 * tr * tr - det;          // (tr/2)^2 - det
        // Larger eigenvalue. If the discriminant is negative the pair is complex;
        // the real part is tr/2, so use max(real-part-based) growth = tr/2.
        const double lambda = (disc >= 0.0) ? (0.5 * tr + std::sqrt(disc)) : (0.5 * tr);
        if (lambda > best_lambda) { best_lambda = lambda; best_k2 = k2; }
    }
    *k2_out = best_k2;
    return best_lambda;
}

int main(int argc, char** argv) {
    // ---- 1. Load parameters + build the deterministic seed -----------------
    const std::string path = (argc > 1) ? argv[1]
                                        : "data/sample/turing_params.txt";
    TuringParams P;
    try {
        P = load_params(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    std::vector<double> a0, h0;
    init_fields(P, a0, h0);

    // ---- 2. CPU reference (timed) ------------------------------------------
    std::vector<double> a_cpu = a0, h_cpu = h0;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    simulate_cpu(P, a_cpu, h_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU simulation (loop timed inside the wrapper) -----------------
    std::vector<double> a_gpu = a0, h_gpu = h0;
    float gpu_kernel_ms = 0.0f;
    simulate_gpu(P, a_gpu, h_gpu, &gpu_kernel_ms);

    // ---- 4. Verify: the final fields agree ---------------------------------
    double worst = 0.0;
    for (int i = 0; i < P.nx * P.ny; ++i) {
        worst = std::fmax(worst, std::fabs(a_cpu[i] - a_gpu[i]));
        worst = std::fmax(worst, std::fabs(h_cpu[i] - h_gpu[i]));
    }
    const bool pass = worst <= TOLERANCE;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    // Pattern metrics from the GPU activator field: mean level, peak, spatial
    // contrast (max-min, ~0 means "no pattern"), and the count of "peak" cells
    // (activator above the mean) -- a proxy for the number of spots/stripes.
    double sum_a = 0.0, min_a = 1e300, max_a = -1e300;
    for (int i = 0; i < P.nx * P.ny; ++i) {
        sum_a += a_gpu[i];
        min_a = std::fmin(min_a, a_gpu[i]);
        max_a = std::fmax(max_a, a_gpu[i]);
    }
    const double mean_a = sum_a / (P.nx * P.ny);
    int peaks = 0;
    for (int i = 0; i < P.nx * P.ny; ++i)
        if (a_gpu[i] > mean_a) ++peaks;

    // Analytic Turing check: does linear theory predict a pattern-forming mode?
    double best_k2 = 0.0;
    const double lambda_max = turing_growth_rate(P, &best_k2);
    const double k_star = std::sqrt(best_k2);
    // Predicted pattern wavelength (cells). Guard k*=0 (no unstable mode).
    const double wavelength = (k_star > 1e-9)
        ? (2.0 * 3.14159265358979323846 / k_star) : 0.0;
    const bool turing_regime = (lambda_max > 0.0) && (best_k2 > 1e-9);

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("Gierer-Meinhardt: %dx%d grid, %d steps, Da=%.4f Dh=%.4f "
                "(Dh/Da=%.1f) rho=%.3f mu_a=%.3f mu_h=%.3f\n",
                P.nx, P.ny, P.steps, P.Da, P.Dh, P.Dh / P.Da, P.rho, P.mu_a, P.mu_h);
    std::printf("pattern: mean a=%.6f, min a=%.6f, max a=%.6f, contrast=%.6f, "
                "peak cells (a>mean)=%d of %d\n",
                mean_a, min_a, max_a, max_a - min_a, peaks, P.nx * P.ny);
    std::printf("linear stability: Turing regime=%s, max growth=%.6f at "
                "k*=%.4f (predicted wavelength=%.2f cells)\n",
                turing_regime ? "YES" : "NO", lambda_max, k_star, wavelength);
    std::printf("a along center row (8 samples):");
    const int cy = P.ny / 2;
    for (int s = 0; s < 8; ++s) {
        const int x = (s * (P.nx - 1)) / 7;
        std::printf(" %.4f", a_gpu[cy * P.nx + x]);
    }
    std::printf("\n");
    std::printf("RESULT: %s (GPU field matches CPU within tol=1.0e-06)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%d cells, %d steps)\n",
                 path.c_str(), P.nx * P.ny, P.steps);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- the GPU's edge over "
                         "the CPU grows with grid size and step count.\n");
    std::fprintf(stderr, "[verify] worst field diff = %.3e  (tolerance %.1e)\n",
                 worst, TOLERANCE);

    return pass ? 0 : 1;
}
