// ===========================================================================
// src/main.cu  --  Entry point: solve lobule perfusion, verify, report
// ---------------------------------------------------------------------------
// Project 6.25 : Liver & Kidney Perfusion Modeling
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the lobule config (fixed sinusoid physics + a velocity sweep).
//   2. CPU reference: integrate every sinusoid serially (reference_cpu.cpp).
//   3. GPU: one thread per sinusoid, full RK4 spatial march (kernels.cu).
//   4. VERIFY (two ways):
//        (a) per-sinusoid GPU vs CPU results agree to round-off (same RK4), AND
//        (b) the mean extraction ratio matches the ANALYTIC first-order limit --
//            a check on the SCIENCE, not just CPU==GPU agreement (PATTERNS.md 4).
//   5. REPORT: deterministic per-zone/sinusoid table + lobule summary to stdout;
//      timings and run-varying detail to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Timings go to STDERR (shown, not diffed).
//
//   >>> Educational, NOT for clinical use. All data are SYNTHETIC. <<<
//
// Code tour: start here, then perfusion.h (the ODE + RK4), reference_cpu.h/.cpp,
//   kernels.cuh -> kernels.cu. See ../THEORY.md for the "why".
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // integrate_gpu, LobuleConfig, SinusoidResult
#include "reference_cpu.h"    // load_lobule, integrate_cpu, sinusoid_velocity
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "6.25";
static const char* PROJECT_NAME = "Liver & Kidney Perfusion Modeling";

// (a) GPU-vs-CPU tolerance. Both sides run the SAME double-precision RK4 from
//     perfusion.h, so agreement is to floating-point round-off. 1e-9 is generous
//     headroom over the ~1e-13 differences FMA ordering can introduce.
static constexpr double TOLERANCE = 1.0e-9;

// (b) Analytic-limit tolerance. The closed-form profile below assumes the
//     UNSATURATED (first-order) Michaelis-Menten regime C << Km; the sample is
//     engineered so C_in << Km, but the small remaining nonlinearity means we
//     only expect agreement to a physical ~1%% (0.01). Documented, not pretended.
static constexpr double ANALYTIC_TOL = 1.0e-2;

// ---------------------------------------------------------------------------
// analytic_extraction: the first-order (C << Km) closed form for one sinusoid.
//   In that limit R = (Vmax(x)/Km)*C, so v*dC/dx = -(Vmax(x)/Km)*C, an ODE with
//   a spatially-varying rate. Integrating from 0..L:
//       C_out = C_in * exp( -(1/v) * integral_0^L (Vmax(x)/Km) dx )
//   With the LINEAR zonation Vmax(x) = Vmax_pp + (Vmax_cl-Vmax_pp)*(x/L), the
//   integral of Vmax over [0,L] is L * (Vmax_pp+Vmax_cl)/2 (the average Vmax).
//   Hence the extraction ratio E = 1 - exp( -(Vmax_avg/Km) * L / v ).
//   This is a hand-derivable sanity check on the numerics (THEORY section 6).
// ---------------------------------------------------------------------------
static double analytic_extraction(const LobuleConfig& c, double v) {
    const double vmax_avg = 0.5 * (c.p.Vmax_pp + c.p.Vmax_cl);   // mean over the zonation ramp
    const double k = vmax_avg / c.p.Km;                          // first-order rate constant (1/s)
    const double C_out = c.p.C_in * std::exp(-(k) * c.p.L / v);  // exponential washout
    return (c.p.C_in > 0.0) ? (c.p.C_in - C_out) / c.p.C_in : 0.0;
}

int main(int argc, char** argv) {
    // ---- 1. Load -----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/lobule.txt";
    LobuleConfig c;
    try {
        c = load_lobule(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    const int M = lobule_size(c);

    // ---- 2. CPU reference (timed) -----------------------------------------
    std::vector<SinusoidResult> res_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    integrate_cpu(c, res_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU ensemble (kernel timed) -----------------------------------
    std::vector<SinusoidResult> res_gpu;
    float gpu_kernel_ms = 0.0f;
    integrate_gpu(c, res_gpu, &gpu_kernel_ms);

    // ---- 4a. Verify GPU vs CPU (per sinusoid) -----------------------------
    double worst = 0.0;
    for (int i = 0; i < M; ++i) {
        worst = std::fmax(worst, std::fabs(res_cpu[i].C_out          - res_gpu[i].C_out));
        worst = std::fmax(worst, std::fabs(res_cpu[i].extraction_ratio - res_gpu[i].extraction_ratio));
    }
    const bool pass_gpu = worst <= TOLERANCE;

    // ---- 4b. Verify against the analytic first-order limit ----------------
    // Compare the mean numerical extraction ratio to the mean analytic one.
    double sum_num = 0.0, sum_ana = 0.0, worst_ana = 0.0;
    for (int i = 0; i < M; ++i) {
        const double v = sinusoid_velocity(c, i);
        const double ana = analytic_extraction(c, v);
        sum_num += res_gpu[i].extraction_ratio;
        sum_ana += ana;
        worst_ana = std::fmax(worst_ana, std::fabs(res_gpu[i].extraction_ratio - ana));
    }
    const double mean_num = sum_num / M;
    const double mean_ana = sum_ana / M;
    const bool pass_ana = std::fabs(mean_num - mean_ana) <= ANALYTIC_TOL;

    const bool pass = pass_gpu && pass_ana;

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    // Lobule-level summary: whole-lobule mean extraction (the hepatic clearance
    // proxy) and the min/max across the perfusion (velocity) spread.
    double emin = 1e300, emax = -1e300;
    for (int i = 0; i < M; ++i) {
        emin = std::fmin(emin, res_gpu[i].extraction_ratio);
        emax = std::fmax(emax, res_gpu[i].extraction_ratio);
    }

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("SYNTHETIC liver lobule: %d parallel sinusoids, L=%.3f mm, C_in=%.3f uM, Km=%.3f uM\n",
                M, c.p.L, c.p.C_in, c.p.Km);
    std::printf("zonation Vmax: periportal=%.3f -> centrilobular=%.3f uM/s; velocity sweep %.4f..%.4f mm/s over %d RK4 steps\n",
                c.p.Vmax_pp, c.p.Vmax_cl, c.v_lo, c.v_hi, c.p.nseg);
    std::printf("sample sinusoids (v[mm/s] -> C_out[uM] extraction[%%]):\n");
    const int picks[5] = {0, M / 4, M / 2, (3 * M) / 4, M - 1};
    for (int s = 0; s < 5; ++s) {
        const int i = picks[s];
        const double v = sinusoid_velocity(c, i);
        std::printf("  s%-5d: %8.4f -> %8.4f %7.3f\n",
                    i, v, res_gpu[i].C_out, 100.0 * res_gpu[i].extraction_ratio);
    }
    std::printf("lobule extraction ratio: mean=%.4f  min=%.4f  max=%.4f\n", mean_num, emin, emax);
    std::printf("analytic first-order limit: mean extraction=%.4f\n", mean_ana);
    std::printf("RESULT: %s (GPU==CPU within %.0e; mean extraction within %.0e of analytic)\n",
                pass ? "PASS" : "FAIL", TOLERANCE, ANALYTIC_TOL);

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d sinusoids)\n", path.c_str(), M);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernel: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- tiny lobules are launch-bound; the GPU edge "
                         "grows toward the millions of segments a real organ model needs.\n");
    std::fprintf(stderr, "[verify] worst per-sinusoid |GPU-CPU| = %.3e (tol %.1e); worst |num-analytic| = %.3e\n",
                 worst, TOLERANCE, worst_ana);

    return pass ? 0 : 1;
}
