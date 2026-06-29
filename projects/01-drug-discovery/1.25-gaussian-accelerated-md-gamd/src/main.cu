// ===========================================================================
// src/main.cu  --  Entry point: run GaMD ensemble (CPU + GPU), verify, report
// ---------------------------------------------------------------------------
// Project 1.25 : Gaussian-Accelerated MD (GaMD)   (reduced-scope teaching version)
//
// WHAT THIS FILE DOES  (the 5-step shape EVERY project in this repo follows)
//   1. Load the GaMD run config from data/sample/ (the model potential, the
//      thermostat, the boost parameters, the ensemble + histogram settings).
//   2. CPU reference: run every walker serially -> trusted fixed-point tally.
//   3. GPU: one thread per walker, atomic fixed-point tally -> the thing taught.
//   4. VERIFY: the GPU tally equals the CPU tally BIT-FOR-BIT (integer atomics
//      are order-independent), so tolerance is EXACTLY 0. Then a second,
//      scientific check: the reweighted PMF recovers the known barrier height.
//   5. REPORT: deterministic PMF + recovery metrics to stdout; timing to stderr.
//
//   STDOUT is byte-for-byte deterministic so demo/run_demo can diff it against
//   demo/expected_output.txt. Timings (which vary run to run) go to STDERR, shown
//   but not diffed (PATTERNS.md §3 rule 1).
//
// Code tour: read this first, then gamd.h (the physics + boost + reweighting),
// then kernels.cuh -> kernels.cu (GPU twin), reference_cpu.cpp (serial baseline).
// See ../THEORY.md for the science and the GPU mapping.
// ===========================================================================
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // run_ensemble_gpu, GamdConfig
#include "reference_cpu.h"    // load_config, run_ensemble_cpu
#include "gamd.h"             // reweight_pmf_bin, analytic_pmf, bin geometry
#include "util/io.hpp"        // util::CpuTimer

// Identify the project. MUST stay in sync with demo/expected_output.txt.
static const char* PROJECT_ID   = "1.25";
static const char* PROJECT_NAME = "Gaussian-Accelerated MD (GaMD)";

// ---------------------------------------------------------------------------
// reconstruct_pmf: turn the fixed-point (count|S1|S2) tally into a free-energy
//   profile F(bin), shifted so its minimum is 0 (a PMF is defined up to a
//   constant). Uses the 2nd-order cumulant reweighting in gamd.h. Bins that were
//   never visited are left at +inf (sentinel 1e30) and skipped by the caller.
// ---------------------------------------------------------------------------
static std::vector<double> reconstruct_pmf(const GamdConfig& c,
                                           const std::vector<int64_t>& acc) {
    // Total tallied samples = sum of all bin counts (the first n_bins slots).
    double total = 0.0;
    for (int b = 0; b < c.n_bins; ++b) total += (double)acc[acc_count_idx(c, b)];

    std::vector<double> pmf(c.n_bins, 1e30);
    double fmin = 1e30;
    for (int b = 0; b < c.n_bins; ++b) {
        const int64_t cnt = acc[acc_count_idx(c, b)];
        const int64_t s1  = acc[acc_s1_idx(c, b)];
        const int64_t s2  = acc[acc_s2_idx(c, b)];
        const double f = reweight_pmf_bin(c, cnt, s1, s2, total);
        pmf[b] = f;
        if (f < fmin) fmin = f;                 // track the global minimum
    }
    // Shift so min(F) == 0 for a clean, comparable plot.
    for (int b = 0; b < c.n_bins; ++b)
        if (pmf[b] < 1e29) pmf[b] -= fmin;
    return pmf;
}

// Bin-center x for bin b (used for the analytic comparison + printing).
static double bin_center(const GamdConfig& c, int b) {
    return c.x_lo + (b + 0.5) * bin_width(c);
}

int main(int argc, char** argv) {
    // ---- 1. Load the GaMD run config ---------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/gamd_config.txt";
    GamdConfig c;
    try {
        c = load_config(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) ------------------------------------------
    std::vector<int64_t> acc_cpu;
    std::vector<double>  finalx_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    run_ensemble_cpu(c, acc_cpu, finalx_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU ensemble (kernel timed inside the wrapper) -----------------
    std::vector<int64_t> acc_gpu;
    std::vector<double>  finalx_gpu;
    float gpu_kernel_ms = 0.0f;
    run_ensemble_gpu(c, acc_gpu, finalx_gpu, &gpu_kernel_ms);

    // ---- 4a. Verify GPU == CPU EXACTLY (integer fixed-point tally) ---------
    // Because both sides accumulate the SAME integers and integer addition is
    // order-independent, the two tallies must be bit-for-bit identical. Any
    // difference is a real bug, so the tolerance is exactly 0 (PATTERNS §4).
    int64_t worst_acc_diff = 0;
    for (std::size_t i = 0; i < acc_cpu.size(); ++i) {
        int64_t d = acc_cpu[i] - acc_gpu[i];
        if (d < 0) d = -d;
        if (d > worst_acc_diff) worst_acc_diff = d;
    }
    const bool tally_exact = (worst_acc_diff == 0);

    // ---- 4b. Reconstruct the PMF and check the SCIENCE ---------------------
    // Reweight the (identical) tally into a free-energy profile, then check that
    // GaMD recovered the KNOWN double-well barrier height. This validates the
    // method, not just CPU==GPU agreement (the stronger check in PATTERNS §4).
    const std::vector<double> pmf = reconstruct_pmf(c, acc_gpu);

    // Recovered barrier = PMF at the central barrier bin (x ~ 0), since the wells
    // (PMF minima) are shifted to 0. Locate the bin nearest x = 0.
    int bar_bin = bin_of(c, 0.0);
    if (bar_bin < 0) bar_bin = c.n_bins / 2;
    const double recovered_barrier = pmf[bar_bin];
    const double true_barrier      = c.u_barrier;        // analytic answer (kT)
    // The recovered barrier is checked to a documented PHYSICAL tolerance, NOT to
    // machine precision, for two honest reasons (see THEORY §6, "Numerical
    // considerations" and "How we verify correctness"):
    //   (1) finite sampling -- a short ensemble has statistical noise; and
    //   (2) the 2nd-order CUMULANT truncation in the reweighting systematically
    //       overestimates barriers when the boost variance is non-negligible
    //       (a real, well-known GaMD limitation -- the boost must be kept gentle).
    // With the gentle boost in the committed sample (k0=0.15) the recovery lands
    // within ~0.3 kT; we allow a band of ~0.6 kT (= 0.2*barrier) so the gate is a
    // genuine correctness check, not a rubber stamp.
    const double BARRIER_TOL = 0.20 * true_barrier;      // physical teaching band (kT)
    const double barrier_err = std::fabs(recovered_barrier - true_barrier);
    const bool barrier_ok = barrier_err <= BARRIER_TOL;

    // Cross-well sampling: with the boost ON, walkers that started in one well
    // should have populated BOTH wells. Count how many of the visited bins lie in
    // each well (x<0 vs x>0) to show enhanced sampling crossed the barrier.
    int left_bins = 0, right_bins = 0;
    for (int b = 0; b < c.n_bins; ++b) {
        if (acc_gpu[acc_count_idx(c, b)] > 0) {
            (bin_center(c, b) < 0.0 ? left_bins : right_bins)++;
        }
    }
    const bool both_wells = (left_bins > 0 && right_bins > 0);

    const bool pass = tally_exact && barrier_ok && both_wells;

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("model: double-well U(x)=%.1f*(x^2-1)^2  kT=%.2f  walkers=%d  steps=%d (equil %d)\n",
                c.u_barrier, c.kT, c.n_walkers, c.steps, c.equil_steps);
    std::printf("GaMD boost: E=%.2f  k0=%.2f  k=%.4f  (dV=0.5*k*(E-U)^2 for U<E)\n",
                c.e_threshold, c.k0, compute_k(c));
    std::printf("reweighted PMF (2nd-order cumulant), F(x) in kT, min shifted to 0:\n");

    // Print a fixed, deterministic set of bin centers spanning the range so the
    // double-well shape is visible (and identical every run). We sample 9 bins.
    const int n_show = 9;
    for (int s = 0; s < n_show; ++s) {
        const int b = (s * (c.n_bins - 1)) / (n_show - 1);   // evenly spaced bins
        const double xc = bin_center(c, b);
        const double f_sim = pmf[b];
        const double f_ana = analytic_pmf(c, xc);            // true U(x) (min 0)
        if (f_sim < 1e29) {
            std::printf("  x=%+5.2f : F_sim=%6.2f  F_true=%6.2f\n", xc, f_sim, f_ana);
        } else {
            std::printf("  x=%+5.2f : F_sim=  n/a  F_true=%6.2f\n", xc, f_ana);
        }
    }
    std::printf("recovered barrier height = %.2f kT  (true = %.2f kT)\n",
                recovered_barrier, true_barrier);
    std::printf("enhanced sampling: %d left-well + %d right-well bins visited (both wells: %s)\n",
                left_bins, right_bins, both_wells ? "yes" : "no");
    std::printf("RESULT: %s (GPU tally == CPU exactly; barrier recovered; both wells sampled)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%d walkers x %d steps)\n",
                 path.c_str(), c.n_walkers, c.steps);
    std::fprintf(stderr, "[timing] CPU: %.3f ms   GPU kernel: %.3f ms\n", cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- speed-up grows with #walkers x #steps; "
                         "production GaMD boosts a full all-atom force field.\n");
    std::fprintf(stderr, "[verify] GPU-vs-CPU worst tally diff = %lld (must be 0)\n",
                 (long long)worst_acc_diff);
    std::fprintf(stderr, "[verify] barrier err = %.3f kT (tol %.3f); cross-well sampling = %s\n",
                 barrier_err, BARRIER_TOL, both_wells ? "yes" : "no");

    return pass ? 0 : 1;
}
