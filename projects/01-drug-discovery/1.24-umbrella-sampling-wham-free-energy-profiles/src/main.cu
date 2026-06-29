// ===========================================================================
// src/main.cu  --  Entry point: sample windows (CPU+GPU), verify, WHAM, report
// ---------------------------------------------------------------------------
// Project 1.24 : Umbrella Sampling / WHAM Free Energy Profiles
//
// WHAT THIS FILE DOES  (the shape EVERY project in this repo follows)
//   1. Load the experiment (data/sample/umbrella.txt, or an argv path).
//   2. CPU reference: simulate every umbrella window serially (reference_cpu.cpp).
//   3. GPU: one thread per window, identical Langevin physics (kernels.cu).
//   4. VERIFY: the GPU histograms must equal the CPU histograms EXACTLY (integer
//      counts, identical RNG/dynamics -> bit-for-bit). Then run WHAM on the GPU
//      histograms and check the reconstructed PMF recovers the KNOWN double-well
//      to a small physical tolerance (a second, scientific check).
//   5. REPORT: the PMF at a few bin centers + headline numbers to STDOUT
//      (deterministic, diffed by the demo); timings to STDERR (shown, not diffed).
//
// Code tour: read this first, then umbrella.h (the physics), kernels.cuh/.cu
// (the GPU mapping), reference_cpu.cpp (the baseline + WHAM). See ../THEORY.md.
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // sample_windows_gpu (GPU path)
#include "reference_cpu.h"    // load_config, sample_windows_cpu, wham_solve (CPU)
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "1.24";
static const char* PROJECT_NAME = "Umbrella Sampling / WHAM Free Energy Profiles";

// Number of WHAM self-consistency sweeps. ~200 is far more than enough for this
// well-conditioned 1-D problem to converge (the per-window shifts stop moving);
// it is fixed so the reported PMF is byte-identical every run.
static constexpr int WHAM_ITERS = 200;

// Scientific tolerance for "the reconstructed PMF recovers the true double-well".
// This is a SAMPLING comparison, not a round-off one: even with identical CPU/GPU
// histograms, the WHAM estimate differs from the analytic U by finite-sampling
// noise. 0.30 kT over the interior of the scan is a generous, honest bound for
// the small demo (more steps -> tighter; see the exercises). THEORY.md section
// "How we verify correctness" explains why we compare on the INTERIOR only.
static constexpr double PMF_TOL_KT = 0.30;

// We judge the PMF only where the coordinate is well-sampled: the OUTERMOST
// windows have a one-sided harmonic well (no neighbour beyond the edge), so their
// bins are noisier. We exclude a margin of this many coordinate units from each
// end of the window span before comparing to the analytic U. (Honest reporting,
// not cherry-picking: the full sweep is still printed.)
static constexpr double PMF_EDGE_MARGIN = 0.25;

int main(int argc, char** argv) {
    // ---- 1. Load the experiment --------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/umbrella.txt";
    UmbrellaConfig c;
    try {
        c = load_config(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // ---- 2. CPU reference (timed) ------------------------------------------
    std::vector<unsigned int> hist_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    sample_windows_cpu(c, hist_cpu);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU ensemble (kernel timed inside the wrapper) -----------------
    std::vector<unsigned int> hist_gpu;
    float gpu_kernel_ms = 0.0f;
    sample_windows_gpu(c, hist_gpu, &gpu_kernel_ms);

    // ---- 4a. Verify the histograms are bit-identical -----------------------
    // Integer counts + identical RNG/dynamics => the GPU and CPU histograms must
    // match EXACTLY. A single mismatch means the shared physics diverged.
    long long total_cpu = 0, total_gpu = 0, mismatches = 0;
    for (std::size_t i = 0; i < hist_cpu.size(); ++i) {
        total_cpu += hist_cpu[i];
        total_gpu += hist_gpu[i];
        if (hist_cpu[i] != hist_gpu[i]) ++mismatches;
    }
    const bool hist_match = (mismatches == 0) && (hist_cpu.size() == hist_gpu.size());

    // ---- 4b. WHAM on the GPU histograms; compare PMF to the true potential --
    std::vector<double> pmf;
    int n_used = 0;
    wham_solve(c, hist_gpu, WHAM_ITERS, pmf, &n_used);

    // The interior comparison span (window span shrunk by the edge margin).
    const double cmp_lo = c.win_min + PMF_EDGE_MARGIN;
    const double cmp_hi = c.win_max - PMF_EDGE_MARGIN;

    // The analytic PMF is the bare potential U(x) shifted so its minimum over the
    // comparison span is 0 (PMFs are relative). Both WHAM and U are zeroed at the
    // SAME reference so the comparison is apples-to-apples.
    double pmf_min_analytic = 1.0e30;
    for (int i = 0; i < c.grid.nbins; ++i) {
        const double xi = grid_bin_center(c.grid, i);
        if (xi >= cmp_lo && xi <= cmp_hi)
            pmf_min_analytic = std::fmin(pmf_min_analytic, potential_U(c.pot, xi));
    }
    double worst_pmf_err = 0.0;
    int    compared = 0;
    for (int i = 0; i < c.grid.nbins; ++i) {
        const double xi = grid_bin_center(c.grid, i);
        if (xi < cmp_lo || xi > cmp_hi) continue;         // interior, well-sampled
        if (pmf[i] > 1.0e29) continue;                    // skip unsampled bins
        const double analytic = potential_U(c.pot, xi) - pmf_min_analytic;
        worst_pmf_err = std::fmax(worst_pmf_err, std::fabs(pmf[i] - analytic));
        ++compared;
    }
    const bool pmf_ok = (compared > 0) && (worst_pmf_err <= PMF_TOL_KT);
    const bool pass = hist_match && pmf_ok;

    // Locate the barrier (the PMF maximum over the sampled span) -- the headline
    // free-energy quantity umbrella sampling exists to measure.
    int    barrier_bin = -1;
    double barrier_pmf = -1.0;
    for (int i = 0; i < c.grid.nbins; ++i) {
        const double xi = grid_bin_center(c.grid, i);
        if (xi < c.win_min || xi > c.win_max) continue;
        if (pmf[i] > 1.0e29) continue;
        if (pmf[i] > barrier_pmf) { barrier_pmf = pmf[i]; barrier_bin = i; }
    }

    // ---- 5a. Deterministic report -> STDOUT (diffed by the demo) -----------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("double-well U(x) = A (x^2 - b^2)^2 / b^4   [A=%.2f kT, b=%.2f]\n",
                c.pot.A, c.pot.b);
    std::printf("%d windows over x0 in [%.2f, %.2f], k_spring=%.2f kT; "
                "%d sample steps/window\n",
                c.n_windows, c.win_min, c.win_max, c.k_spring, c.n_sample);
    std::printf("grid: %d bins over [%.2f, %.2f]; sampled bins = %d\n",
                c.grid.nbins, c.grid.x_min, c.grid.x_max, n_used);

    // Print the WHAM PMF at a fixed set of bin centers (deterministic): a coarse
    // sweep across the coordinate so the learner can see the two wells (~0 kT) and
    // the barrier (~A kT). We pick every (nbins/8)-th bin for a stable layout.
    std::printf("WHAM PMF (kT) vs analytic U at sampled bin centers:\n");
    const int stride = (c.grid.nbins >= 8) ? c.grid.nbins / 8 : 1;
    for (int i = 0; i < c.grid.nbins; i += stride) {
        const double xi = grid_bin_center(c.grid, i);
        if (xi < c.win_min || xi > c.win_max || pmf[i] > 1.0e29) continue;
        const double analytic = potential_U(c.pot, xi) - pmf_min_analytic;
        std::printf("  x=%+.3f : WHAM %6.3f   U %6.3f\n", xi, pmf[i], analytic);
    }
    if (barrier_bin >= 0) {
        std::printf("barrier: PMF max %.3f kT at x=%+.3f (true A=%.3f kT)\n",
                    barrier_pmf, grid_bin_center(c.grid, barrier_bin), c.pot.A);
    }
    std::printf("histograms: %lld total counts; GPU==CPU bins: %s\n",
                total_gpu, hist_match ? "YES" : "NO");
    std::printf("RESULT: %s (GPU histograms == CPU exactly; WHAM PMF within %.2f kT of U)\n",
                pass ? "PASS" : "FAIL", PMF_TOL_KT);

    // ---- 5b. Varying detail -> STDERR (shown, not diffed) ------------------
    std::fprintf(stderr, "[data]   source: %s  (%d windows x %d bins)\n",
                 path.c_str(), c.n_windows, c.grid.nbins);
    std::fprintf(stderr, "[timing] CPU sampling: %.3f ms   GPU kernel: %.3f ms\n",
                 cpu_ms, gpu_kernel_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- per-window work is small here; "
                         "the GPU's edge grows with windows and steps.\n");
    std::fprintf(stderr, "[verify] histogram mismatches = %lld  (must be 0)\n", mismatches);
    std::fprintf(stderr, "[verify] worst |WHAM-U| over %d sampled bins = %.4f kT (tol %.2f)\n",
                 compared, worst_pmf_err, PMF_TOL_KT);

    return pass ? 0 : 1;
}
