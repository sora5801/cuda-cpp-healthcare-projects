// ===========================================================================
// src/main.cu  --  Entry point: load DWI, fit tensors + trace, verify, report
// ---------------------------------------------------------------------------
// Project 4.15 : Diffusion MRI & Tractography
//
// THE 5-STEP SHAPE every project in this repo follows:
//   1. Load the problem (a small DWI volume from data/sample).
//   2. CPU reference (reference_cpu.cpp)  -> trusted DTI fit + streamlines.
//   3. GPU result    (kernels.cu)         -> the thing being taught.
//   4. VERIFY: GPU agrees with CPU within a documented tolerance.
//   5. REPORT: deterministic FA/MD summary + streamline stats to stdout;
//              timing + verification error to stderr.
//
// STDOUT is byte-for-byte deterministic (diffed by demo/run_demo against
// demo/expected_output.txt); run-to-run timings go to STDERR.
//
// PIPELINE: fit every voxel's diffusion tensor (kernel 1) -> pick the highest-FA
// tissue voxels as tractography seeds -> trace deterministic streamlines through
// the fitted direction field (kernel 2). Both kernels are verified against the
// CPU reference.
//
// Code tour: start here, then kernels.cuh -> kernels.cu, then dti_core.h /
// tract_core.h (the shared physics), then reference_cpu.*.
// ===========================================================================
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "kernels.cuh"        // fit_gpu, tract_gpu
#include "reference_cpu.h"    // load_dwi, make_gradient_scheme, build_pseudo_inverse, CPU refs
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "4.15";
static const char* PROJECT_NAME = "Diffusion MRI & Tractography";

// --- Tolerances (documented; see THEORY "How we verify correctness") --------
// The scalar-map fit uses IDENTICAL double-precision math on CPU and GPU (shared
// dti_core.h), so FA/MD/eigenvalues agree to near machine precision; 1e-9 is a
// generous safety margin over the ~1e-12 we actually observe.
static constexpr double FIT_TOL = 1.0e-9;
// Streamlines share the same stepping code but accumulate thousands of FMA ops;
// the GPU's fused-multiply-add and the host compiler can diverge by ~1e-5 per
// step, so we verify streamline geometry to a physically-negligible 1e-3 voxels
// (a small fraction of a voxel). This is the honest tolerance from PATTERNS.md §4.
static constexpr double TRACT_TOL = 1.0e-3;

// --- Tractography parameters (fixed so the demo is deterministic) -----------
static constexpr int   MAX_STEPS = 200;    // cap on vertices per half-streamline
static constexpr float STEP      = 0.5f;   // Euler step length (voxels)
static constexpr float FA_MIN    = 0.15f;  // stop below this anisotropy
static constexpr float COS_MIN   = 0.80f;  // stop on turns sharper than acos(0.8)~37 deg
static constexpr int   N_SEEDS   = 5;      // number of highest-FA seeds to trace

// Largest per-voxel discrepancy between two fit arrays (over FA, MD, eigenvalues).
// This single number is the headline correctness metric for the fit.
static double fit_max_err(const std::vector<VoxelResult>& a,
                          const std::vector<VoxelResult>& b) {
    if (a.size() != b.size()) return 1e300;
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, std::fabs(a[i].fa - b[i].fa));
        worst = std::max(worst, std::fabs(a[i].md - b[i].md));
        worst = std::max(worst, std::fabs(a[i].l1 - b[i].l1));
        worst = std::max(worst, std::fabs(a[i].l2 - b[i].l2));
        worst = std::max(worst, std::fabs(a[i].l3 - b[i].l3));
    }
    return worst;
}

// Largest positional discrepancy between two streamline sets (over all points).
// If the streamlines have different lengths we return a large number so a
// structural mismatch cannot masquerade as agreement.
static double tract_max_err(const std::vector<Streamline>& a,
                            const std::vector<Streamline>& b) {
    if (a.size() != b.size()) return 1e300;
    double worst = 0.0;
    for (std::size_t s = 0; s < a.size(); ++s) {
        if (a[s].nsteps != b[s].nsteps) return 1e300;
        for (std::size_t k = 0; k < a[s].pts.size(); ++k)
            worst = std::max(worst, std::fabs((double)a[s].pts[k] - (double)b[s].pts[k]));
    }
    return worst;
}

// Pick the N highest-FA tissue voxels as deterministic seeds (ties -> lower voxel
// index). Returns a flat [3*N] list of (x,y,z) voxel-center coordinates. Seeding
// on high anisotropy is the standard "seed in white matter" heuristic.
static std::vector<float> pick_seeds(const DwiVolume& vol,
                                     const std::vector<VoxelResult>& fit, int nseeds) {
    std::vector<int> idx;
    for (int v = 0; v < vol.nvox; ++v)
        if (vol.mask[v]) idx.push_back(v);
    const int k = std::min<int>(nseeds, static_cast<int>(idx.size()));
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int p, int q) {
        if (fit[p].fa != fit[q].fa) return fit[p].fa > fit[q].fa;  // higher FA first
        return p < q;                                              // tie -> lower index
    });
    std::vector<float> seeds;
    for (int i = 0; i < k; ++i) {
        const int v = idx[i];
        const int x = v % vol.nx;
        const int y = (v / vol.nx) % vol.ny;
        const int z = v / (vol.nx * vol.ny);
        seeds.push_back((float)x); seeds.push_back((float)y); seeds.push_back((float)z);
    }
    return seeds;
}

int main(int argc, char** argv) {
    // ---- 1. Load ----------------------------------------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/dwi_sample.txt";
    DwiVolume vol;
    try {
        vol = load_dwi(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }
    // The fixed acquisition geometry and its once-per-run OLS operator.
    const GradientScheme scheme = make_gradient_scheme();
    const std::vector<double> Minv = build_pseudo_inverse(scheme);

    // ---- 2. CPU reference: fit all voxels (timed) -------------------------
    std::vector<VoxelResult> fit_cpu;
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    fit_all_voxels_cpu(vol, Minv, fit_cpu);
    const double cpu_fit_ms = cpu_timer.stop_ms();

    // ---- 3. GPU: fit all voxels (kernel timed inside the wrapper) ---------
    std::vector<VoxelResult> fit_gpu_res;
    float gpu_fit_ms = 0.0f;
    fit_gpu(vol, Minv, fit_gpu_res, &gpu_fit_ms);

    // ---- 4a. Verify the fit -----------------------------------------------
    const double fit_err = fit_max_err(fit_cpu, fit_gpu_res);
    const bool fit_pass = fit_err <= FIT_TOL;

    // Seeds are chosen (deterministically) from the verified GPU fit.
    const std::vector<float> seeds = pick_seeds(vol, fit_gpu_res, N_SEEDS);

    // ---- 2b/3b. Tractography: CPU reference then GPU ----------------------
    // IMPORTANT DESIGN CHOICE: both tractographies trace through the SAME (already
    // verified, CPU==GPU to ~1e-12) fitted field -- the GPU fit. This is exactly
    // what a real pipeline does: fit once, then trace. It also isolates the two
    // verification stages: stage 4a checks the FIT kernel, stage 4b checks the
    // TRACTOGRAPHY kernel, without one stage's round-off leaking into the other.
    // (Tractography is a threshold-sensitive integrator: feeding it the CPU fit vs
    // the GPU fit -- which differ by a harmless ~1e-12 -- can change WHERE a
    // streamline crosses the FA_MIN stop boundary and thus its length. That real
    // sensitivity is discussed in THEORY "Numerical considerations"; here we avoid
    // it by tracing both on the one verified field.)
    std::vector<Streamline> lines_cpu, lines_gpu;
    cpu_timer.start();
    trace_streamlines_cpu(vol, fit_gpu_res, seeds, MAX_STEPS, STEP, FA_MIN, COS_MIN, lines_cpu);
    const double cpu_tract_ms = cpu_timer.stop_ms();
    float gpu_tract_ms = 0.0f;
    tract_gpu(fit_gpu_res, vol, seeds, MAX_STEPS, STEP, FA_MIN, COS_MIN, lines_gpu, &gpu_tract_ms);

    // ---- 4b. Verify tractography ------------------------------------------
    const double tract_err = tract_max_err(lines_cpu, lines_gpu);
    const bool tract_pass = tract_err <= TRACT_TOL;
    const bool pass = fit_pass && tract_pass;

    // ---- 5a. Deterministic report -> STDOUT -------------------------------
    // Global FA statistics over tissue voxels (deterministic aggregates).
    int ntissue = 0; double fa_sum = 0.0, fa_max = 0.0, md_sum = 0.0;
    for (int v = 0; v < vol.nvox; ++v) if (vol.mask[v]) {
        ntissue++; fa_sum += fit_gpu_res[v].fa; md_sum += fit_gpu_res[v].md;
        fa_max = std::max(fa_max, fit_gpu_res[v].fa);
    }
    const double fa_mean = ntissue ? fa_sum / ntissue : 0.0;
    const double md_mean = ntissue ? md_sum / ntissue : 0.0;

    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("DWI volume: %dx%dx%d = %d voxels, %d measurements (1 b0 + %d dirs)\n",
                vol.nx, vol.ny, vol.nz, vol.nvox, NMEAS, NDIR);
    std::printf("DTI fit (per-voxel, %d tissue voxels):\n", ntissue);
    std::printf("  mean FA = %.6f   max FA = %.6f   mean MD = %.6e mm^2/s\n",
                fa_mean, fa_max, md_mean);
    // A few individual voxel fits for a concrete, checkable readout (the seeds).
    std::printf("seed voxels (highest FA), tensor fit:\n");
    for (int i = 0; i < static_cast<int>(seeds.size() / 3); ++i) {
        const int x = (int)seeds[3*i+0], y = (int)seeds[3*i+1], z = (int)seeds[3*i+2];
        const int v = (z * vol.ny + y) * vol.nx + x;
        const VoxelResult& R = fit_gpu_res[v];
        std::printf("  (%2d,%2d,%2d)  FA=%.6f  MD=%.4e  v1=(% .4f,% .4f,% .4f)\n",
                    x, y, z, R.fa, R.md, R.v1x, R.v1y, R.v1z);
    }
    std::printf("tractography (%d seeds, step=%.2f, FA_min=%.2f):\n",
                (int)(seeds.size() / 3), STEP, FA_MIN);
    long total_pts = 0;
    for (std::size_t s = 0; s < lines_gpu.size(); ++s) {
        std::printf("  streamline %zu: %d points\n", s, lines_gpu[s].nsteps);
        total_pts += lines_gpu[s].nsteps;
    }
    std::printf("  total streamline points: %ld\n", total_pts);
    std::printf("RESULT: %s (GPU matches CPU: fit tol=%.0e, tract tol=%.0e)\n",
                pass ? "PASS" : "FAIL", FIT_TOL, TRACT_TOL);

    // ---- 5b. Varying detail -> STDERR -------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d voxels x %d measurements)\n",
                 path.c_str(), vol.nvox, NMEAS);
    std::fprintf(stderr, "[timing] fit  : CPU %.3f ms   GPU %.3f ms\n", cpu_fit_ms, gpu_fit_ms);
    std::fprintf(stderr, "[timing] tract: CPU %.3f ms   GPU %.3f ms\n", cpu_tract_ms, gpu_tract_ms);
    std::fprintf(stderr, "[timing] teaching artifact only -- this tiny volume is dominated by "
                         "launch/copy overhead; the GPU wins at whole-brain scale (~10^5-10^6 voxels).\n");
    std::fprintf(stderr, "[verify] fit   max_abs_err = %.3e  (tol %.1e)\n", fit_err, FIT_TOL);
    std::fprintf(stderr, "[verify] tract max_abs_err = %.3e  (tol %.1e)\n", tract_err, TRACT_TOL);

    return pass ? 0 : 1;
}
