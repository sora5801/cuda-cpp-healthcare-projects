// ===========================================================================
// src/main.cu  --  Entry point: load field map, invert (CPU+GPU), verify, report
// ---------------------------------------------------------------------------
// Project 4.22 : Quantitative Susceptibility Mapping (QSM)
//
// THE SHAPE (the repo's standard 5-step main):
//   1. Load the measured FIELD-SHIFT volume (data/sample) -- the QSM input.
//   2. CPU reference reconstructions of the susceptibility chi:
//        * TKD (Threshold-based K-space Division),
//        * Tikhonov closed-form (Wiener filter),
//        * Tikhonov ITERATIVE (gradient descent) -- the pattern the GPU mirrors.
//   3. GPU reconstructions with cuFFT (kernels.cu): TKD and Tikhonov iterative.
//   4. VERIFY three things:
//        (a) GPU TKD      == CPU TKD      within tolerance,
//        (b) GPU iterative== CPU iterative within tolerance,
//        (c) the ITERATIVE solve CONVERGED to the closed-form minimizer
//            (a real algorithmic check, not just CPU==GPU agreement).
//   5. REPORT deterministic reconstruction scalars to stdout; timings to stderr.
//
// GROUND-TRUTH RECOVERY (the science check)
//   The committed sample is the field map SYNTHESIZED from a known susceptibility
//   phantom (make_synthetic.py). main.cu re-derives that SAME known phantom via
//   make_ground_truth_chi() (deterministic, identical formula), so it can report
//   how well the reconstruction recovered the true chi at the known "iron blob"
//   locations -- turning an abstract inverse problem into an interpretable number.
//
// STDOUT is byte-for-byte deterministic (diffed by demo/run_demo); timings and
// run-varying numbers go to STDERR (shown, not diffed). Scalars are printed at a
// fixed number of decimals so ~1e-12 FFT/FMA noise never flips the stdout bytes.
//
// Code tour: start here, then kernels.cuh -> kernels.cu (cuFFT), qsm_core.h (the
// shared per-bin math), reference_cpu.cpp (the direct-DFT baseline).
// ===========================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.cuh"        // reconstruct_tkd_gpu, reconstruct_tikhonov_iter_gpu
#include "reference_cpu.h"    // Volume, CPU reconstructions, make_field_from_chi
#include "util/io.hpp"        // util::CpuTimer

static const char* PROJECT_ID   = "4.22";
static const char* PROJECT_NAME = "Quantitative Susceptibility Mapping (QSM)";

// ---- Fixed reconstruction parameters (shared contract with make_synthetic.py) --
//   TKD_THRESHOLD : clamp |D(k)| to at least this before dividing (bias/variance).
//   TIK_ALPHA     : Tikhonov regularization weight for the least-squares inverse.
//   TIK_STEP      : gradient-descent step size for the iterative solve.
//   TIK_ITERS     : number of gradient iterations (enough to converge on this grid).
// These MUST match the phantom the generator built (data/README documents it).
static constexpr double TKD_THRESHOLD = 0.15;
static constexpr double TIK_ALPHA     = 0.05;
static constexpr double TIK_STEP      = 0.5;
static constexpr int    TIK_ITERS     = 200;

// Verification tolerance. Both paths run in double precision, but cuFFT's FFT and
// the CPU's direct DFT order their sums differently, and fused-multiply-add
// reorders arithmetic, so the two reconstructions match to ~1e-9 per voxel, not
// bit-exactly. We assert an RMS voxel difference below a physically-negligible
// 1e-6 on chi values of order 0.1..1. Documented and honest (PATTERNS.md 4).
static constexpr double ATOL = 1.0e-6;

// The iterative solve should converge CLOSE to the closed-form Tikhonov minimizer,
// but gradient descent stops at finite iterations, so we allow a looser (still
// small) gap here and print the actual gap on stderr.
static constexpr double CONVERGE_TOL = 5.0e-3;

// ---------------------------------------------------------------------------
// make_ground_truth_chi: rebuild the KNOWN synthetic susceptibility phantom the
// generator used, so we can score recovery. This MUST match make_synthetic.py's
// make_ground_truth() exactly (same grid, same source voxels, same values).
//   A dim zero background with a handful of compact "sources":
//     * three positive blobs (paramagnetic, like iron-rich deep-brain nuclei),
//     * one negative blob (diamagnetic, like a calcification).
//   Values are in ppm-like units (chi is dimensionless, ~1e-6 in tissue; we use
//   O(1) numbers so the demo prints legibly -- a scaling, not a physics change).
// Returns the ground-truth chi Volume of the given dimensions.
// ---------------------------------------------------------------------------
static Volume make_ground_truth_chi(int nx, int ny, int nz) {
    Volume chi;
    chi.nx = nx; chi.ny = ny; chi.nz = nz;
    chi.vox.assign(static_cast<std::size_t>(nx) * ny * nz, 0.0);   // zero background

    // Compact susceptibility sources at fixed grid fractions. (x,y,z,value).
    struct Src { int x, y, z; double val; };
    const Src sources[] = {
        { nx / 4,     ny / 2,     nz / 2, 1.0 },   // paramagnetic blob 1
        { nx / 2,     ny / 2,     nz / 2, 0.6 },   // paramagnetic blob 2 (center)
        { 3 * nx / 4, ny / 2,     nz / 2, 0.8 },   // paramagnetic blob 3
        { nx / 2,     ny / 4,     nz / 2, -0.7 },  // diamagnetic blob (calcification)
    };
    for (const Src& s : sources)
        if (s.x >= 0 && s.x < nx && s.y >= 0 && s.y < ny && s.z >= 0 && s.z < nz)
            chi.vox[static_cast<std::size_t>(chi.idx(s.x, s.y, s.z))] = s.val;
    return chi;
}

// Print the reconstructed chi at the known source voxels: the single most
// interpretable check that inversion recovered the phantom. Rounded to 4 decimals
// so the low, FFT-noisy digits never flip the deterministic stdout bytes.
static void report_sources(const char* label, const Volume& chi) {
    const int nx = chi.nx, ny = chi.ny, nz = chi.nz;
    const int xs[4] = { nx / 4, nx / 2, 3 * nx / 4, nx / 2 };
    const int ys[4] = { ny / 2, ny / 2, ny / 2,     ny / 4 };
    std::printf("%s recovered chi at sources:", label);
    for (int s = 0; s < 4; ++s) {
        const double v = chi.vox[static_cast<std::size_t>(chi.idx(xs[s], ys[s], nz / 2))];
        std::printf(" %+.4f", v);
    }
    std::printf("\n");
}

int main(int argc, char** argv) {
    // ---- 1. Load the measured field-shift volume -------------------------
    const std::string path = (argc > 1) ? argv[1] : "data/sample/field_map.txt";
    Volume field;
    try {
        field = load_volume(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[error] %s\n", e.what());
        return 2;
    }

    // Rebuild the known phantom so we can score recovery (see note at top).
    const Volume truth = make_ground_truth_chi(field.nx, field.ny, field.nz);

    // ---- 2. CPU reference reconstructions (timed) ------------------------
    util::CpuTimer cpu_timer;
    cpu_timer.start();
    const Volume chi_tkd_cpu  = reconstruct_tkd_cpu(field, TKD_THRESHOLD);
    const Volume chi_wien_cpu = reconstruct_tikhonov_cpu(field, TIK_ALPHA);
    const Volume chi_iter_cpu = reconstruct_tikhonov_iter_cpu(field, TIK_ALPHA,
                                                              TIK_STEP, TIK_ITERS);
    const double cpu_ms = cpu_timer.stop_ms();

    // ---- 3. GPU reconstructions with cuFFT (timed inside) ----------------
    Volume chi_tkd_gpu, chi_iter_gpu;
    float gpu_tkd_ms = 0.0f, gpu_iter_ms = 0.0f;
    reconstruct_tkd_gpu(field, TKD_THRESHOLD, chi_tkd_gpu, &gpu_tkd_ms);
    reconstruct_tikhonov_iter_gpu(field, TIK_ALPHA, TIK_STEP, TIK_ITERS,
                                  chi_iter_gpu, &gpu_iter_ms);

    // ---- 4. Verify -------------------------------------------------------
    const double err_tkd  = rms_diff(chi_tkd_cpu,  chi_tkd_gpu);   // (a) GPU==CPU TKD
    const double err_iter = rms_diff(chi_iter_cpu, chi_iter_gpu);  // (b) GPU==CPU iter
    const double gap_conv = rms_diff(chi_iter_cpu, chi_wien_cpu);  // (c) iter->closed form
    const bool pass = (err_tkd <= ATOL) && (err_iter <= ATOL) && (gap_conv <= CONVERGE_TOL);

    // Data-consistency of the TKD reconstruction: re-apply the forward dipole model
    // to the recovered chi and compare with the input field. A small residual means
    // the reconstruction actually explains the data (a ground-truth-free quality
    // metric, in addition to the phantom-recovery numbers below).
    const Volume field_refit = make_field_from_chi(chi_tkd_gpu);
    const double data_resid  = rms_diff(field_refit, field);

    // ---- 5a. Deterministic report -> STDOUT ------------------------------
    std::printf("%s -- %s\n", PROJECT_ID, PROJECT_NAME);
    std::printf("volume: %dx%dx%d voxels   B0 || z   dipole kernel D(k)=1/3 - kz^2/|k|^2\n",
                field.nx, field.ny, field.nz);
    std::printf("methods: TKD(thr=%.2f)  Tikhonov(alpha=%.2f, closed-form + %d-iter GD, step=%.2f)\n",
                TKD_THRESHOLD, TIK_ALPHA, TIK_ITERS, TIK_STEP);

    // Recovered susceptibility at the four known source voxels, per method.
    report_sources("TKD ", chi_tkd_gpu);
    report_sources("TIK ", chi_iter_gpu);
    std::printf("ground-truth chi at sources: %+.4f %+.4f %+.4f %+.4f\n",
                truth.vox[static_cast<std::size_t>(truth.idx(field.nx/4,   field.ny/2, field.nz/2))],
                truth.vox[static_cast<std::size_t>(truth.idx(field.nx/2,   field.ny/2, field.nz/2))],
                truth.vox[static_cast<std::size_t>(truth.idx(3*field.nx/4, field.ny/2, field.nz/2))],
                truth.vox[static_cast<std::size_t>(truth.idx(field.nx/2,   field.ny/4, field.nz/2))]);

    // Global RMS agreement between each reconstruction and the truth (rounded).
    std::printf("chi RMS vs ground truth:  TKD=%.4f  Tikhonov=%.4f\n",
                rms_diff(chi_tkd_gpu, truth), rms_diff(chi_iter_gpu, truth));
    std::printf("data-consistency residual (forward-model refit): %.4f\n", data_resid);
    std::printf("RESULT: %s (GPU cuFFT inversion matches CPU reference; iterative solve converged)\n",
                pass ? "PASS" : "FAIL");

    // ---- 5b. Varying detail -> STDERR ------------------------------------
    std::fprintf(stderr, "[data]   source: %s  (%d x %d x %d field map)\n",
                 path.c_str(), field.nx, field.ny, field.nz);
    std::fprintf(stderr, "[timing] CPU (direct DFT, 3 methods): %.3f ms\n", cpu_ms);
    std::fprintf(stderr, "[timing] GPU cuFFT: TKD %.3f ms   Tikhonov %d-iter %.3f ms\n",
                 gpu_tkd_ms, TIK_ITERS, gpu_iter_ms);
    std::fprintf(stderr, "[timing] teaching artifact -- the CPU does an O(N^2) direct DFT; "
                         "cuFFT is O(N log N). The GPU's edge explodes with volume size.\n");
    std::fprintf(stderr, "[verify] RMS(GPU-CPU) TKD=%.3e  iter=%.3e  (atol=%.1e)\n",
                 err_tkd, err_iter, ATOL);
    std::fprintf(stderr, "[verify] iterative->closed-form gap = %.3e  (converge tol=%.1e)\n",
                 gap_conv, CONVERGE_TOL);

    return pass ? 0 : 1;
}
