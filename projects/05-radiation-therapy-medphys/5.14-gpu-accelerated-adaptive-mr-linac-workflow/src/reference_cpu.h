// ===========================================================================
// src/reference_cpu.h  --  oART job description + CPU reference workflow
// ---------------------------------------------------------------------------
// Project 5.14 : GPU-Accelerated Adaptive MR-Linac Workflow (reduced-scope)
//
// Pure C++ (no CUDA), so kernels.cu can reuse these types and the host compiler
// can build reference_cpu.cpp. The actual per-voxel math lives in the shared
// mrl_registration.h so the CPU reference and the GPU kernels are byte-for-byte
// identical (the key to exact verification).
//
// WHAT THE WORKFLOW COMPUTES (the reduced-scope oART chain)
//   Given a PLANNING MR (fixed image F), a DAILY MR (moving image M), a reference
//   DOSE distribution planned on F, and a GTV (gross tumour volume) mask on F:
//     1. REGISTER  : estimate a displacement field (u,v) that deforms M -> F
//                    (iterative Demons + Gaussian smoothing).
//     2. WARP DOSE : carry the planned dose through (u,v) so it lands on the
//                    daily anatomy (backward-warp / bilinear pull).
//     3. APPRAISE  : compute deterministic plan-approval metrics over the GTV
//                    (mean dose, D95 = dose covering 95% of the target).
//   The CPU path and the GPU path each run all three steps and we compare their
//   outputs voxel-by-voxel (registration field, warped dose) plus the scalar
//   metrics. See ../THEORY.md for the full derivation.
//
// READ THIS AFTER: mrl_registration.h; BEFORE: reference_cpu.cpp, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "mrl_registration.h"   // shared per-voxel physics (host+device)

// ---------------------------------------------------------------------------
// OartCase: everything needed to run one adaptive fraction on a 2-D slice.
//   All images are row-major, length nx*ny, in `double` for exact CPU/GPU parity.
// ---------------------------------------------------------------------------
struct OartCase {
    int nx = 0, ny = 0;          // image dimensions in voxels (x fast, y slow)
    int iters = 0;               // number of Demons iterations
    double sigma = 0.0;          // Gaussian smoothing sigma (voxels) per iteration
    double k_norm = 1.0;         // Thirion normaliser K (squared voxel spacing)
    double dose_thresh = 0.0;    // dose level defining the "target coverage" report

    std::vector<double> fixed;   // planning MR   F  (intensity ~[0,1]), size nx*ny
    std::vector<double> moving;  // daily MR      M  (intensity ~[0,1]), size nx*ny
    std::vector<double> dose;    // planned dose on F (Gray),            size nx*ny
    std::vector<double> gtv;     // GTV mask on F (1.0 inside, 0.0 out), size nx*ny
};

// ---------------------------------------------------------------------------
// OartResult: the outputs we verify and report.
// ---------------------------------------------------------------------------
struct OartResult {
    std::vector<double> u;            // displacement x-component (voxels), nx*ny
    std::vector<double> v;            // displacement y-component (voxels), nx*ny
    std::vector<double> warped_dose;  // dose mapped onto daily anatomy (Gray), nx*ny
    std::vector<double> warped_moving;// moving image after final warp (for a MSE check)
    double mean_gtv_dose = 0.0;       // mean warped dose inside the GTV (Gray)
    double d95 = 0.0;                 // dose covering >=95% of GTV voxels (Gray)
    double gtv_coverage = 0.0;        // fraction of GTV with dose>=dose_thresh, [0,1]
    double mse_before = 0.0;          // mean-squared |M - F| before registration
    double mse_after = 0.0;           // mean-squared |M(warped) - F| after (should drop)
};

// ---------------------------------------------------------------------------
// load_case: read an OartCase from the tiny text format (see data/README.md):
//   line 1: nx ny iters sigma k_norm dose_thresh
//   then 4 blocks of nx*ny whitespace-separated doubles:
//     fixed(F), moving(M), dose, gtv       (row-major, y outer, x inner)
//   Throws std::runtime_error on any malformed input so demos fail loudly.
// ---------------------------------------------------------------------------
OartCase load_case(const std::string& path);

// ---------------------------------------------------------------------------
// compute_metrics: fill mse_before / mse_after / mean_gtv_dose / d95 /
//   gtv_coverage on a result, given the case. Shared by CPU and GPU paths so the
//   reported numbers are computed identically (apples-to-apples).
// ---------------------------------------------------------------------------
void compute_metrics(const OartCase& c, OartResult& r);

// ---------------------------------------------------------------------------
// oart_cpu: the trusted serial reference for the whole 3-step workflow.
//   Runs Demons registration, warps the dose, and computes metrics. This is the
//   baseline the GPU must match within tolerance.
// ---------------------------------------------------------------------------
void oart_cpu(const OartCase& c, OartResult& r);

// ---------------------------------------------------------------------------
// gaussian_kernel_1d: build a normalized 1-D Gaussian of the given sigma with a
//   radius of ceil(3*sigma) (captures >99% of the mass). Shared by CPU and GPU
//   so the separable smoothing is identical on both. Returns the half-width in
//   `radius` and the (2*radius+1) weights (summing to 1) in `w`.
// ---------------------------------------------------------------------------
void gaussian_kernel_1d(double sigma, int& radius, std::vector<double>& w);
