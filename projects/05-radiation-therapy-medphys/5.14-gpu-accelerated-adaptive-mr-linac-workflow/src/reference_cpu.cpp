// ===========================================================================
// src/reference_cpu.cpp  --  The plain-C++ baseline we trust
// ---------------------------------------------------------------------------
// Project 5.14 : GPU-Accelerated Adaptive MR-Linac Workflow (reduced-scope)
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It runs the
//   whole reduced-scope oART chain -- Demons registration, dose warp, plan
//   metrics -- as a single readable sequence of serial loops, no parallelism, no
//   cleverness, so that when the GPU and CPU agree we believe the GPU.
//
//   Compiled by the host C++ compiler only (no CUDA here). The per-voxel math is
//   the shared mrl_registration.h, so this file is really just "loop the shared
//   physics over every voxel" -- the exact operations the GPU kernels perform,
//   in a different order of execution but the same order of arithmetic.
//
// READ THIS AFTER: reference_cpu.h, mrl_registration.h. Compare against
//   kernels.cu (the GPU twin, which calls the SAME shared functions).
// ===========================================================================
#include "reference_cpu.h"

#include <algorithm>   // std::sort, std::max
#include <cmath>       // std::exp, std::ceil, std::sqrt
#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// gaussian_kernel_1d: normalized 1-D Gaussian weights of half-width 3*sigma.
//   Defined here (host) and reused verbatim by the GPU path, which uploads the
//   weights to the device -- so both smooth with identical coefficients.
// ---------------------------------------------------------------------------
void gaussian_kernel_1d(double sigma, int& radius, std::vector<double>& w) {
    // Radius 3*sigma captures >99.7% of the Gaussian mass; at least 1 so the
    // kernel is never empty even for a tiny sigma.
    radius = std::max(1, static_cast<int>(std::ceil(3.0 * sigma)));
    w.assign(static_cast<std::size_t>(2 * radius + 1), 0.0);
    const double inv2s2 = 1.0 / (2.0 * sigma * sigma);  // 1/(2 sigma^2)
    double sum = 0.0;
    for (int t = -radius; t <= radius; ++t) {
        const double g = std::exp(-(t * t) * inv2s2);   // unnormalized weight
        w[static_cast<std::size_t>(t + radius)] = g;
        sum += g;
    }
    // Normalize so the weights sum to exactly 1 (preserves image DC level).
    for (double& x : w) x /= sum;
}

// ---------------------------------------------------------------------------
// smooth_separable_cpu: convolve an image with a 2-D Gaussian, done as two 1-D
//   passes (horizontal then vertical). A 2-D Gaussian is separable, so an
//   O(N * R) two-pass convolution replaces an O(N * R^2) direct one -- the same
//   trick the GPU uses. Borders clamp to edge (matches sample_bilinear's policy).
// ---------------------------------------------------------------------------
static void smooth_separable_cpu(std::vector<double>& img, int nx, int ny,
                                 int radius, const std::vector<double>& w) {
    std::vector<double> tmp(img.size(), 0.0);
    // Horizontal pass: img -> tmp.
    for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
            double acc = 0.0;
            for (int t = -radius; t <= radius; ++t) {
                const int xs = clampi(x + t, nx);
                acc += w[static_cast<std::size_t>(t + radius)] * img[flat_idx(xs, y, nx)];
            }
            tmp[flat_idx(x, y, nx)] = acc;
        }
    // Vertical pass: tmp -> img (in place is fine since we read tmp, write img).
    for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
            double acc = 0.0;
            for (int t = -radius; t <= radius; ++t) {
                const int ys = clampi(y + t, ny);
                acc += w[static_cast<std::size_t>(t + radius)] * tmp[flat_idx(x, ys, nx)];
            }
            img[flat_idx(x, y, nx)] = acc;
        }
}

// ---------------------------------------------------------------------------
// load_case: parse the tiny text sample into an OartCase (format in the header).
// ---------------------------------------------------------------------------
OartCase load_case(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open oART case file: " + path);
    OartCase c;
    if (!(in >> c.nx >> c.ny >> c.iters >> c.sigma >> c.k_norm >> c.dose_thresh))
        throw std::runtime_error("bad header (want 'nx ny iters sigma k_norm dose_thresh') in " + path);
    if (c.nx <= 0 || c.ny <= 0 || c.iters < 0 || c.sigma <= 0.0 || c.k_norm <= 0.0)
        throw std::runtime_error("invalid oART parameters in " + path);

    const std::size_t n = static_cast<std::size_t>(c.nx) * c.ny;
    auto read_block = [&](std::vector<double>& dst, const char* what) {
        dst.assign(n, 0.0);
        for (std::size_t i = 0; i < n; ++i)
            if (!(in >> dst[i]))
                throw std::runtime_error(std::string("short ") + what + " block in " + path);
    };
    read_block(c.fixed,  "fixed(F)");
    read_block(c.moving, "moving(M)");
    read_block(c.dose,   "dose");
    read_block(c.gtv,    "gtv");
    return c;
}

// ---------------------------------------------------------------------------
// warp_image_cpu: backward-warp `src` by displacement (u,v) into `dst`.
//   dst(x,y) = src sampled at (x + u(x,y), y + v(x,y)). This is a pure gather:
//   every output voxel reads independently -> trivially parallel on the GPU.
// ---------------------------------------------------------------------------
static void warp_image_cpu(const std::vector<double>& src, int nx, int ny,
                           const std::vector<double>& u, const std::vector<double>& v,
                           std::vector<double>& dst) {
    dst.assign(src.size(), 0.0);
    for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
            const std::size_t i = flat_idx(x, y, nx);
            const double fx = x + u[i];   // where in `src` this output pulls from
            const double fy = y + v[i];
            dst[i] = sample_bilinear(src.data(), nx, ny, fx, fy);
        }
}

// ---------------------------------------------------------------------------
// compute_metrics: MSE (before/after) + GTV dose statistics. Deterministic:
//   the D95 sort is a stable numeric sort over the (fixed) set of GTV doses.
// ---------------------------------------------------------------------------
void compute_metrics(const OartCase& c, OartResult& r) {
    const std::size_t n = static_cast<std::size_t>(c.nx) * c.ny;

    // MSE before registration: raw |M - F|^2 averaged over all voxels.
    double sb = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = c.moving[i] - c.fixed[i];
        sb += d * d;
    }
    r.mse_before = sb / static_cast<double>(n);

    // MSE after registration: |M(warped) - F|^2 (should be much smaller).
    double sa = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = r.warped_moving[i] - c.fixed[i];
        sa += d * d;
    }
    r.mse_after = sa / static_cast<double>(n);

    // GTV dose statistics on the WARPED dose (the dose actually delivered to the
    // daily anatomy). Collect the dose at each GTV voxel.
    std::vector<double> gtv_doses;
    gtv_doses.reserve(n);
    double sum = 0.0;
    int covered = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (c.gtv[i] > 0.5) {                    // this voxel is inside the target
            const double d = r.warped_dose[i];
            gtv_doses.push_back(d);
            sum += d;
            if (d >= c.dose_thresh) ++covered;
        }
    }
    const std::size_t ng = gtv_doses.size();
    r.mean_gtv_dose = ng ? sum / static_cast<double>(ng) : 0.0;
    r.gtv_coverage  = ng ? static_cast<double>(covered) / static_cast<double>(ng) : 0.0;

    // D95 = the dose level that 95% of GTV voxels receive AT LEAST. Sort ascending
    // and take the voxel at the 5th percentile: 5% get less, 95% get >= it.
    if (ng) {
        std::sort(gtv_doses.begin(), gtv_doses.end());
        // Index of the 5th percentile (floor). E.g. ng=100 -> idx 5 -> 95 above.
        std::size_t idx = static_cast<std::size_t>(0.05 * static_cast<double>(ng));
        if (idx >= ng) idx = ng - 1;
        r.d95 = gtv_doses[idx];
    } else {
        r.d95 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// oart_cpu: the full reference workflow (register -> warp dose -> metrics).
// ---------------------------------------------------------------------------
void oart_cpu(const OartCase& c, OartResult& r) {
    const std::size_t n = static_cast<std::size_t>(c.nx) * c.ny;

    // Displacement field starts at zero (identity transform = no deformation).
    r.u.assign(n, 0.0);
    r.v.assign(n, 0.0);

    // Precompute the fixed-image gradient once (it never changes during the
    // iteration -- only the warped moving image does).
    std::vector<double> gfx(n), gfy(n);
    for (int y = 0; y < c.ny; ++y)
        for (int x = 0; x < c.nx; ++x) {
            gfx[flat_idx(x, y, c.nx)] = grad_x(c.fixed.data(), c.nx, c.ny, x, y);
            gfy[flat_idx(x, y, c.nx)] = grad_y(c.fixed.data(), c.nx, c.ny, x, y);
        }

    // The Gaussian smoothing weights (the diffusion regulariser).
    int radius; std::vector<double> w;
    gaussian_kernel_1d(c.sigma, radius, w);

    std::vector<double> warped(n, 0.0);
    // --- Demons iteration -------------------------------------------------
    for (int it = 0; it < c.iters; ++it) {
        // 1. Warp the moving image with the current field (backward gather).
        warp_image_cpu(c.moving, c.nx, c.ny, r.u, r.v, warped);
        // 2. Add each voxel's demons force to the running displacement field.
        for (int y = 0; y < c.ny; ++y)
            for (int x = 0; x < c.nx; ++x) {
                const std::size_t i = flat_idx(x, y, c.nx);
                double du, dv;
                demons_force(warped[i], c.fixed[i], gfx[i], gfy[i], c.k_norm, &du, &dv);
                r.u[i] += du;
                r.v[i] += dv;
            }
        // 3. Gaussian-smooth the field (elastic regularisation -> plausible,
        //    invertible deformations instead of tearing).
        smooth_separable_cpu(r.u, c.nx, c.ny, radius, w);
        smooth_separable_cpu(r.v, c.nx, c.ny, radius, w);
    }

    // Final warped moving image (for the after-MSE check).
    warp_image_cpu(c.moving, c.nx, c.ny, r.u, r.v, r.warped_moving);

    // --- Warp the planned dose onto the daily anatomy ---------------------
    // The dose was planned on F; the same (u,v) that maps M->F maps the dose to
    // the daily frame (backward-warp the dose image).
    warp_image_cpu(c.dose, c.nx, c.ny, r.u, r.v, r.warped_dose);

    // --- Plan-approval metrics -------------------------------------------
    compute_metrics(c, r);
}
