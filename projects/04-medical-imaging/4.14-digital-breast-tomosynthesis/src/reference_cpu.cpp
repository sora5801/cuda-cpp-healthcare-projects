// ===========================================================================
// src/reference_cpu.cpp  --  Loader, angle table, and serial SART reference
// ---------------------------------------------------------------------------
// Project 4.14 : Digital Breast Tomosynthesis
//
// This translation unit is compiled by the plain host C++ compiler only (it is
// listed as <ClCompile> in the .vcxproj, not <CudaCompile>). It provides the
// trusted, readable, SERIAL implementation of the whole SART pipeline. The GPU
// kernels in kernels.cu mirror the per-ray/per-pixel math *exactly* by including
// the same dbt_geometry.h helpers, so main.cu can VERIFY the GPU output against
// what this file produces.
//
// SART in this file is split into the same three stages the GPU uses, so the two
// implementations line up one-to-one:
//   forward_project_cpu()   : image estimate -> simulated projections   (per ray)
//   backproject_update_cpu(): residual       -> per-pixel correction     (per pixel)
//   reconstruct_sart_cpu()  : loops the two stages n_iters times
//
// See reference_cpu.h for the contract and dbt_geometry.h for the shared math.
// ===========================================================================
#include "reference_cpu.h"
#include "dbt_geometry.h"     // forward_ray_integral, bilinear_sample (shared HD core)

#include <cmath>              // std::cos, std::sin, std::floor, std::sqrt
#include <fstream>            // std::ifstream
#include <stdexcept>          // std::runtime_error

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// load_dbt: parse the text problem file (format documented in data/README.md).
// We validate aggressively -- a truncated or nonsensical file should fail here,
// loudly, rather than silently reconstructing noise.
// ---------------------------------------------------------------------------
DBTProblem load_dbt(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open DBT problem file: " + path);

    DBTProblem p;
    // Header: 8 scalars describing geometry + the SART schedule.
    if (!(in >> p.n_angles >> p.n_det >> p.ds >> p.img
             >> p.world_half >> p.half_span >> p.relax >> p.n_iters)) {
        throw std::runtime_error(
            "bad header (expect: n_angles n_det ds img world_half half_span relax n_iters) in " + path);
    }
    if (p.n_angles <= 0 || p.n_det <= 0 || p.img <= 0 || p.n_iters <= 0)
        throw std::runtime_error("non-positive geometry/iteration count in " + path);
    if (p.ds <= 0.0f || p.world_half <= 0.0f || p.half_span <= 0.0f)
        throw std::runtime_error("non-positive spacing/extent/span in " + path);

    // Body: n_angles rows of n_det measured line integrals.
    p.proj.resize(static_cast<std::size_t>(p.n_angles) * p.n_det);
    for (std::size_t k = 0; k < p.proj.size(); ++k) {
        if (!(in >> p.proj[k]))
            throw std::runtime_error("projection data truncated in " + path);
    }
    return p;
}

// ---------------------------------------------------------------------------
// compute_angles: build the cos/sin table for the NARROW DBT wedge.
//   theta_k = -half_span + k * (2*half_span / (n_angles-1)),  k = 0..n_angles-1
// so the angles are symmetric about 0 (straight-down view) and span the wedge.
// Computed once in double precision, then stored as float -> CPU and GPU read
// the SAME float table, guaranteeing identical trig on both sides.
// ---------------------------------------------------------------------------
void compute_angles(const DBTProblem& p, std::vector<float>& cosv, std::vector<float>& sinv) {
    cosv.resize(p.n_angles);
    sinv.resize(p.n_angles);
    const double step = (p.n_angles > 1)
                      ? (2.0 * p.half_span / (p.n_angles - 1))
                      : 0.0;
    for (int k = 0; k < p.n_angles; ++k) {
        const double theta = -static_cast<double>(p.half_span) + k * step;
        cosv[k] = static_cast<float>(std::cos(theta));
        sinv[k] = static_cast<float>(std::sin(theta));
    }
}

// ---------------------------------------------------------------------------
// n_ray_steps: sample each ray at ~2 samples per pixel across the image, so the
// forward integral is well resolved (Nyquist-ish). Shared by CPU and GPU.
// ---------------------------------------------------------------------------
int n_ray_steps(const DBTProblem& p) {
    return 2 * p.img;   // e.g. img=64 -> 128 samples along each ray
}

// ---------------------------------------------------------------------------
// forward_project_cpu: image estimate -> simulated projections.
//   For every angle k and detector bin j, integrate the current image along the
//   corresponding ray (the shared forward_ray_integral()) and scale by the
//   per-step world length dt so the result is a physical line integral (matching
//   the units of the measured data). One independent output per ray -> this is
//   the GPU forward kernel's serial twin.
//
//   sim  : sized to n_angles*n_det, filled with the simulated projections.
// ---------------------------------------------------------------------------
static void forward_project_cpu(const DBTProblem& p,
                                const std::vector<float>& image,
                                const std::vector<float>& cosv,
                                const std::vector<float>& sinv,
                                std::vector<float>& sim) {
    const int   N       = p.img;
    const int   n_det   = p.n_det;
    const float W       = p.world_half;
    const float pix     = (N > 1) ? (2.0f * W / (N - 1)) : 0.0f;   // world units / pixel
    const float center  = 0.5f * (n_det - 1);                      // detector index of s=0
    const int   steps   = n_ray_steps(p);
    const float L       = 1.41421356f * W;                         // ray half-length
    const float dt      = (steps > 1) ? (2.0f * L / (steps - 1)) : 0.0f;  // world length/step

    sim.assign(p.proj.size(), 0.0f);
    for (int k = 0; k < p.n_angles; ++k) {
        const float ck = cosv[k], sk = sinv[k];
        for (int j = 0; j < n_det; ++j) {
            const float s = (j - center) * p.ds;                  // signed detector offset
            // Raw sum of samples along the ray * per-step world length = integral.
            const float raw = forward_ray_integral(image.data(), N, ck, sk, s, W, pix, steps);
            sim[static_cast<std::size_t>(k) * n_det + j] = raw * dt;
        }
    }
}

// ---------------------------------------------------------------------------
// backproject_update_cpu: residual -> per-pixel SART correction, applied to the
// image estimate in place.
//
//   For each output PIXEL (px,py) we gather the residual contribution from every
//   angle: find where this pixel's world position projects onto the detector
//   (s = x*cos + y*sin -> fractional bin), linearly interpolate the residual
//   there, and average over all angles. We then add lambda * (that average) to
//   the pixel and clamp to >= 0 (attenuation cannot be negative).
//
//   This "average of interpolated residuals over angles" is the smoothed,
//   normalised backprojection SART uses; dividing by n_angles is the simple,
//   fully-deterministic column normalisation (see THEORY.md, algorithm). Every
//   pixel is independent -> exactly the GPU backprojection kernel's serial twin.
//
//   residual : [n_angles*n_det] = measured - simulated, already scaled.
//   image    : updated in place (the running estimate).
// ---------------------------------------------------------------------------
static void backproject_update_cpu(const DBTProblem& p,
                                   const std::vector<float>& residual,
                                   const std::vector<float>& cosv,
                                   const std::vector<float>& sinv,
                                   std::vector<float>& image) {
    const int   N      = p.img;
    const int   n_det  = p.n_det;
    const float W      = p.world_half;
    const float pix    = (N > 1) ? (2.0f * W / (N - 1)) : 0.0f;
    const float center = 0.5f * (n_det - 1);
    const float inv_na = 1.0f / static_cast<float>(p.n_angles);   // column normalisation
    const float lambda = p.relax;

    for (int py = 0; py < N; ++py) {
        const float wy = -W + py * pix;              // world y of this pixel row
        for (int px = 0; px < N; ++px) {
            const float wx = -W + px * pix;          // world x of this pixel
            float acc = 0.0f;
            // Gather this pixel's residual from every projection angle.
            for (int k = 0; k < p.n_angles; ++k) {
                const float s    = wx * cosv[k] + wy * sinv[k];      // detector offset
                const float fidx = s / p.ds + center;                // fractional bin
                const int   j0   = static_cast<int>(std::floor(fidx));
                if (j0 >= 0 && j0 + 1 < n_det) {
                    const float w   = fidx - j0;                     // interp weight
                    const float* r  = &residual[static_cast<std::size_t>(k) * n_det];
                    acc += r[j0] * (1.0f - w) + r[j0 + 1] * w;
                }
            }
            const std::size_t idx = static_cast<std::size_t>(py) * N + px;
            float v = image[idx] + lambda * acc * inv_na;            // relaxed update
            if (v < 0.0f) v = 0.0f;                                  // physicality: mu >= 0
            image[idx] = v;
        }
    }
}

// ---------------------------------------------------------------------------
// reconstruct_sart_cpu: the full serial SART loop -- the verification baseline.
//   Start from a zero image (all air) and run n_iters sweeps of
//   {forward-project, residual, backproject-update}. The projection values are
//   already scaled to physical line-integral units, so we simulate in the same
//   units and the residual is directly comparable.
// ---------------------------------------------------------------------------
void reconstruct_sart_cpu(const DBTProblem& p,
                          const std::vector<float>& cosv,
                          const std::vector<float>& sinv,
                          std::vector<float>& image) {
    const std::size_t n_pix = static_cast<std::size_t>(p.img) * p.img;
    image.assign(n_pix, 0.0f);                       // initial estimate: empty breast

    std::vector<float> sim(p.proj.size(), 0.0f);     // simulated projections
    std::vector<float> res(p.proj.size(), 0.0f);     // residual = measured - simulated

    for (int it = 0; it < p.n_iters; ++it) {
        // (1) forward-project the current estimate.
        forward_project_cpu(p, image, cosv, sinv, sim);
        // (2) residual: how far the simulated projections are from the measured.
        for (std::size_t m = 0; m < res.size(); ++m)
            res[m] = p.proj[m] - sim[m];
        // (3) backproject the residual and apply the relaxed, clamped update.
        backproject_update_cpu(p, res, cosv, sinv, image);
    }
}
