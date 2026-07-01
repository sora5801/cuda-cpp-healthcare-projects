// ===========================================================================
// src/reference_cpu.cpp  --  Loader, projectors, serial SIRT (the trusted answer)
// ---------------------------------------------------------------------------
// Project 4.2 : Iterative / Model-Based CT Reconstruction
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- readable loops, no parallelism, no cleverness --
//   so that when the GPU and CPU agree we believe the GPU. Every projector here
//   calls the SAME per-ray geometry (ct_geometry.h) that the GPU kernels call,
//   so the two run bit-compatible math (PATTERNS.md §2).
//
//   Compiled by the host C++ compiler only (no CUDA). See reference_cpu.h.
//
// READ THIS AFTER: ct_geometry.h, reference_cpu.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"
#include "ct_geometry.h"     // detector_coord, interp_stencil, pixel_world (shared)

#include <cmath>
#include <fstream>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// load_ct: parse the text sinogram + geometry (format in data/README.md).
//   The header carries everything the reconstruction needs, INCLUDING the
//   iteration budget (iters), step size (lambda) and TV weight, so a learner can
//   change reconstruction behaviour by editing the data file -- no recompile.
// ---------------------------------------------------------------------------
CTProblem load_ct(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open sinogram file: " + path);
    CTProblem ct;
    int has_truth = 0;
    // Header fields, in order. If any is missing the stream fails and we throw.
    if (!(in >> ct.n_angles >> ct.n_det >> ct.ds >> ct.img >> ct.world_half
             >> ct.iters >> ct.lambda >> ct.tv_weight >> has_truth))
        throw std::runtime_error("bad header (need: n_angles n_det ds img world_half "
                                 "iters lambda tv_weight has_truth) in " + path);
    if (ct.n_angles <= 0 || ct.n_det <= 0 || ct.img <= 0 || ct.iters <= 0)
        throw std::runtime_error("non-positive geometry/iters in " + path);

    // Read the measured sinogram: n_angles rows x n_det columns, row-major.
    ct.sino.resize(static_cast<std::size_t>(ct.n_angles) * ct.n_det);
    for (std::size_t k = 0; k < ct.sino.size(); ++k)
        if (!(in >> ct.sino[k]))
            throw std::runtime_error("sinogram truncated in " + path);

    // Read the optional ground-truth image (img x img) if the flag says so.
    if (has_truth) {
        ct.truth.resize(static_cast<std::size_t>(ct.img) * ct.img);
        for (std::size_t p = 0; p < ct.truth.size(); ++p)
            if (!(in >> ct.truth[p]))
                throw std::runtime_error("truth image truncated in " + path);
    }
    return ct;
}

// ---------------------------------------------------------------------------
// compute_trig: cos/sin of every projection angle, computed ONCE in double then
//   stored as float. Both projectors read these, so CPU and GPU share the exact
//   same trig values (no cos-vs-cosf disagreement to accumulate over iterations).
// ---------------------------------------------------------------------------
void compute_trig(int n_angles, std::vector<float>& cosv, std::vector<float>& sinv) {
    cosv.resize(n_angles);
    sinv.resize(n_angles);
    for (int k = 0; k < n_angles; ++k) {
        const double theta = M_PI * k / n_angles;   // 180-degree parallel scan
        cosv[k] = static_cast<float>(std::cos(theta));
        sinv[k] = static_cast<float>(std::sin(theta));
    }
}

// ---------------------------------------------------------------------------
// The VOXEL-DRIVEN projector pair. Both operators loop over (pixel, angle) and
// use the shared interp_stencil, which guarantees FORWARD (A) and BACK (A^T) are
// exact transposes -- the property SIRT relies on to converge.
//
//   FORWARD  A : sino[k,j] += img[p] * weight     (scatter pixel -> 2 det bins)
//   BACK     A^T: img[p]   += sino[k, interp]     (gather 2 det bins -> pixel)
//
// Because forward SCATTERS with += into shared detector bins, the serial CPU
// version simply adds; the GPU forward kernel will need atomics or a per-ray
// formulation (see kernels.cu) to get the identical sum.
// ---------------------------------------------------------------------------

// Convenience: derive the per-image scalars used by every projector call.
namespace {
struct Geom {
    int   N, n_det, n_angles;
    float ds, W, pix, center;
};
Geom make_geom(const CTProblem& ct) {
    Geom g;
    g.N        = ct.img;
    g.n_det    = ct.n_det;
    g.n_angles = ct.n_angles;
    g.ds       = ct.ds;
    g.W        = ct.world_half;
    g.pix      = (g.N > 1) ? (2.0f * g.W / (g.N - 1)) : 0.0f;  // world units/pixel
    g.center   = 0.5f * (g.n_det - 1);                         // detector index of s=0
    return g;
}
}  // namespace

void forward_project_cpu(const CTProblem& ct, const std::vector<float>& image,
                         const std::vector<float>& cosv, const std::vector<float>& sinv,
                         std::vector<float>& sino_out) {
    const Geom g = make_geom(ct);
    sino_out.assign(static_cast<std::size_t>(g.n_angles) * g.n_det, 0.0f);

    // For each pixel, scatter its value into the detector bins it hits at every
    // angle. This is the transpose of backproject_cpu below (same stencil).
    for (int py = 0; py < g.N; ++py) {
        const float wy = pixel_world(py, g.W, g.pix);
        for (int px = 0; px < g.N; ++px) {
            const float wx  = pixel_world(px, g.W, g.pix);
            const float val = image[static_cast<std::size_t>(py) * g.N + px];
            if (val == 0.0f) continue;                 // nothing to scatter
            for (int k = 0; k < g.n_angles; ++k) {
                const float fidx =
                    detector_coord(wx, wy, cosv[k], sinv[k], g.ds, g.center);
                int j0; float w;
                if (interp_stencil(fidx, g.n_det, &j0, &w)) {
                    float* row = &sino_out[static_cast<std::size_t>(k) * g.n_det];
                    row[j0]     += val * (1.0f - w);   // lower bin
                    row[j0 + 1] += val * w;            // upper bin
                }
            }
        }
    }
}

void backproject_cpu(const CTProblem& ct, const std::vector<float>& sino,
                     const std::vector<float>& cosv, const std::vector<float>& sinv,
                     std::vector<float>& image_out) {
    const Geom g = make_geom(ct);
    image_out.assign(static_cast<std::size_t>(g.N) * g.N, 0.0f);

    // For each pixel, gather the sinogram sampled where its ray hits the detector
    // at each angle. Each pixel is independent -> the natural GPU mapping.
    for (int py = 0; py < g.N; ++py) {
        const float wy = pixel_world(py, g.W, g.pix);
        for (int px = 0; px < g.N; ++px) {
            const float wx = pixel_world(px, g.W, g.pix);
            float acc = 0.0f;
            for (int k = 0; k < g.n_angles; ++k) {
                const float fidx =
                    detector_coord(wx, wy, cosv[k], sinv[k], g.ds, g.center);
                int j0; float w;
                if (interp_stencil(fidx, g.n_det, &j0, &w)) {
                    const float* row = &sino[static_cast<std::size_t>(k) * g.n_det];
                    acc += row[j0] * (1.0f - w) + row[j0 + 1] * w;
                }
            }
            image_out[static_cast<std::size_t>(py) * g.N + px] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// compute_sirt_weights: the SIRT preconditioners R and C.
//   row_scale[ray]  = 1 / (A applied to an all-ones image)      -> per detector bin
//   col_scale[pixel]= 1 / (A^T applied to an all-ones sinogram) -> per pixel
//   (with 0 where the denominator is 0, so unused rays/pixels are inert).
//   These normalize the update so SIRT is a contraction: a residual in a bin hit
//   by many pixels, and a pixel touched by many rays, are damped accordingly.
// ---------------------------------------------------------------------------
void compute_sirt_weights(const CTProblem& ct,
                          const std::vector<float>& cosv, const std::vector<float>& sinv,
                          std::vector<float>& row_scale, std::vector<float>& col_scale) {
    const Geom g = make_geom(ct);
    const std::size_t n_rays = static_cast<std::size_t>(g.n_angles) * g.n_det;
    const std::size_t n_pix  = static_cast<std::size_t>(g.N) * g.N;

    // Row sums: forward-project an all-ones image; each bin gets the total weight
    // of pixels projecting into it.
    std::vector<float> ones_img(n_pix, 1.0f), row_sum;
    forward_project_cpu(ct, ones_img, cosv, sinv, row_sum);
    row_scale.assign(n_rays, 0.0f);
    for (std::size_t i = 0; i < n_rays; ++i)
        row_scale[i] = (row_sum[i] > 1e-8f) ? (1.0f / row_sum[i]) : 0.0f;

    // Column sums: backproject an all-ones sinogram; each pixel gets the total
    // weight of rays passing through it.
    std::vector<float> ones_sino(n_rays, 1.0f), col_sum;
    backproject_cpu(ct, ones_sino, cosv, sinv, col_sum);
    col_scale.assign(n_pix, 0.0f);
    for (std::size_t p = 0; p < n_pix; ++p)
        col_scale[p] = (col_sum[p] > 1e-8f) ? (1.0f / col_sum[p]) : 0.0f;
}

// ---------------------------------------------------------------------------
// tv_step_cpu (file-local): one gradient-descent step on the isotropic TOTAL
//   VARIATION of the image. TV = sum of |grad(image)|; its gradient pushes each
//   pixel toward the local mean of its 4 neighbours UNLESS there is a strong
//   edge (large gradient magnitude), which the 1/sqrt(eps^2+|grad|^2) weighting
//   protects. This is the "prior" that removes streaks/noise while keeping edges
//   -- the reason model-based reconstruction beats FBP at low dose.
//   We use an explicit, tiny step (weight = ct.tv_weight) so the CPU and GPU do
//   the identical update. eps avoids divide-by-zero on flat regions.
// ---------------------------------------------------------------------------
namespace {
void tv_step_cpu(int N, float weight, std::vector<float>& img) {
    if (weight <= 0.0f) return;
    const float eps = 1e-3f;
    std::vector<float> upd(img.size(), 0.0f);
    // Neighbour helper with clamped (Neumann) boundaries: edges reuse themselves,
    // so the image is treated as if it extends flat past the border.
    auto at = [&](int x, int y) -> float {
        if (x < 0) x = 0; if (x >= N) x = N - 1;
        if (y < 0) y = 0; if (y >= N) y = N - 1;
        return img[static_cast<std::size_t>(y) * N + x];
    };
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            const float c  = at(x, y);
            const float dl = c - at(x - 1, y);   // differences to 4 neighbours
            const float dr = at(x + 1, y) - c;
            const float du = c - at(x, y - 1);
            const float dd = at(x, y + 1) - c;
            // Edge-preserving weight: small where the local gradient is large.
            const float gmag = std::sqrt(eps * eps + dl * dl + dr * dr + du * du + dd * dd);
            const float lap  = (dr - dl) + (dd - du);   // discrete Laplacian
            // Move the pixel a little toward its neighbours, scaled by 1/gmag so
            // strong edges barely move. This is one explicit TV-descent step.
            upd[static_cast<std::size_t>(y) * N + x] = c + weight * lap / gmag;
        }
    }
    img.swap(upd);
}
}  // namespace

// ---------------------------------------------------------------------------
// reconstruct_sirt_cpu: the full serial reconstruction loop.
//   x^0 = 0
//   repeat iters times:
//       r    = b - A x                       (residual in sinogram space)
//       g    = A^T (R .* r)                  (backprojected, row-normalized)
//       x   += lambda * (C .* g)             (column-normalized SIRT update)
//       x    = max(x, 0)                     (non-negativity: densities >= 0)
//       x    = TV_step(x)                    (optional edge-preserving smoothing)
//   Complexity: O(iters * n_angles * N^2). This is the reference SIRT the GPU
//   driver (sirt_gpu in kernels.cu) mirrors step for step.
// ---------------------------------------------------------------------------
void reconstruct_sirt_cpu(const CTProblem& ct,
                          const std::vector<float>& cosv, const std::vector<float>& sinv,
                          const std::vector<float>& row_scale, const std::vector<float>& col_scale,
                          std::vector<float>& image) {
    const Geom g = make_geom(ct);
    const std::size_t n_rays = static_cast<std::size_t>(g.n_angles) * g.n_det;
    const std::size_t n_pix  = static_cast<std::size_t>(g.N) * g.N;

    image.assign(n_pix, 0.0f);                 // x^0 = blank image
    std::vector<float> sim(n_rays), resid(n_rays), grad(n_pix);

    for (int it = 0; it < ct.iters; ++it) {
        // 1. Forward project the current estimate: sim = A x.
        forward_project_cpu(ct, image, cosv, sinv, sim);
        // 2. Row-normalized residual: resid = R .* (b - A x).
        for (std::size_t i = 0; i < n_rays; ++i)
            resid[i] = (ct.sino[i] - sim[i]) * row_scale[i];
        // 3. Backproject the residual: grad = A^T resid.
        backproject_cpu(ct, resid, cosv, sinv, grad);
        // 4. Column-normalized, relaxed update + non-negativity.
        for (std::size_t p = 0; p < n_pix; ++p) {
            float v = image[p] + ct.lambda * col_scale[p] * grad[p];
            image[p] = (v > 0.0f) ? v : 0.0f;  // X-ray attenuation is non-negative
        }
        // 5. Optional TV smoothing step (the model-based prior).
        tv_step_cpu(g.N, ct.tv_weight, image);
    }
}

// ---------------------------------------------------------------------------
// rms_error: sqrt(mean((a-b)^2)) in double precision. A single scalar quality
//   number; used to compare a reconstruction against the ground truth.
// ---------------------------------------------------------------------------
double rms_error(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return -1.0;
    double acc = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        acc += d * d;
    }
    return std::sqrt(acc / static_cast<double>(a.size()));
}
