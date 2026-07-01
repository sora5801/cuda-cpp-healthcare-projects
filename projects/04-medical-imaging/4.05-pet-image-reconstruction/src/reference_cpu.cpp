// ===========================================================================
// src/reference_cpu.cpp  --  Loader, projectors, and serial MLEM (the baseline)
// ---------------------------------------------------------------------------
// Project 4.5 : PET Image Reconstruction (MLEM / OS-EM)
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- plain nested loops, no parallelism, no cleverness
//   -- so that when the GPU and CPU agree, we believe the GPU. Every arithmetic
//   step that also runs on the GPU calls the SAME shared helper from
//   pet_geometry.h, so the two implementations are the same math by construction.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h + pet_geometry.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>       // std::cos, std::sin
#include <cstddef>     // std::size_t
#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// load_pet: parse the tiny text sinogram in data/sample/ (format in data/README).
//   header: "<K> <D> <ds> <N> <W> <iters>", then K*D count values, row-major.
//   We validate every field so a malformed file aborts loudly instead of
//   silently reconstructing garbage.
// ---------------------------------------------------------------------------
PetProblem load_pet(const std::string& path, int& iters_out) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open sinogram file: " + path);

    int   K = 0, D = 0, N = 0, iters = 0;
    float ds = 0.0f, W = 0.0f;
    if (!(in >> K >> D >> ds >> N >> W >> iters))
        throw std::runtime_error("bad header (expected 'K D ds N W iters') in " + path);
    if (K <= 0 || D <= 0 || N <= 0 || ds <= 0.0f || W <= 0.0f)
        throw std::runtime_error("non-positive geometry in " + path);

    PetProblem p;
    p.geom = make_geom(N, K, D, ds, W);   // fills derived pix/center once
    iters_out = iters;

    // Read the K*D measured counts. Counts are physically non-negative; we clamp
    // a stray tiny-negative (e.g. from a hand-edited file) to 0 rather than let a
    // negative count poison the Poisson model.
    p.counts.resize(static_cast<std::size_t>(K) * D);
    for (std::size_t i = 0; i < p.counts.size(); ++i) {
        if (!(in >> p.counts[i]))
            throw std::runtime_error("sinogram truncated in " + path);
        if (p.counts[i] < 0.0f) p.counts[i] = 0.0f;
    }
    return p;
}

// ---------------------------------------------------------------------------
// compute_trig: theta_k = k*pi/K spans 180 degrees (parallel-beam rebinning).
//   Computed in double then stored as float so BOTH projectors -- and the GPU --
//   read the identical trig values (kernels.cu uploads these same arrays).
// ---------------------------------------------------------------------------
void compute_trig(PetProblem& p) {
    const int K = p.geom.K;
    p.cosv.resize(K);
    p.sinv.resize(K);
    for (int k = 0; k < K; ++k) {
        const double theta = M_PI * k / K;
        p.cosv[k] = static_cast<float>(std::cos(theta));
        p.sinv[k] = static_cast<float>(std::sin(theta));
    }
}

// ---------------------------------------------------------------------------
// forward_project_cpu: y_hat = A x. One PIXEL-DRIVEN pass builds the sinogram.
//   For every pixel we find, at every angle, the fractional detector bin it hits
//   and SPLIT its value between the two nearest bins (linear interpolation). We
//   accumulate into a double buffer per LOR to keep the sum well-conditioned,
//   then store as float. Pixel-driven forward + pixel-driven back (below) using
//   the SAME split are exact transposes -- what MLEM requires.
//   Complexity: O(N^2 * K) -- every pixel touches every angle.
// ---------------------------------------------------------------------------
void forward_project_cpu(const PetProblem& p, const std::vector<float>& image,
                         std::vector<float>& sino) {
    const PetGeom& g = p.geom;
    const std::size_t n_lor = static_cast<std::size_t>(g.K) * g.D;

    // Accumulate in double for a faithful sum, then narrow to float.
    std::vector<double> acc(n_lor, 0.0);

    for (int py = 0; py < g.N; ++py) {
        const float wy = pixel_world_y(g, py);
        for (int px = 0; px < g.N; ++px) {
            const float wx = pixel_world_x(g, px);
            const float xv = image[static_cast<std::size_t>(py) * g.N + px];
            if (xv == 0.0f) continue;           // empty pixel contributes nothing
            for (int k = 0; k < g.K; ++k) {
                const float fidx = detector_fidx(g, wx, wy, p.cosv[k], p.sinv[k]);
                int j0; float w;
                if (!split_bin(g, fidx, j0, w)) continue;  // ray leaves the detector
                const std::size_t base = static_cast<std::size_t>(k) * g.D + j0;
                acc[base]     += static_cast<double>(xv) * (1.0 - w);  // lower bin
                acc[base + 1] += static_cast<double>(xv) * w;          // upper bin
            }
        }
    }
    sino.resize(n_lor);
    for (std::size_t i = 0; i < n_lor; ++i) sino[i] = static_cast<float>(acc[i]);
}

// ---------------------------------------------------------------------------
// backproject_cpu: img = A^T r. One PIXEL-DRIVEN pass sums, for each pixel, the
//   interpolated sinogram value on every LOR through it -- the exact transpose of
//   forward_project_cpu (same split weights). This is a per-pixel GATHER: pixel
//   outputs are independent, which is why the GPU twin needs no atomics.
//   Complexity: O(N^2 * K).
// ---------------------------------------------------------------------------
void backproject_cpu(const PetProblem& p, const std::vector<float>& sino,
                     std::vector<float>& image) {
    const PetGeom& g = p.geom;
    image.assign(static_cast<std::size_t>(g.N) * g.N, 0.0f);

    for (int py = 0; py < g.N; ++py) {
        const float wy = pixel_world_y(g, py);
        for (int px = 0; px < g.N; ++px) {
            const float wx = pixel_world_x(g, px);
            double acc = 0.0;                    // double accumulator for the gather
            for (int k = 0; k < g.K; ++k) {
                const float fidx = detector_fidx(g, wx, wy, p.cosv[k], p.sinv[k]);
                int j0; float w;
                if (!split_bin(g, fidx, j0, w)) continue;
                const std::size_t base = static_cast<std::size_t>(k) * g.D + j0;
                // Transpose of the forward split: read both neighbor bins with the
                // same 1-w / w weights we WROTE with in forward projection.
                acc += static_cast<double>(sino[base])     * (1.0 - w)
                     + static_cast<double>(sino[base + 1]) * w;
            }
            image[static_cast<std::size_t>(py) * g.N + px] = static_cast<float>(acc);
        }
    }
}

// ---------------------------------------------------------------------------
// sensitivity_cpu: s_j = A^T 1. Back-project a sinogram of all ones -> for each
//   pixel, how much total detection weight it accrues across all LORs. This is
//   the per-pixel denominator in the MLEM update; a pixel that no LOR sees would
//   get s_j = 0 and is frozen (we guard the divide).
// ---------------------------------------------------------------------------
void sensitivity_cpu(const PetProblem& p, std::vector<float>& sens) {
    const std::vector<float> ones(static_cast<std::size_t>(p.geom.K) * p.geom.D, 1.0f);
    backproject_cpu(p, ones, sens);
}

// ---------------------------------------------------------------------------
// mlem_cpu: the serial reference solver. Starts from a UNIFORM POSITIVE image
//   (MLEM must start positive -- the multiplicative update can never resurrect a
//   zero pixel) and applies the Shepp-Vardi update `iters` times.
//
//   Per iteration:
//     1. y_hat = A x                         (forward_project_cpu)
//     2. ratio_i = y_i / y_hat_i             (0 where y_hat_i == 0, guarded)
//     3. corr = A^T ratio                    (backproject_cpu)
//     4. x_j <- x_j * corr_j / s_j           (multiplicative update, s_j guarded)
//
//   All float state; the update order here is mirrored EXACTLY by the GPU
//   update_kernel so the two reconstructions track step for step.
// ---------------------------------------------------------------------------
void mlem_cpu(const PetProblem& p, const std::vector<float>& sens, int iters,
              std::vector<float>& image) {
    const PetGeom& g = p.geom;
    const std::size_t n_pix = static_cast<std::size_t>(g.N) * g.N;
    const std::size_t n_lor = static_cast<std::size_t>(g.K) * g.D;

    image.assign(n_pix, 1.0f);   // uniform positive initial estimate x^0 = 1

    std::vector<float> yhat(n_lor), ratio(n_lor), corr(n_pix);
    for (int it = 0; it < iters; ++it) {
        // (1) expected counts under the current estimate
        forward_project_cpu(p, image, yhat);
        // (2) measured / expected, guarded against divide-by-zero
        for (std::size_t i = 0; i < n_lor; ++i)
            ratio[i] = (yhat[i] > 0.0f) ? (p.counts[i] / yhat[i]) : 0.0f;
        // (3) back-project the ratio
        backproject_cpu(p, ratio, corr);
        // (4) multiplicative update, normalized by sensitivity
        for (std::size_t j = 0; j < n_pix; ++j) {
            const float s = sens[j];
            image[j] = (s > 0.0f) ? (image[j] * corr[j] / s) : image[j];
        }
    }
}
