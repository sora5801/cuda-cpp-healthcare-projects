// ===========================================================================
// src/reference_cpu.cpp  --  The plain-C++ baseline we trust
// ---------------------------------------------------------------------------
// Project 2.4 : Cryo-ET Subtomogram Averaging  (reduced-scope teaching version)
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. Written to be
//   OBVIOUSLY correct -- plain loops, no parallelism, no FFT cleverness -- so
//   that when the GPU's FFT-based cross-correlation agrees with this direct
//   spatial-domain computation, we believe the GPU. Compiled by the host C++
//   compiler only (no CUDA here). See reference_cpu.h for the contracts.
//
//   The KEY teaching pairing: this file computes cross-correlation the DEFINITION
//   way (a sum of products, O(V) per shift); kernels.cu computes the SAME thing
//   for ALL shifts at once in O(V log V) using cuFFT and the cross-correlation
//   theorem. main.cu checks they agree at zero shift.
//
// READ THIS AFTER: reference_cpu.h. Compare against kernels.cu (the GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>       // std::cos, std::sin, std::sqrt, std::floor, std::fabs
#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// trial_angle: the k-th angle of the discrete in-plane rotation search.
//   We sweep a full turn in n_angles equal steps. Defined ONCE here and called
//   from both the host reference and (recomputed by the same formula) the kernel
//   so the two paths search identical orientations. Units: radians.
// ---------------------------------------------------------------------------
double trial_angle(int k, int n_angles) {
    return 2.0 * M_PI * static_cast<double>(k) / static_cast<double>(n_angles);
}

// ---------------------------------------------------------------------------
// load_subtomograms: parse the simple whitespace text format (data/README.md).
//   Layout:  n_sub d n_angles | ref[d^3] | cand_0[d^3] ... cand_{n_sub-1}[d^3].
//   Every cube is zero-meaned on load so correlation peaks are meaningful.
// ---------------------------------------------------------------------------
SubtomogramSet load_subtomograms(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open subtomogram file: " + path);

    SubtomogramSet s;
    if (!(in >> s.n_sub >> s.d >> s.n_angles) ||
        s.n_sub <= 0 || s.d <= 0 || s.n_angles <= 0) {
        throw std::runtime_error("bad header (expected 'n_sub d n_angles') in " + path);
    }
    const int V = s.vol();   // voxels per cube = d*d*d

    // Read the reference cube, then all candidate cubes, one float at a time.
    s.ref.resize(static_cast<std::size_t>(V));
    for (int i = 0; i < V; ++i)
        if (!(in >> s.ref[static_cast<std::size_t>(i)]))
            throw std::runtime_error("reference cube truncated in " + path);

    s.cand.resize(static_cast<std::size_t>(s.n_sub) * V);
    for (std::size_t i = 0; i < s.cand.size(); ++i)
        if (!(in >> s.cand[i]))
            throw std::runtime_error("candidate cubes truncated in " + path);

    // Zero-mean every cube (see normalize_zero_mean's rationale).
    normalize_zero_mean(s.ref, V);
    for (int c = 0; c < s.n_sub; ++c) {
        // Copy one candidate out, normalize, copy back. A small extra vector
        // keeps normalize_zero_mean's signature simple and reusable.
        std::vector<float> tmp(s.cand.begin() + static_cast<std::ptrdiff_t>(c) * V,
                               s.cand.begin() + static_cast<std::ptrdiff_t>(c + 1) * V);
        normalize_zero_mean(tmp, V);
        for (int i = 0; i < V; ++i)
            s.cand[static_cast<std::size_t>(c) * V + i] = tmp[static_cast<std::size_t>(i)];
    }
    return s;
}

// ---------------------------------------------------------------------------
// normalize_zero_mean: subtract the arithmetic mean in place.
//   A cross-correlation sum(a.*b) is dominated by the product of the two means
//   unless both signals are centered; centering turns correlation into the
//   covariance, whose normalized form (NCC) peaks at +1 for identical shapes.
// ---------------------------------------------------------------------------
void normalize_zero_mean(std::vector<float>& cube, int vol) {
    double mean = 0.0;
    for (int i = 0; i < vol; ++i) mean += cube[static_cast<std::size_t>(i)];
    mean /= static_cast<double>(vol);
    for (int i = 0; i < vol; ++i)
        cube[static_cast<std::size_t>(i)] -= static_cast<float>(mean);
}

// ---------------------------------------------------------------------------
// rotate_cube_cpu: in-plane rotation about z with bilinear interpolation.
//
//   For each OUTPUT voxel (x,y,z) we ask "where in the INPUT did this density
//   come from?". Rotating the output coordinate by -theta about the cube center
//   gives the source point (sx, sy); we read the input there with bilinear
//   interpolation (a weighted blend of the 4 surrounding voxels). This
//   "backward mapping" guarantees every output voxel is filled exactly once
//   (a forward map would leave holes). z is untouched -- this is a 2-D rotation
//   applied to every z-slice. The GPU rotate_kernel uses identical arithmetic.
// ---------------------------------------------------------------------------
void rotate_cube_cpu(const float* in, float* out, int d, double theta) {
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    const double center = 0.5 * (static_cast<double>(d) - 1.0);   // cube center

    for (int z = 0; z < d; ++z) {
        for (int y = 0; y < d; ++y) {
            for (int x = 0; x < d; ++x) {
                // Offset of this output voxel from the center.
                const double ox = static_cast<double>(x) - center;
                const double oy = static_cast<double>(y) - center;
                // Inverse rotation (-theta): source offset in the input cube.
                // Rotating by +theta CCW means the source is at angle -theta.
                const double sx = center + (c * ox + s * oy);
                const double sy = center + (-s * ox + c * oy);

                // Bilinear interpolation at (sx, sy). Voxels outside read 0.
                float val = 0.0f;
                const int x0 = static_cast<int>(std::floor(sx));
                const int y0 = static_cast<int>(std::floor(sy));
                const double fx = sx - static_cast<double>(x0);  // frac in [0,1)
                const double fy = sy - static_cast<double>(y0);
                // Accumulate the 4 corner contributions that lie in bounds.
                for (int dy = 0; dy <= 1; ++dy) {
                    for (int dx = 0; dx <= 1; ++dx) {
                        const int xx = x0 + dx;
                        const int yy = y0 + dy;
                        if (xx < 0 || xx >= d || yy < 0 || yy >= d) continue;
                        const double wx = dx ? fx : (1.0 - fx);
                        const double wy = dy ? fy : (1.0 - fy);
                        val += static_cast<float>(wx * wy) *
                               in[(static_cast<std::size_t>(z) * d + yy) * d + xx];
                    }
                }
                out[(static_cast<std::size_t>(z) * d + y) * d + x] = val;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ncc_zero_shift_one: normalized cross-correlation of two equal cubes at shift 0.
//   = sum(a.*b) / sqrt(sum(a^2) * sum(b^2)). Both inputs are zero-mean, so this
//   is exactly the Pearson correlation coefficient in [-1, +1]. Returns 0 if
//   either cube has no variance (avoids 0/0).
// ---------------------------------------------------------------------------
static double ncc_zero_shift_one(const float* a, const float* b, int vol) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < vol; ++i) {
        const double av = a[static_cast<std::size_t>(i)];
        const double bv = b[static_cast<std::size_t>(i)];
        dot += av * bv;     // numerator: the raw correlation at zero shift
        na  += av * av;     // energy of a (for normalization)
        nb  += bv * bv;     // energy of b
    }
    const double denom = std::sqrt(na * nb);
    return denom > 0.0 ? dot / denom : 0.0;
}

// ---------------------------------------------------------------------------
// correlate_cpu: the full per-candidate, per-angle NCC table + best angle.
//   Complexity: O(n_sub * n_angles * V) where V = d^3 (one rotate + one dot per
//   (candidate, angle)). Transparent and slow -- exactly what a reference is for.
// ---------------------------------------------------------------------------
void correlate_cpu(const SubtomogramSet& set,
                   std::vector<double>& ncc_zero_shift,
                   std::vector<int>& best_angle) {
    const int V = set.vol();
    ncc_zero_shift.assign(static_cast<std::size_t>(set.n_sub) * set.n_angles, 0.0);
    best_angle.assign(static_cast<std::size_t>(set.n_sub), 0);

    std::vector<float> rot(static_cast<std::size_t>(V));   // scratch rotated cube
    for (int s = 0; s < set.n_sub; ++s) {
        const float* cand = &set.cand[static_cast<std::size_t>(s) * V];
        double best = -2.0;   // below the [-1,1] range so any real NCC wins
        int    best_k = 0;
        for (int k = 0; k < set.n_angles; ++k) {
            rotate_cube_cpu(cand, rot.data(), set.d, trial_angle(k, set.n_angles));
            const double v = ncc_zero_shift_one(set.ref.data(), rot.data(), V);
            ncc_zero_shift[static_cast<std::size_t>(s) * set.n_angles + k] = v;
            // Strictly-greater keeps the FIRST (lowest-k) angle on ties, which
            // makes best_angle deterministic (matches the GPU's tie rule).
            if (v > best) { best = v; best_k = k; }
        }
        best_angle[static_cast<std::size_t>(s)] = best_k;
    }
}

// ---------------------------------------------------------------------------
// build_average_cpu: rotate each candidate to its chosen pose and average.
//   The averaged cube is the refined reference; we report mean(|voxel|) as a
//   single deterministic scalar so stdout has a stable, meaningful number that
//   rises as the (now aligned) signals add coherently.
// ---------------------------------------------------------------------------
double build_average_cpu(const SubtomogramSet& set,
                         const std::vector<int>& best_angle,
                         std::vector<float>& out_avg) {
    const int V = set.vol();
    out_avg.assign(static_cast<std::size_t>(V), 0.0f);

    std::vector<float> rot(static_cast<std::size_t>(V));
    for (int s = 0; s < set.n_sub; ++s) {
        const float* cand = &set.cand[static_cast<std::size_t>(s) * V];
        rotate_cube_cpu(cand, rot.data(), set.d,
                        trial_angle(best_angle[static_cast<std::size_t>(s)], set.n_angles));
        for (int i = 0; i < V; ++i) out_avg[static_cast<std::size_t>(i)] += rot[static_cast<std::size_t>(i)];
    }
    const double inv_n = 1.0 / static_cast<double>(set.n_sub);
    double mean_abs = 0.0;
    for (int i = 0; i < V; ++i) {
        out_avg[static_cast<std::size_t>(i)] *= static_cast<float>(inv_n);
        mean_abs += std::fabs(out_avg[static_cast<std::size_t>(i)]);
    }
    return mean_abs / static_cast<double>(V);
}
