// ===========================================================================
// src/reference_cpu.cpp  --  The plain-C++ DTI baseline we trust + data loader
// ---------------------------------------------------------------------------
// Project 4.15 : Diffusion MRI & Tractography
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. Written to be
//   OBVIOUSLY correct -- readable serial loops, no cleverness -- so that when GPU
//   and CPU agree, we believe the GPU. It contains four things:
//     (1) load_dwi()            : parse the tiny text dataset (data/README.md).
//     (2) make_gradient_scheme(): the fixed 12-direction acquisition geometry.
//     (3) build_pseudo_inverse(): the once-per-run OLS operator Minv = (B^TB)^-1 B^T.
//     (4) fit_all_voxels_cpu()  : the reference per-voxel tensor fit (calls the
//                                 shared fit_voxel() from dti_core.h).
//     (5) trace_streamlines_cpu(): the reference deterministic tractography.
//
//   Compiled by the host C++ compiler only (no CUDA). See reference_cpu.h. The
//   per-voxel PHYSICS is in dti_core.h and the per-step tractography math is in
//   tract_core.h -- both shared with the GPU so results match bit-for-bit.
//
// READ THIS AFTER: reference_cpu.h, dti_core.h, tract_core.h. Compare against
// kernels.cu (the GPU twins of (4) and (5)).
// ===========================================================================
#include "reference_cpu.h"
#include "tract_core.h"     // sample_dir, nearest_dir (shared streamline stepping)

#include <cmath>
#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// (1) load_dwi -- parse the text DWI volume (format in data/README.md):
//       line 1 : "<nx> <ny> <nz> <nmeas>"   (nmeas MUST equal NMEAS)
//       then nvox blocks, each = 1 mask flag + NMEAS signal values, whitespace
//       separated. Voxel index runs fastest in x, then y, then z (row-major).
//   Throws std::runtime_error on any inconsistency so demos fail loudly.
// ---------------------------------------------------------------------------
DwiVolume load_dwi(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open DWI file: " + path);

    int nx, ny, nz, nmeas;
    if (!(in >> nx >> ny >> nz >> nmeas))
        throw std::runtime_error("bad header (expected '<nx> <ny> <nz> <nmeas>') in " + path);
    if (nmeas != NMEAS)
        throw std::runtime_error("measurement-count mismatch: file has " +
                                 std::to_string(nmeas) + " but this build expects NMEAS=" +
                                 std::to_string(NMEAS));
    if (nx <= 0 || ny <= 0 || nz <= 0)
        throw std::runtime_error("non-positive grid dimension in " + path);

    DwiVolume vol;
    vol.nx = nx; vol.ny = ny; vol.nz = nz;
    vol.nvox = nx * ny * nz;
    vol.mask.resize(vol.nvox);
    vol.signal.resize(static_cast<std::size_t>(vol.nvox) * NMEAS);

    for (int v = 0; v < vol.nvox; ++v) {
        if (!(in >> vol.mask[v]))
            throw std::runtime_error("unexpected end of data (mask) in " + path);
        for (int k = 0; k < NMEAS; ++k) {
            double s;
            if (!(in >> s))
                throw std::runtime_error("unexpected end of data (signal) in " + path);
            vol.signal[static_cast<std::size_t>(v) * NMEAS + k] = s;
        }
    }
    return vol;
}

// ---------------------------------------------------------------------------
// (2) make_gradient_scheme -- the fixed acquisition geometry.
//   Row 0 is the b=0 image (zero gradient). The other 12 rows share one shell
//   b-value (b = 1000 s/mm^2, a common clinical value) and point along a
//   12-direction electrostatic-repulsion (icosahedral) set: the vertices of an
//   icosahedron give a near-uniform, well-conditioned sampling of the sphere,
//   which is what makes the least-squares tensor fit robust. (Real acquisitions
//   use 30-64+ directions; 12 is the minimum for a stable teaching fit.)
// ---------------------------------------------------------------------------
GradientScheme make_gradient_scheme() {
    GradientScheme s;
    s.bval.resize(NMEAS);
    s.gx.resize(NMEAS); s.gy.resize(NMEAS); s.gz.resize(NMEAS);

    // The 12 icosahedron vertices (unit vectors), using the golden ratio phi.
    const double phi = (1.0 + std::sqrt(5.0)) / 2.0;   // ~1.618
    const double inv = 1.0 / std::sqrt(1.0 + phi * phi);   // normalisation
    // Vertices are cyclic permutations of (0, +-1, +-phi), then normalised.
    const double raw[12][3] = {
        { 0,  1,  phi}, { 0,  1, -phi}, { 0, -1,  phi}, { 0, -1, -phi},
        { 1,  phi, 0 }, { 1, -phi, 0 }, {-1,  phi, 0 }, {-1, -phi, 0 },
        { phi, 0,  1 }, {-phi, 0,  1 }, { phi, 0, -1 }, {-phi, 0, -1 }
    };

    // Row 0: the non-diffusion-weighted (b=0) measurement.
    s.bval[0] = 0.0; s.gx[0] = 0.0; s.gy[0] = 0.0; s.gz[0] = 0.0;
    for (int d = 0; d < NDIR; ++d) {
        s.bval[1 + d] = 1000.0;                 // s/mm^2 (single shell)
        s.gx[1 + d] = raw[d][0] * inv;
        s.gy[1 + d] = raw[d][1] * inv;
        s.gz[1 + d] = raw[d][2] * inv;
    }
    return s;
}

// ---------------------------------------------------------------------------
// invert_small: Gauss-Jordan inverse of an n x n matrix (row-major), n <= NPARAM.
//   Used only for the fixed 7x7 normal-equation matrix B^T B, ONCE per run, so a
//   simple, readable O(n^3) elimination with partial pivoting is perfect (no
//   need for cuSOLVER here -- the matrix is tiny and host-side). Throws if the
//   matrix is singular (would mean a degenerate gradient scheme).
// ---------------------------------------------------------------------------
static std::vector<double> invert_small(std::vector<double> A, int n) {
    std::vector<double> I(static_cast<std::size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) I[static_cast<std::size_t>(i) * n + i] = 1.0;  // identity

    for (int col = 0; col < n; ++col) {
        // Partial pivot: find the row with the largest |A[row][col]| to divide by,
        // which keeps the elimination numerically stable.
        int piv = col;
        double best = std::fabs(A[static_cast<std::size_t>(col) * n + col]);
        for (int r = col + 1; r < n; ++r) {
            double v = std::fabs(A[static_cast<std::size_t>(r) * n + col]);
            if (v > best) { best = v; piv = r; }
        }
        if (best < 1e-15) throw std::runtime_error("singular normal-equation matrix");
        if (piv != col) {   // swap rows piv <-> col in both A and I
            for (int c = 0; c < n; ++c) {
                std::swap(A[static_cast<std::size_t>(col) * n + c], A[static_cast<std::size_t>(piv) * n + c]);
                std::swap(I[static_cast<std::size_t>(col) * n + c], I[static_cast<std::size_t>(piv) * n + c]);
            }
        }
        // Normalise the pivot row so A[col][col] == 1.
        double d = A[static_cast<std::size_t>(col) * n + col];
        for (int c = 0; c < n; ++c) {
            A[static_cast<std::size_t>(col) * n + c] /= d;
            I[static_cast<std::size_t>(col) * n + c] /= d;
        }
        // Eliminate the column from every OTHER row.
        for (int r = 0; r < n; ++r) {
            if (r == col) continue;
            double f = A[static_cast<std::size_t>(r) * n + col];
            for (int c = 0; c < n; ++c) {
                A[static_cast<std::size_t>(r) * n + c] -= f * A[static_cast<std::size_t>(col) * n + c];
                I[static_cast<std::size_t>(r) * n + c] -= f * I[static_cast<std::size_t>(col) * n + c];
            }
        }
    }
    return I;
}

// ---------------------------------------------------------------------------
// (3) build_pseudo_inverse -- Minv = (B^T B)^{-1} B^T  (NPARAM x NMEAS).
//   B (NMEAS x NPARAM) encodes the linearised Stejskal-Tanner model. Row k:
//     [ 1, -b gx^2, -b gy^2, -b gz^2, -2b gx gy, -2b gx gz, -2b gy gz ]
//   so that  B * [lnS0, Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]^T  approximates  ln(S).
//   The ordinary-least-squares solution of  B d = y  is  d = (B^TB)^{-1}B^T y,
//   and since B is the SAME for every voxel we precompute Minv = (B^TB)^{-1}B^T
//   once here. On the GPU, Minv lives in constant memory and the kernel just does
//   d = Minv * y -- a fixed 7x13 matrix-vector product per voxel.
// ---------------------------------------------------------------------------
std::vector<double> build_pseudo_inverse(const GradientScheme& scheme) {
    // Assemble B (NMEAS x NPARAM), row-major.
    std::vector<double> B(static_cast<std::size_t>(NMEAS) * NPARAM, 0.0);
    for (int k = 0; k < NMEAS; ++k) {
        const double b = scheme.bval[k];
        const double gx = scheme.gx[k], gy = scheme.gy[k], gz = scheme.gz[k];
        double* row = &B[static_cast<std::size_t>(k) * NPARAM];
        row[0] = 1.0;                 // intercept -> ln(S0)
        row[1] = -b * gx * gx;        // Dxx
        row[2] = -b * gy * gy;        // Dyy
        row[3] = -b * gz * gz;        // Dzz
        row[4] = -2.0 * b * gx * gy;  // Dxy (off-diagonal counted twice)
        row[5] = -2.0 * b * gx * gz;  // Dxz
        row[6] = -2.0 * b * gy * gz;  // Dyz
    }

    // Normal-equation matrix N = B^T B  (NPARAM x NPARAM).
    std::vector<double> N(static_cast<std::size_t>(NPARAM) * NPARAM, 0.0);
    for (int i = 0; i < NPARAM; ++i)
        for (int j = 0; j < NPARAM; ++j) {
            double acc = 0.0;
            for (int k = 0; k < NMEAS; ++k)
                acc += B[static_cast<std::size_t>(k) * NPARAM + i] * B[static_cast<std::size_t>(k) * NPARAM + j];
            N[static_cast<std::size_t>(i) * NPARAM + j] = acc;
        }

    // Ninv = N^{-1}.
    std::vector<double> Ninv = invert_small(N, NPARAM);

    // Minv = Ninv * B^T  (NPARAM x NMEAS).
    std::vector<double> Minv(static_cast<std::size_t>(NPARAM) * NMEAS, 0.0);
    for (int p = 0; p < NPARAM; ++p)
        for (int k = 0; k < NMEAS; ++k) {
            double acc = 0.0;
            for (int j = 0; j < NPARAM; ++j)
                acc += Ninv[static_cast<std::size_t>(p) * NPARAM + j] * B[static_cast<std::size_t>(k) * NPARAM + j];
            Minv[static_cast<std::size_t>(p) * NMEAS + k] = acc;
        }
    return Minv;
}

// ---------------------------------------------------------------------------
// (4) fit_all_voxels_cpu -- the reference per-voxel fit.
//   A single serial loop over all voxels, each calling the shared fit_voxel()
//   (dti_core.h). This is O(nvox * NMEAS * NPARAM) and embarrassingly parallel:
//   there are NO dependencies between voxels, which is exactly why the GPU kernel
//   in kernels.cu can give each voxel its own thread.
// ---------------------------------------------------------------------------
void fit_all_voxels_cpu(const DwiVolume& vol, const std::vector<double>& Minv,
                        std::vector<VoxelResult>& out) {
    out.resize(vol.nvox);
    for (int v = 0; v < vol.nvox; ++v) {
        const double* sig = &vol.signal[static_cast<std::size_t>(v) * NMEAS];
        out[v] = fit_voxel(sig, Minv.data());   // shared host+device physics
    }
}

// ---------------------------------------------------------------------------
// (5) trace_streamlines_cpu -- reference deterministic tractography.
//   For each seed, integrate the principal-direction field with Euler steps,
//   using the shared trilinear sampler (tract_core.h) so the GPU produces the
//   same polylines. We trace in BOTH directions from the seed (forward along v1
//   and backward along -v1) and concatenate, because a fiber has no intrinsic
//   direction. Stops on: leaving the volume, FA < fa_min, or a turn sharper than
//   acos(cos_min).
// ---------------------------------------------------------------------------
static void trace_one(const DwiVolume& vol, const std::vector<VoxelResult>& fit,
                      double sx, double sy, double sz, double sign,
                      int max_steps, float step, float fa_min, float cos_min,
                      std::vector<float>& pts_out) {
    const int nx = vol.nx, ny = vol.ny, nz = vol.nz;
    double px = sx, py = sy, pz = sz;

    // Seed the reference direction from the containing voxel, oriented by `sign`.
    double rdx, rdy, rdz;
    nearest_dir(fit.data(), nx, ny, nz, px, py, pz, rdx, rdy, rdz);
    rdx *= sign; rdy *= sign; rdz *= sign;

    for (int s = 0; s < max_steps; ++s) {
        // Stop if we have wandered outside the volume.
        if (px < 0 || py < 0 || pz < 0 || px > nx - 1 || py > ny - 1 || pz > nz - 1)
            break;
        // Sample the local direction + FA (trilinear, oriented by rdx..).
        double dx, dy, dz, fa;
        sample_dir(fit.data(), nx, ny, nz, px, py, pz, rdx, rdy, rdz, dx, dy, dz, fa);
        if (fa < fa_min) break;                        // left the white matter
        // Curvature check: reject a step that turns too sharply from the last.
        const double turn = dx*rdx + dy*rdy + dz*rdz;  // cos(angle) between steps
        if (turn < cos_min) break;
        // Record the point, then take an Euler step along the (oriented) direction.
        pts_out.push_back((float)px); pts_out.push_back((float)py); pts_out.push_back((float)pz);
        px += step * dx; py += step * dy; pz += step * dz;
        rdx = dx; rdy = dy; rdz = dz;                  // this step becomes the new reference
    }
}

void trace_streamlines_cpu(const DwiVolume& vol,
                           const std::vector<VoxelResult>& fit,
                           const std::vector<float>& seeds,
                           int max_steps, float step, float fa_min, float cos_min,
                           std::vector<Streamline>& lines) {
    const int nseeds = static_cast<int>(seeds.size() / 3);
    lines.resize(nseeds);
    for (int i = 0; i < nseeds; ++i) {
        const double sx = seeds[3*i+0], sy = seeds[3*i+1], sz = seeds[3*i+2];
        std::vector<float> back, fwd;
        // Trace backward (sign -1), reverse it, then trace forward (sign +1) so
        // the polyline reads continuously from one fiber end to the other.
        trace_one(vol, fit, sx, sy, sz, -1.0, max_steps, step, fa_min, cos_min, back);
        trace_one(vol, fit, sx, sy, sz, +1.0, max_steps, step, fa_min, cos_min, fwd);
        Streamline& L = lines[i];
        // Reverse the backward half (it was traced outward from the seed).
        for (int p = (int)back.size() / 3 - 1; p >= 1; --p) {   // skip p=0 dup of seed
            L.pts.push_back(back[3*p+0]); L.pts.push_back(back[3*p+1]); L.pts.push_back(back[3*p+2]);
        }
        for (float f : fwd) L.pts.push_back(f);
        L.nsteps = static_cast<int>(L.pts.size() / 3);
    }
}
