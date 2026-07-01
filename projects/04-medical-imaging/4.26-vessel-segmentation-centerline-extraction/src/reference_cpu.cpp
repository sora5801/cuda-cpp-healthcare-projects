// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial Frangi vesselness reference
// ---------------------------------------------------------------------------
// Project 4.26 : Vessel Segmentation & Centerline Extraction
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against: obviously-correct,
//   single-threaded loops, no cleverness. The per-voxel math is the SHARED
//   frangi.h, so the only difference between this and the GPU kernel is who runs
//   the loop -- which is exactly what lets us verify the GPU to ~1e-9.
//
//   Compiled by the host C++ compiler only (no CUDA). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, frangi.h.  Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>       // std::exp, std::sqrt, std::round
#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// load_volume: parse the header line then nx*ny*nz float intensities.
// ---------------------------------------------------------------------------
VesselJob load_volume(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open volume file: " + path);

    VesselJob job;
    int bright = 1;
    // Header: nx ny nz sigma alpha beta c bright mask_threshold
    if (!(in >> job.vol.nx >> job.vol.ny >> job.vol.nz
             >> job.fp.sigma >> job.fp.alpha >> job.fp.beta >> job.fp.c
             >> bright >> job.mask_threshold)) {
        throw std::runtime_error("bad header (expected 'nx ny nz sigma alpha "
                                 "beta c bright mask_threshold') in " + path);
    }
    job.fp.bright_vessels = bright;
    if (job.vol.nx <= 2 || job.vol.ny <= 2 || job.vol.nz <= 2)
        throw std::runtime_error("volume too small (need >2 in each dim) in " + path);
    if (job.fp.sigma <= 0.0)
        throw std::runtime_error("sigma must be > 0 in " + path);

    const std::size_t n = job.vol.size();
    job.vol.data.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (!(in >> job.vol.data[i]))
            throw std::runtime_error("volume truncated (fewer than nx*ny*nz "
                                     "voxels) in " + path);
    }
    return job;
}

// ---------------------------------------------------------------------------
// clampi: clamp an index to [0, hi] (clamp-to-edge border handling). Used by
//   both the smoother and the finite-difference Hessian so out-of-range
//   neighbours are the nearest in-range voxel (no wraparound, no zeros).
// ---------------------------------------------------------------------------
static inline int clampi(int v, int hi) {
    return v < 0 ? 0 : (v > hi ? hi : v);
}

// ---------------------------------------------------------------------------
// gaussian_smooth: separable 3-D Gaussian at scale sigma, clamp-to-edge border.
//   A 3-D Gaussian G(x,y,z) factorizes as g(x)*g(y)*g(z), so we convolve with
//   the 1-D kernel g three times (once per axis) using two ping-pong buffers.
//   Cost: O(N * (2r+1) * 3) instead of O(N * (2r+1)^3) for a naive 3-D kernel.
// ---------------------------------------------------------------------------
void gaussian_smooth(const Volume& in, double sigma, Volume& out) {
    const int nx = in.nx, ny = in.ny, nz = in.nz;
    // Kernel radius = 3 sigma (captures >99% of the Gaussian mass), >=1.
    const int r = std::max(1, (int)std::round(3.0 * sigma));
    // Build and L1-normalize the 1-D weights so the blur preserves total mass.
    std::vector<double> g(2 * r + 1);
    double gsum = 0.0;
    for (int k = -r; k <= r; ++k) {
        const double w = std::exp(-(double)k * k / (2.0 * sigma * sigma));
        g[k + r] = w; gsum += w;
    }
    for (double& w : g) w /= gsum;

    // Ping-pong between two float buffers as we sweep the three axes.
    std::vector<float> a = in.data;                 // start from the raw image
    std::vector<float> b(in.size(), 0.0f);

    // --- Pass 1: convolve along x -----------------------------------------
    for (int z = 0; z < nz; ++z)
      for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
            double acc = 0.0;
            for (int k = -r; k <= r; ++k)
                acc += g[k + r] * a[vox_idx(clampi(x + k, nx - 1), y, z, nx, ny)];
            b[vox_idx(x, y, z, nx, ny)] = (float)acc;
        }
    // --- Pass 2: convolve along y -----------------------------------------
    for (int z = 0; z < nz; ++z)
      for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
            double acc = 0.0;
            for (int k = -r; k <= r; ++k)
                acc += g[k + r] * b[vox_idx(x, clampi(y + k, ny - 1), z, nx, ny)];
            a[vox_idx(x, y, z, nx, ny)] = (float)acc;
        }
    // --- Pass 3: convolve along z -----------------------------------------
    out.nx = nx; out.ny = ny; out.nz = nz;
    out.data.assign(in.size(), 0.0f);
    for (int z = 0; z < nz; ++z)
      for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
            double acc = 0.0;
            for (int k = -r; k <= r; ++k)
                acc += g[k + r] * a[vox_idx(x, y, clampi(z + k, nz - 1), nx, ny)];
            out.data[vox_idx(x, y, z, nx, ny)] = (float)acc;
        }
}

// ---------------------------------------------------------------------------
// hessian_at: the six unique second derivatives of the smoothed image at (x,y,z)
//   by CENTRAL finite differences (clamp-to-edge). This helper is also mirrored
//   verbatim inside the GPU kernel so both compute the same Hessian.
//     Hxx = f(x+1)-2f(x)+f(x-1)          (second derivative along x)
//     Hxy = (f(x+1,y+1)-f(x-1,y+1)-f(x+1,y-1)+f(x-1,y-1)) / 4   (mixed)
//   ...and analogously for the other axes. Voxel spacing is taken as 1.
// ---------------------------------------------------------------------------
static inline void hessian_at(const std::vector<float>& v, int x, int y, int z,
                              int nx, int ny, int nz,
                              double& h00, double& h11, double& h22,
                              double& h01, double& h02, double& h12) {
    auto V = [&](int xi, int yi, int zi) -> double {
        return v[vox_idx(clampi(xi, nx - 1), clampi(yi, ny - 1),
                         clampi(zi, nz - 1), nx, ny)];
    };
    const double c = V(x, y, z);
    // Pure second derivatives (the diagonal of H).
    h00 = V(x + 1, y, z) - 2.0 * c + V(x - 1, y, z);   // d^2/dx^2
    h11 = V(x, y + 1, z) - 2.0 * c + V(x, y - 1, z);   // d^2/dy^2
    h22 = V(x, y, z + 1) - 2.0 * c + V(x, y, z - 1);   // d^2/dz^2
    // Mixed partials (the off-diagonal, symmetric entries).
    h01 = (V(x + 1, y + 1, z) - V(x - 1, y + 1, z)
         - V(x + 1, y - 1, z) + V(x - 1, y - 1, z)) * 0.25;   // d^2/dxdy
    h02 = (V(x + 1, y, z + 1) - V(x - 1, y, z + 1)
         - V(x + 1, y, z - 1) + V(x - 1, y, z - 1)) * 0.25;   // d^2/dxdz
    h12 = (V(x, y + 1, z + 1) - V(x, y - 1, z + 1)
         - V(x, y + 1, z - 1) + V(x, y - 1, z - 1)) * 0.25;   // d^2/dydz
}

void vesselness_cpu(const Volume& s, const FrangiParams& fp,
                    std::vector<float>& vness) {
    vness.assign(s.size(), 0.0f);
    for (int z = 0; z < s.nz; ++z)
      for (int y = 0; y < s.ny; ++y)
        for (int x = 0; x < s.nx; ++x) {
            // 1) local Hessian of the smoothed intensity
            double h00, h11, h22, h01, h02, h12;
            hessian_at(s.data, x, y, z, s.nx, s.ny, s.nz,
                       h00, h11, h22, h01, h02, h12);
            // 2) eigenvalues (ascending by value)
            double e0, e1, e2;
            eig_sym3(h00, h01, h02, h11, h12, h22, e0, e1, e2);
            // 3) sort by magnitude (Frangi convention) then score
            double l1 = e0, l2 = e1, l3 = e2;
            sort_abs3(l1, l2, l3);
            vness[vox_idx(x, y, z, s.nx, s.ny)] =
                (float)frangi_response(l1, l2, l3, fp);
        }
}

// ---------------------------------------------------------------------------
// summarize: deterministic reduction of the vesselness field.
// ---------------------------------------------------------------------------
void summarize(const Volume& d, const std::vector<float>& vness,
               double threshold,
               long long& n_vessel, double& vsum,
               int& px, int& py, int& pz, double& pmax) {
    n_vessel = 0; vsum = 0.0; pmax = -1.0; px = py = pz = 0;
    // Fixed row-major scan; the FIRST voxel that attains the max wins ties, so
    // the reported peak location is deterministic regardless of platform.
    for (int z = 0; z < d.nz; ++z)
      for (int y = 0; y < d.ny; ++y)
        for (int x = 0; x < d.nx; ++x) {
            const double v = vness[vox_idx(x, y, z, d.nx, d.ny)];
            vsum += v;
            if (v >= threshold) ++n_vessel;
            if (v > pmax) { pmax = v; px = x; py = y; pz = z; }
        }
}
