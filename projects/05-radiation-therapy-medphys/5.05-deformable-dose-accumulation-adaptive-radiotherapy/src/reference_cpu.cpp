// ===========================================================================
// src/reference_cpu.cpp  --  Loader, serial Demons DIR, serial dose warp + DVH
// ---------------------------------------------------------------------------
// Project 5.5 : Deformable Dose Accumulation & Adaptive Radiotherapy
//               (reduced-scope 2-D teaching version)
//
// ROLE IN THE PROJECT
//   The trusted, obviously-correct baseline. It runs the whole ART mini-pipeline
//   -- register daily->planning, warp the delivered dose by the DVF, accumulate,
//   and histogram -- entirely on the CPU in plain nested loops (no parallelism,
//   no cleverness). When the GPU (kernels.cu) agrees with this, we believe the
//   GPU. All per-voxel math is shared with the GPU through demons.h and dose.h,
//   so both sides run byte-for-byte-identical formulas (PATTERNS.md §2).
//
//   Compiled by the host C++ compiler only (no CUDA syntax here).
//
// READ THIS AFTER: demons.h, dose.h, reference_cpu.h. Compare vs kernels.cu (twin).
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>       // std::fabs (via headers); std::floor lives in demons.h
#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// read_grid (file-local helper): read exactly N doubles into `dst`, or throw.
//   Keeps load_case tidy: it must read four equally-sized grids in sequence, and
//   a short/garbled file should fail loudly rather than leave a grid half-filled.
// ---------------------------------------------------------------------------
static void read_grid(std::ifstream& in, std::vector<double>& dst,
                      std::size_t N, const char* what, const std::string& path) {
    dst.resize(N);
    for (std::size_t i = 0; i < N; ++i)
        if (!(in >> dst[i]))
            throw std::runtime_error(std::string("ran out of ") + what +
                                     " values in " + path);
}

// ---------------------------------------------------------------------------
// load_case: parse the tiny synthetic sample.
//   Format (whitespace-separated, see data/README.md):
//     nx ny
//     plan_img   [nx*ny]   (planning anatomy,  FIXED image,  values in [0,1])
//     daily_img  [nx*ny]   (today's anatomy,   MOVING image, values in [0,1])
//     plan_dose  [nx*ny]   (intended dose on the planning grid, Gy)
//     daily_dose [nx*ny]   (delivered dose on today's grid,     Gy)
//   operator>> makes newlines and spaces interchangeable. Throws on a missing
//   file or a short body so the demo fails loudly instead of running on garbage.
// ---------------------------------------------------------------------------
ArtCase load_case(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open case file: " + path);

    ArtCase c;
    if (!(in >> c.nx >> c.ny))
        throw std::runtime_error("bad header (expected 'nx ny') in " + path);
    if (c.nx <= 2 || c.ny <= 2)
        throw std::runtime_error("grid too small (need nx,ny > 2) in " + path);

    const std::size_t N = static_cast<std::size_t>(c.nx) * c.ny;
    read_grid(in, c.plan_img,   N, "plan_img",   path);
    read_grid(in, c.daily_img,  N, "daily_img",  path);
    read_grid(in, c.plan_dose,  N, "plan_dose",  path);
    read_grid(in, c.daily_dose, N, "daily_dose", path);
    return c;
}

// ---------------------------------------------------------------------------
// register_cpu: the serial Demons solver -- the ground-truth DVF.
//
//   Data layout: two displacement buffers per component so the Gaussian
//   smoothing (which reads a whole neighbourhood) never reads a half-updated
//   field. Per iteration:
//     (A) FORCE   : for every voxel, du = dm_demons_force(...); u += du.
//                   (reads the CURRENT u, writes the same u -- safe because the
//                    force at a voxel only reads u[i] at THAT voxel plus the
//                    images, never neighbouring u values.)
//     (B) SMOOTH-X: ux_tmp = GaussianX(ux); uy_tmp = GaussianX(uy).
//     (C) SMOOTH-Y: ux = GaussianY(ux_tmp); uy = GaussianY(uy_tmp).
//
//   Splitting the separable Gaussian into an X pass then a Y pass, each writing a
//   fresh buffer, is exactly what the GPU does (two stencil kernels with ping-pong
//   buffers). Keeping the CPU structured the same way is deliberate: it reads as a
//   direct serial mirror of the parallel kernels.
//
//   Complexity: O(iters * nx * ny * radius). Tiny for our sample; ~10^11 for a
//   256^3 volume -- the reason ART wants a GPU (catalog deep-dive).
// ---------------------------------------------------------------------------
void register_cpu(const ArtCase& c, const DemonsParams& P,
                  std::vector<double>& ux, std::vector<double>& uy) {
    const std::size_t N = static_cast<std::size_t>(c.nx) * c.ny;

    // Displacement field, both components, zeroed (identity map: before any
    // iteration the "registration" is just the daily image itself).
    ux.assign(N, 0.0);
    uy.assign(N, 0.0);

    // Scratch buffers for the two-pass separable Gaussian (see (B)/(C) above).
    std::vector<double> ux_tmp(N), uy_tmp(N);

    for (int it = 0; it < P.iters; ++it) {
        // (A) FORCE pass: accumulate the Demons update into the field. daily_img
        //     is the MOVING image (M), plan_img is the FIXED image (F).
        for (int y = 0; y < c.ny; ++y) {
            for (int x = 0; x < c.nx; ++x) {
                const int i = y * c.nx + x;
                double dux, duy;
                dm_demons_force(c.plan_img.data(), c.daily_img.data(),
                                ux.data(), uy.data(), x, y, P, &dux, &duy);
                ux[i] += dux;   // step this voxel's displacement downhill
                uy[i] += duy;
            }
        }

        // (B) SMOOTH along X: blur ux,uy horizontally into the tmp buffers.
        for (int y = 0; y < c.ny; ++y) {
            for (int x = 0; x < c.nx; ++x) {
                const int i = y * c.nx + x;
                ux_tmp[i] = dm_gauss_x(ux.data(), x, y, c.nx, c.ny, P.sigma, P.radius);
                uy_tmp[i] = dm_gauss_x(uy.data(), x, y, c.nx, c.ny, P.sigma, P.radius);
            }
        }

        // (C) SMOOTH along Y: blur the tmp buffers vertically back into ux,uy.
        for (int y = 0; y < c.ny; ++y) {
            for (int x = 0; x < c.nx; ++x) {
                const int i = y * c.nx + x;
                ux[i] = dm_gauss_y(ux_tmp.data(), x, y, c.nx, c.ny, P.sigma, P.radius);
                uy[i] = dm_gauss_y(uy_tmp.data(), x, y, c.nx, c.ny, P.sigma, P.radius);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// warp_dose_cpu: deformable dose warp (serial). For every planning-frame voxel,
//   gather the delivered daily dose at the deformed location via warp_dose_at
//   (bilinear, shared with the GPU). This is one fraction's "deformed dose",
//   ready to be summed into the total. O(N) gathers, each O(1).
// ---------------------------------------------------------------------------
void warp_dose_cpu(const ArtCase& c,
                   const std::vector<double>& ux, const std::vector<double>& uy,
                   std::vector<double>& warped) {
    const std::size_t N = static_cast<std::size_t>(c.nx) * c.ny;
    warped.resize(N);
    for (int y = 0; y < c.ny; ++y) {
        for (int x = 0; x < c.nx; ++x) {
            const int i = y * c.nx + x;
            // warp_dose_at samples c.daily_dose at (x+ux, y+uy) -- the SAME
            // function the GPU warp kernel calls, so the result is identical.
            warped[i] = warp_dose_at(c.daily_dose.data(),
                                     ux.data(), uy.data(), x, y, c.nx, c.ny);
        }
    }
}

// ---------------------------------------------------------------------------
// accumulate_cpu: total += add, voxel by voxel. Summation of deformed doses in
//   the common planning frame. If total is empty we size+zero it first so the
//   caller can accumulate several fractions with the same routine.
// ---------------------------------------------------------------------------
void accumulate_cpu(std::vector<double>& total, const std::vector<double>& add) {
    if (total.empty()) total.assign(add.size(), 0.0);
    for (std::size_t i = 0; i < add.size(); ++i)
        total[i] += add[i];
}

// ---------------------------------------------------------------------------
// build_dvh_cpu: differential dose-volume histogram. One integer count per voxel,
//   placed with dvh_bin (shared with the GPU). Integer counts commute, so this is
//   bit-for-bit reproducible and directly comparable to the GPU's atomic-int
//   histogram (PATTERNS.md §3). O(N).
// ---------------------------------------------------------------------------
std::vector<unsigned> build_dvh_cpu(const std::vector<double>& dose) {
    std::vector<unsigned> hist(DVH_BINS, 0u);
    for (double d : dose)
        hist[dvh_bin(d)] += 1u;
    return hist;
}

// ---------------------------------------------------------------------------
// dose_sum / dose_max: deterministic scalar summaries. dose_sum is a proxy for
//   total deposited energy (voxel-volume-weighted dose, but our voxels are unit,
//   so it is just the sum); dose_max is the hot-spot. Both in double for stable
//   digits in stdout. O(N).
// ---------------------------------------------------------------------------
double dose_sum(const std::vector<double>& dose) {
    double s = 0.0;
    for (double d : dose) s += d;
    return s;
}

double dose_max(const std::vector<double>& dose) {
    double m = 0.0;
    for (double d : dose) if (d > m) m = d;
    return m;
}
