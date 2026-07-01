// ===========================================================================
// src/reference_cpu.cpp  --  Loader, image warp, SSD, and the serial Demons
// ---------------------------------------------------------------------------
// Project 4.8 : Deformable Image Registration (reduced-scope teaching version)
//
// ROLE IN THE PROJECT
//   The trusted, obviously-correct baseline. It runs Thirion's Demons entirely
//   on the CPU in plain nested loops -- no parallelism, no cleverness -- so that
//   when the GPU field (kernels.cu) agrees with this one, we believe the GPU.
//   All the per-pixel math is shared with the GPU through demons.h, so the two
//   sides run byte-for-byte-identical formulas (see PATTERNS.md §2).
//
//   Compiled by the host C++ compiler only (no CUDA syntax here).
//
// READ THIS AFTER: demons.h, reference_cpu.h. Compare against kernels.cu (twin).
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>       // std::floor via demons.h; std::fabs
#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// load_images: parse the tiny synthetic sample.
//   Format (whitespace-separated, see data/README.md):
//     nx ny
//     F[0] F[1] ... F[nx*ny-1]        (fixed image, row-major, values in [0,1])
//     M[0] M[1] ... M[nx*ny-1]        (moving image, same layout)
//   We read everything with operator>> so newlines/spaces are interchangeable.
//   Throws on a missing file or a short/garbled body so the demo fails loudly
//   rather than silently registering an empty image.
// ---------------------------------------------------------------------------
DirImages load_images(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open image file: " + path);

    DirImages im;
    if (!(in >> im.nx >> im.ny))
        throw std::runtime_error("bad header (expected 'nx ny') in " + path);
    if (im.nx <= 2 || im.ny <= 2)
        throw std::runtime_error("image too small (need nx,ny > 2) in " + path);

    const std::size_t N = static_cast<std::size_t>(im.nx) * im.ny;
    im.fixed.resize(N);
    im.moving.resize(N);
    for (std::size_t i = 0; i < N; ++i)
        if (!(in >> im.fixed[i]))
            throw std::runtime_error("ran out of fixed-image pixels in " + path);
    for (std::size_t i = 0; i < N; ++i)
        if (!(in >> im.moving[i]))
            throw std::runtime_error("ran out of moving-image pixels in " + path);
    return im;
}

// ---------------------------------------------------------------------------
// ssd: sum of squared intensity differences. O(N), the correctness/quality
//   number the whole method exists to reduce. Done in double so the reported
//   before/after values are stable to many digits.
// ---------------------------------------------------------------------------
double ssd(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

// ---------------------------------------------------------------------------
// warp_image: Mw(x) = M(x + u(x)) for every pixel, via the shared bilinear
//   sampler. This is the same "gather" the solver does internally; we expose it
//   so main.cu can build the warped image once and report SSD-after for BOTH the
//   CPU and the GPU displacement fields.
// ---------------------------------------------------------------------------
void warp_image(const DirImages& im,
                const std::vector<double>& ux, const std::vector<double>& uy,
                std::vector<double>& warped) {
    warped.resize(static_cast<std::size_t>(im.nx) * im.ny);
    for (int y = 0; y < im.ny; ++y) {
        for (int x = 0; x < im.nx; ++x) {
            const int i = y * im.nx + x;
            // Sample the moving image at the displaced location. dm_bilinear is
            // the SAME function the GPU warp kernel calls -> identical result.
            warped[i] = dm_bilinear(im.moving.data(),
                                    (double)x + ux[i], (double)y + uy[i],
                                    im.nx, im.ny);
        }
    }
}

// ---------------------------------------------------------------------------
// register_cpu: the serial Demons solver -- the ground truth.
//
//   Data layout: we keep two displacement buffers per component so the Gaussian
//   smoothing (which reads a whole neighbourhood) never reads a half-updated
//   field. The pattern per iteration:
//
//     (A) FORCE : for every pixel, du = dm_demons_force(...); u += du.
//                 (reads the CURRENT u, writes into the same u -- safe because
//                  the force at a pixel only reads u[i] at that pixel plus the
//                  images, never neighbouring u values.)
//     (B) SMOOTH-X : ux_tmp = GaussianX(ux); uy_tmp = GaussianX(uy).
//     (C) SMOOTH-Y : ux = GaussianY(ux_tmp); uy = GaussianY(uy_tmp).
//
//   Splitting the separable Gaussian into an X pass then a Y pass, each writing
//   a fresh buffer, is what the GPU does too (two stencil kernels with ping-pong
//   buffers). Keeping the CPU structured the same way is deliberate: the code
//   reads as a direct serial mirror of the parallel version.
//
//   Complexity: O(iters * nx * ny * radius). For our sample that is tiny; for a
//   256^3 volume it is ~10^11 -- the reason DIR wants a GPU (catalog deep-dive).
// ---------------------------------------------------------------------------
void register_cpu(const DirImages& im, const DemonsParams& P,
                  std::vector<double>& ux, std::vector<double>& uy) {
    const std::size_t N = static_cast<std::size_t>(im.nx) * im.ny;

    // Displacement field, both components, initialized to zero (identity map:
    // before any iteration the "registration" is just M itself).
    ux.assign(N, 0.0);
    uy.assign(N, 0.0);

    // Scratch buffers for the two-pass separable Gaussian (see (B)/(C) above).
    std::vector<double> ux_tmp(N), uy_tmp(N);

    for (int it = 0; it < P.iters; ++it) {
        // (A) FORCE pass: accumulate the Demons update into the field.
        for (int y = 0; y < im.ny; ++y) {
            for (int x = 0; x < im.nx; ++x) {
                const int i = y * im.nx + x;
                double dux, duy;
                dm_demons_force(im.fixed.data(), im.moving.data(),
                                ux.data(), uy.data(), x, y, P, &dux, &duy);
                ux[i] += dux;   // step this pixel's displacement downhill
                uy[i] += duy;
            }
        }

        // (B) SMOOTH along X: blur ux,uy horizontally into the tmp buffers.
        for (int y = 0; y < im.ny; ++y) {
            for (int x = 0; x < im.nx; ++x) {
                const int i = y * im.nx + x;
                ux_tmp[i] = dm_gauss_x(ux.data(), x, y, im.nx, im.ny, P.sigma, P.radius);
                uy_tmp[i] = dm_gauss_x(uy.data(), x, y, im.nx, im.ny, P.sigma, P.radius);
            }
        }

        // (C) SMOOTH along Y: blur the tmp buffers vertically back into ux,uy.
        for (int y = 0; y < im.ny; ++y) {
            for (int x = 0; x < im.nx; ++x) {
                const int i = y * im.nx + x;
                ux[i] = dm_gauss_y(ux_tmp.data(), x, y, im.nx, im.ny, P.sigma, P.radius);
                uy[i] = dm_gauss_y(uy_tmp.data(), x, y, im.nx, im.ny, P.sigma, P.radius);
            }
        }
    }
}
