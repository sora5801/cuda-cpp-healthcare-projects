// ===========================================================================
// src/reference_cpu.cpp  --  Serial CPU reference: the trusted reconstruction
// ---------------------------------------------------------------------------
// Project 4.4 : Deep-Learning MRI/CT Reconstruction  (REDUCED-SCOPE TEACHING VERSION)
//
// WHAT THIS FILE DOES
//   Implements the SAME unrolled reconstruction as kernels.cu, but serially on
//   the CPU: nested loops instead of one thread per pixel. Because both call the
//   shared __host__ __device__ cores (recon_core.h, dft_core.h), the CPU and GPU
//   sums are identical (to FMA reassociation), so main.cu can VERIFY the GPU
//   against this reference within a tight tolerance.
//
//   It also owns the DATA PLUMBING: building the built-in synthetic phantom +
//   under-sampled scan, and parsing a sample file. Keeping data setup here (not
//   in main.cu) means the exact same Acquisition feeds both recon paths.
//
//   Compiled by the plain host C++ compiler (cl.exe/g++): NO CUDA syntax here.
//   The shared cores are safe to include because RECON_HD/DFT_HD expand to
//   nothing without nvcc.
//
// READ THIS AFTER: reference_cpu.h, recon_core.h, dft_core.h.
// ===========================================================================
#include "reference_cpu.h"
#include "recon_core.h"      // denoise/regularize per-pixel (shared with GPU)
#include "dft_core.h"        // forward/inverse DFT per-output (shared with GPU)
#include "util/io.hpp"       // util::read_floats

#include <cmath>             // std::sqrt
#include <stdexcept>

// ---------------------------------------------------------------------------
// forward_dft_cpu: full forward 2-D DFT of a real image into (kre,kim).
//   Serial twin of dft_forward_kernel: loops over every output frequency and
//   calls the shared reduction. O((ny*nx)^2) -- fine for our tiny teaching size.
// ---------------------------------------------------------------------------
static void forward_dft_cpu(const std::vector<float>& img, int ny, int nx,
                            std::vector<float>& kre, std::vector<float>& kim) {
    kre.assign(static_cast<std::size_t>(ny) * nx, 0.0f);
    kim.assign(static_cast<std::size_t>(ny) * nx, 0.0f);
    for (int v = 0; v < ny; ++v)
        for (int u = 0; u < nx; ++u) {
            float re, im;
            dft_forward_pixel(img.data(), v, u, ny, nx, &re, &im);
            const std::size_t k = kidx(v, u, nx);
            kre[k] = re; kim[k] = im;
        }
}

// ---------------------------------------------------------------------------
// inverse_dft_cpu: full inverse 2-D DFT of (kre,kim) into a real image.
//   Serial twin of dc_idft_kernel's transform half.
// ---------------------------------------------------------------------------
static void inverse_dft_cpu(const std::vector<float>& kre, const std::vector<float>& kim,
                            int ny, int nx, std::vector<float>& img) {
    img.assign(static_cast<std::size_t>(ny) * nx, 0.0f);
    for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x)
            img[img_idx(y, x, nx)] = idft_pixel(kre.data(), kim.data(), y, x, ny, nx);
}

// ---------------------------------------------------------------------------
// make_synthetic_acquisition: build the phantom + under-sampled scan.
//   The phantom is a PIECEWISE-CONSTANT image (a Shepp-Logan-in-spirit toy):
//     * dim background (0.1)
//     * a bright disk (1.0) centered in the image
//     * a mid-gray square (0.6) offset to one corner
//   We forward-DFT the phantom to get its FULL k-space, then UNDER-SAMPLE: keep
//   a central low-frequency block (always) plus every other line elsewhere. This
//   is a caricature of an MRI acceleration mask. The kept bins become kmeas;
//   skipped bins are zeroed. All values are synthetic -- see data/README.md.
// ---------------------------------------------------------------------------
Acquisition make_synthetic_acquisition(int ny, int nx) {
    Acquisition acq;
    acq.ny = ny; acq.nx = nx;
    const std::size_t N = static_cast<std::size_t>(ny) * nx;
    acq.truth.assign(N, 0.1f);   // dim background

    // Geometry of the two features, in pixel coordinates.
    const float cy = (ny - 1) * 0.5f, cx = (nx - 1) * 0.5f;   // image center
    const float disk_r = 0.30f * nx;                          // disk radius
    for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
            const float dy = y - cy, dx = x - cx;
            if (dy * dy + dx * dx <= disk_r * disk_r)
                acq.truth[img_idx(y, x, nx)] = 1.0f;           // bright disk
            // A mid-gray square in the upper-left quadrant.
            if (y >= ny / 6 && y < ny / 6 + ny / 4 &&
                x >= nx / 6 && x < nx / 6 + nx / 4)
                acq.truth[img_idx(y, x, nx)] = 0.6f;           // gray square
        }

    // Full k-space of the phantom.
    std::vector<float> fre, fim;
    forward_dft_cpu(acq.truth, ny, nx, fre, fim);

    // Under-sampling mask. Our DFT is UN-SHIFTED, so the energy-dense low
    // frequencies sit near indices 0 and ny-1/nx-1 (DC wraps). Keep: a narrow band
    // of the lowest frequency lines on each edge (always), plus every 2nd line
    // elsewhere. This retains ~40% of the samples (~2.5x acceleration) and leaves
    // enough aliasing that the denoise+data-consistency iteration measurably
    // improves the zero-filled image (see README "Expected output").
    acq.mask.assign(N, 0);
    acq.kmeas_re.assign(N, 0.0f);
    acq.kmeas_im.assign(N, 0.0f);
    const int lowband = ny / 8;   // how many low-frequency lines to always keep
    for (int v = 0; v < ny; ++v) {
        const bool v_low = (v < lowband) || (v >= ny - lowband);   // near DC in v
        const bool v_keep = v_low || (v % 2 == 0);                 // else every 2nd
        for (int u = 0; u < nx; ++u) {
            const bool u_low = (u < lowband) || (u >= nx - lowband);
            const bool keep = v_keep && (u_low || (u % 2 == 0));
            if (keep) {
                const std::size_t k = kidx(v, u, nx);
                acq.mask[k] = 1;
                acq.kmeas_re[k] = fre[k];   // measured value at a sampled bin
                acq.kmeas_im[k] = fim[k];
            }
        }
    }
    return acq;
}

// ---------------------------------------------------------------------------
// load_acquisition: parse a sample file into an Acquisition.
//   Layout (whitespace-separated floats; see data/README.md):
//     ny nx
//     truth[0 .. N-1]            (N = ny*nx)
//     mask[0 .. N-1]            (0/1)
//     kmeas_re[0 .. N-1]
//     kmeas_im[0 .. N-1]
//   Returns false if the file is missing or too short.
// ---------------------------------------------------------------------------
bool load_acquisition(const std::string& path, Acquisition& acq) {
    std::vector<float> v;
    try { v = util::read_floats(path); }
    catch (const std::exception&) { return false; }        // no file -> fall back
    if (v.size() < 2) return false;
    const int ny = static_cast<int>(v[0]);
    const int nx = static_cast<int>(v[1]);
    if (ny <= 0 || nx <= 0) return false;
    const std::size_t N = static_cast<std::size_t>(ny) * nx;
    if (v.size() < 2 + 4 * N) return false;                // need 4 arrays of N
    acq.ny = ny; acq.nx = nx;
    std::size_t off = 2;
    acq.truth.assign(v.begin() + off, v.begin() + off + N); off += N;
    acq.mask.resize(N);
    for (std::size_t i = 0; i < N; ++i) acq.mask[i] = static_cast<int>(v[off + i]);
    off += N;
    acq.kmeas_re.assign(v.begin() + off, v.begin() + off + N); off += N;
    acq.kmeas_im.assign(v.begin() + off, v.begin() + off + N);
    return true;
}

// ---------------------------------------------------------------------------
// recon_cpu: the serial unrolled reconstruction (mirror of recon_gpu).
//   Steps exactly match kernels.cu so the two results agree:
//     0. zero-filled iDFT of the measured k-space   -> initial image estimate
//     for each stage:
//       A. denoise (regularize) every pixel          (recon_core.h)
//       B. forward DFT of the denoised image          (dft_core.h)
//       C1. data consistency: overwrite sampled bins with the measurement
//       C2. inverse DFT of the data-consistent k-space -> new estimate
// ---------------------------------------------------------------------------
void recon_cpu(const Acquisition& acq, const ReconParams& p,
               std::vector<float>& recon) {
    const int ny = acq.ny, nx = acq.nx;
    const std::size_t N = static_cast<std::size_t>(ny) * nx;

    // 0. Zero-filled reconstruction as the starting image.
    std::vector<float> kre = acq.kmeas_re;   // estimate k-space starts = measured
    std::vector<float> kim = acq.kmeas_im;   //   (0 where unsampled)
    std::vector<float> img;
    inverse_dft_cpu(kre, kim, ny, nx, img);

    std::vector<float> den(N);               // scratch: denoised image per stage
    for (int s = 0; s < p.stages; ++s) {
        // A. denoiser step, pixel by pixel (identical to regularize_kernel).
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                const float self = img[img_idx(y, x, nx)];
                den[img_idx(y, x, nx)] =
                    regularize_pixel(img.data(), self, y, x, ny, nx, p.lambda);
            }
        // B. forward transform of the denoised image.
        forward_dft_cpu(den, ny, nx, kre, kim);
        // C1. data consistency: sampled bins snap back to the measurement.
        for (std::size_t k = 0; k < N; ++k)
            if (acq.mask[k]) { kre[k] = acq.kmeas_re[k]; kim[k] = acq.kmeas_im[k]; }
        // C2. inverse transform -> next image estimate.
        inverse_dft_cpu(kre, kim, ny, nx, img);
    }
    recon = img;   // final estimate
}

// ---------------------------------------------------------------------------
// rms_error: sqrt(mean((a-b)^2)) over two equal-length images. Our science-level
//   score. Returns a huge sentinel on a length mismatch so a shape bug cannot be
//   mistaken for a good reconstruction.
// ---------------------------------------------------------------------------
double rms_error(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 1e30;
    double acc = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        acc += d * d;
    }
    return std::sqrt(acc / static_cast<double>(a.size()));
}
