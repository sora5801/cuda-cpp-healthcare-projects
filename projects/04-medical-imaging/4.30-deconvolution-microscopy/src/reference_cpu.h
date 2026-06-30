// ===========================================================================
// src/reference_cpu.h  --  Image model, PSF, and the CPU Richardson-Lucy ref
// ---------------------------------------------------------------------------
// Project 4.30 : Deconvolution Microscopy
//
// WHAT THIS PROJECT COMPUTES
//   A fluorescence microscope blurs the true image of a specimen by convolving
//   it with the optical Point Spread Function (PSF) -- the image of a single
//   point of light. Deconvolution tries to INVERT that blur: given the blurry
//   image and the (known) PSF, recover a sharper estimate of the true image.
//
//   We implement Richardson-Lucy (RL) deconvolution, the workhorse iterative
//   deconvolution algorithm for Poisson (photon-limited) data. Each RL
//   iteration needs TWO convolutions; on the GPU those convolutions are done
//   with cuFFT (the convolution theorem: convolution in space == pointwise
//   multiply in frequency). This CPU reference does the IDENTICAL circular
//   convolution directly in space -- slow but obviously correct -- so we can
//   verify the cuFFT-based GPU result.
//
// WHY A GPU / cuFFT
//   Real microscopy volumes reach 2048^3 voxels and RL runs 20-100 iterations,
//   so the convolutions dominate. Direct convolution is O(N * K) per image
//   (N pixels, K PSF taps); FFT convolution is O(N log N) and the GPU does the
//   FFT in milliseconds. This project's lesson is USING cuFFT WITHOUT IT BEING
//   A BLACK BOX (kernels.cu documents exactly what each cuFFT call computes).
//
//   This is a pure C++ header (no CUDA): kernels.cu / main.cu reuse the Image
//   and Psf structs and the same make_gaussian_psf(), so both paths share
//   inputs.
//
// READ THIS BEFORE: rl_core.h (the shared per-pixel math), kernels.cuh.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

// A single-channel 2-D intensity image stored row-major.
//   pix[y * w + x] = intensity at column x, row y   (x in [0,w), y in [0,h)).
// We use double throughout the RL math (many iterations -> precision matters).
struct Image {
    int w = 0;                  // width  in pixels (columns)
    int h = 0;                  // height in pixels (rows)
    std::vector<double> pix;    // size w*h, row-major
    int size() const { return w * h; }
};

// The Point Spread Function: a small odd-sized kernel (the microscope's blur).
//   k[(dy+r) * d + (dx+r)] = weight for offset (dx,dy), dx,dy in [-r, r].
// `d = 2*r + 1` is the side length; the PSF is normalized to sum to 1 so that
// convolution conserves total intensity (a blur must not create/destroy light).
struct Psf {
    int r = 0;                  // radius (taps extend +/- r in x and y)
    int d() const { return 2 * r + 1; }   // side length
    std::vector<double> k;      // size d*d, row-major, sums to 1
};

// ---- I/O ------------------------------------------------------------------

// Load the blurred (observed) image from the text sample format (data/README):
//   header line: "<w> <h>"
//   then h rows of w whitespace-separated non-negative doubles.
// Throws std::runtime_error on a malformed/missing file so demos fail loudly.
Image load_image(const std::string& path);

// ---- PSF ------------------------------------------------------------------

// Build a normalized 2-D Gaussian PSF of radius r and standard deviation sigma
// (in pixels). This is the canonical model of diffraction-limited blur; a real
// project would instead MEASURE the PSF from sub-resolution fluorescent beads
// (see THEORY.md "real world"). Deterministic -> identical PSF on CPU and GPU.
Psf make_gaussian_psf(int r, double sigma);

// ---- Circular convolution (the shared building block) ---------------------

// Circular (periodic) 2-D convolution of an image with the PSF, computed
// DIRECTLY in space with wrap-around indexing. "Circular" means index y+dy is
// taken modulo h (and x+dx modulo w): the image is treated as if it tiles the
// plane. We deliberately use CIRCULAR (not zero-padded) convolution because it
// is EXACTLY what an FFT computes -- so the CPU reference here and the cuFFT
// path in kernels.cu implement the *same* mathematical operator and can be
// compared pixel-for-pixel.
//
//   src   : input image (w x h)
//   psf   : the kernel
//   flip  : if true, correlate with the 180-degree-flipped PSF instead (the RL
//           "back-projection" step uses the flipped/adjoint PSF). For a
//           symmetric Gaussian flip changes nothing, but we implement it
//           honestly so the code generalizes to asymmetric PSFs.
//   out   : output image (w x h), overwritten.
void convolve_circular(const Image& src, const Psf& psf, bool flip, Image& out);

// ---- The CPU Richardson-Lucy reference ------------------------------------

// Run `iters` Richardson-Lucy iterations on `observed` with PSF `psf`, starting
// from a flat (mean-valued) estimate, and return the deconvolved estimate.
// This is the trusted baseline the GPU (cuFFT) result is checked against. It
// uses convolve_circular() for both convolutions and the shared rl_ratio() /
// rl_update() per-pixel math from rl_core.h.
Image richardson_lucy_cpu(const Image& observed, const Psf& psf, int iters);

// ---- A scalar summary used for the deterministic report -------------------

// Sharpness proxy: the mean squared gradient magnitude of an image (a simple,
// monotone "how sharp / high-frequency is this" number). Deconvolution should
// INCREASE it relative to the blurry input. Reported for blurry vs deconvolved
// so the learner sees the algorithm did something, in one deterministic number.
double sharpness(const Image& img);
